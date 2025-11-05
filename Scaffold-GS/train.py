import os
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel,NeRAFAudioSoundField
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from torch.utils.data import DataLoader
from arguments import RAFDataParserConfig, RAFDataParser, RAFDataset
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
import torch.nn as nn
import torch.nn.functional as F

# NeRAF STFTLoss implementation
class STFTLoss(nn.Module):
    """STFT Loss implementation matching NeRAF"""
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, pred, gt):
        # pred, gt are log-magnitude STFT slices [B, F]
        if self.loss_type == 'mse':
            # Spectral Convergence Loss (在magnitude域)
            # 确保幅度谱不为负
            pred_mag = torch.clamp(torch.exp(pred) - 1e-3, min=1e-6)
            gt_mag = torch.clamp(torch.exp(gt) - 1e-3, min=1e-6)
            
            sc_loss = torch.norm(gt_mag - pred_mag, p="fro") / torch.norm(gt_mag, p="fro")
            mag_loss = torch.nn.functional.mse_loss(pred, gt)
            
            return {
                'audio_sc_loss': sc_loss,
                'audio_mag_loss': mag_loss
            }
        else:  # l1
            # 确保幅度谱不为负
            pred_mag = torch.clamp(torch.exp(pred) - 1e-3, min=1e-6)
            gt_mag = torch.clamp(torch.exp(gt) - 1e-3, min=1e-6)
            
            sc_loss = torch.norm(gt_mag - pred_mag, p="fro") / torch.norm(gt_mag, p="fro")
            mag_loss = torch.nn.functional.l1_loss(pred, gt)
            
            return {
                'audio_sc_loss': sc_loss,
                'audio_mag_loss': mag_loss
            }

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    print("\n\nsaving_iterations: ", saving_iterations)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    #高斯模型和场景
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    
    #参数
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    #添加raf dataset setup
        # ---------------- Audio modules (persistent) ----------------
    # encodings
    time_enc = NeRFEncoding(in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
    pos_enc = NeRFEncoding(in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
    rot_enc = SHEncoding(levels=4, implementation="tcnn")

    time_dim = time_enc.get_out_dim()
    pos_dim = pos_enc.get_out_dim()
    rot_dim = rot_enc.get_out_dim()



    # model heads
    audio_W = 512
    audio_F = 513  # RAF fs=48k -> 513 bins for n_fft=1024
    

    
    grid_feat_dim = 70  # 局部网格特征维度：9x9x9扬声器 + 9x9x9听者 = 70通道池化后
    in_size = grid_feat_dim + time_dim + 2 * pos_dim + rot_dim
    logger.info(f"[Audio] Grid feature dimension: {grid_feat_dim} (local pooled features)")
    logger.info(f"[Audio] Audio network input size: {in_size}")
    audio_field = NeRAFAudioSoundField(in_size=in_size, W=audio_W, sound_rez=1, N_frequencies=audio_F).cuda()

    # optimizer for audio only - 参考NeRAF配置
    audio_lr_init = 1e-4
    audio_lr_final = 1e-8  # NeRAF使用更小的最终学习率
    audio_lr_delay_mult = 0.01
    audio_lr_max_steps = 200000  
    start_step_audio = 2000  
    
    # 创建音频学习率调度器 - 基于声场训练步数而不是iteration
    from utils.general_utils import get_expon_lr_func
    audio_scheduler_args = get_expon_lr_func(
        lr_init=audio_lr_init,
        lr_final=audio_lr_final,
        lr_delay_steps=start_step_audio,  
        lr_delay_mult=audio_lr_delay_mult,
        max_steps=audio_lr_max_steps + start_step_audio  # 总步数包含延迟步数
    )
    
    # 添加全局音频步数计数器
    global_audio_step = 0
    
    # 将local_pooling_net加入音频优化器（独立参数组，较小学习率）
    param_groups = [
        {
            'params': list(audio_field.parameters())
                      + list(time_enc.parameters())
                      + list(pos_enc.parameters())
                      + list(rot_enc.parameters()),
            'lr': audio_lr_init
        }
    ]
    if hasattr(gaussians, 'local_pooling_net') and gaussians.local_pooling_net is not None:
        param_groups.append({
            'params': gaussians.local_pooling_net.parameters(),
            'lr': audio_lr_init 
        })

    optimizer_audio = torch.optim.Adam(param_groups, lr=audio_lr_init, eps=1e-15)
    criterion_audio = STFTLoss(loss_type='mse')  # 使用NeRAF的STFTLoss

    # ---------------- RAF dataset (optional) setup ----------------
    raf_loader = None
    raf_eval_loader = None
    raf_iter = None
    raf_eval_iter = None


    # 修改数据加载器创建部分
    try:
        raf_data_root = getattr(dataset, 'raf_data', '') if hasattr(dataset, 'raf_data') else ''
        raf_fs = int(getattr(dataset, 'raf_fs', 48000))
        raf_max_len_s = float(getattr(dataset, 'raf_max_len', 0.32))
        
        if isinstance(raf_data_root, str) and len(raf_data_root) > 0 and os.path.isdir(raf_data_root):
            logger.info(f"[RAF] Using raf_data={raf_data_root}, fs={raf_fs}, max_len_s={raf_max_len_s}")
            dp_cfg = RAFDataParserConfig(data=Path(raf_data_root))
            dp = RAFDataParser(dp_cfg)
            dpo_train = dp.get_dataparser_outputs(split="train")
            dpo_test  = dp.get_dataparser_outputs(split="test")
            hop_len = 256 if raf_fs == 48000 else 128
            max_len_frames = max(1, int(raf_max_len_s * raf_fs / hop_len))
            
            raf_train_ds = RAFDataset(
                dataparser_outputs=dpo_train,
                mode='train',
                max_len=max_len_frames,
                max_len_time=raf_max_len_s,
                wav_path=os.path.join(raf_data_root, 'data'),
                fs=raf_fs, hop_len=hop_len,
            )
            raf_eval_ds = RAFDataset(
                dataparser_outputs=dpo_test,
                mode='eval',
                max_len=max_len_frames,
                max_len_time=raf_max_len_s,
                wav_path=os.path.join(raf_data_root, 'data'),
                fs=raf_fs, hop_len=hop_len,
            )
            
            # 关键修改：优化数据加载器配置
            import multiprocessing as mp
            
            # 设置多进程启动方法（在aarch64上很重要）
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # 如果已经设置过，忽略错误
            
            # 计算最优的worker数量（针对aarch64优化）
            cpu_count = os.cpu_count()
            if cpu_count >= 64:  # 大系统
                optimal_workers = min(16, cpu_count // 8)  # 保守设置
            elif cpu_count >= 16:  # 中等系统
                optimal_workers = min(2, cpu_count // 8)
            else:  # 小系统
                optimal_workers = 1
                
            logger.info(f"[RAF] 检测到 {cpu_count} 个CPU核心，使用 {optimal_workers} 个数据加载worker")
            
            # 使用更安全的数据加载器配置
            raf_loader = DataLoader(
                raf_train_ds, 
                batch_size=2048, 
                shuffle=True, 
                num_workers=optimal_workers, 
                pin_memory=True,
                persistent_workers=optimal_workers > 0,  # 只有worker>0时才持久化
                timeout=30,  # 设置超时
                multiprocessing_context='spawn' if optimal_workers > 0 else None
            )
            raf_eval_loader = DataLoader(
                raf_eval_ds, 
                batch_size=2048, 
                shuffle=False, 
                num_workers=optimal_workers, 
                pin_memory=True,
                persistent_workers=optimal_workers > 0,
                timeout=30,
                multiprocessing_context='spawn' if optimal_workers > 0 else None
            )
            
            raf_iter = iter(raf_loader)
            raf_eval_iter = iter(raf_eval_loader)
            logger.info(f"[RAF] Train items: {len(raf_train_ds)} | Eval items: {len(raf_eval_ds)}")
            
        else:
            if isinstance(raf_data_root, str) and len(raf_data_root) > 0:
                logger.info(f"[RAF] Provided raf_data not found or not a directory: {raf_data_root}")
    except Exception as e:
        logger.info(f"[RAF] Failed to initialize RAF loader: {e}")
        # 记录详细错误信息以便调试
        import traceback
        logger.info(f"[RAF] Error details: {traceback.format_exc()}")
    
    audio_ckpt_dir = os.path.join(dataset.model_path, "audio_ckpts")
    os.makedirs(audio_ckpt_dir, exist_ok=True)

    def save_audio_ckpt(tag: str):
        try:
            with torch.no_grad():
                # 保存完整的128×128×128网格用于局部特征提取
                current_grid_full = gaussians.get_full_grid().detach().cpu().numpy()  # [35, 128, 128, 128]
                anchors = gaussians.get_anchor.detach()
                min_bound = anchors.min(dim=0)[0].cpu().numpy()
                max_bound = anchors.max(dim=0)[0].cpu().numpy()

            # 保存local_pooling_net（如果存在）
            local_pooling_net_state = None
            if hasattr(gaussians, 'local_pooling_net') and gaussians.local_pooling_net is not None:
                local_pooling_net_state = gaussians.local_pooling_net.state_dict()
            
            torch.save({
                "audio_field": audio_field.state_dict(),
                "time_enc": time_enc.state_dict() if hasattr(time_enc, "state_dict") else None,
                "pos_enc": pos_enc.state_dict() if hasattr(pos_enc, "state_dict") else None,
                "rot_enc": rot_enc.state_dict() if hasattr(rot_enc, "state_dict") else None,
                "optimizer_audio": optimizer_audio.state_dict(),
                "grid_feature": current_grid_full,   # [35, 128, 128, 128] 完整网格
                "local_pooling_net": local_pooling_net_state,  # 局部池化网络权重
                "aabb_min": min_bound,                # [3]
                "aabb_max": max_bound,                # [3]
                "raf_data_root": raf_data_root,      # 保存RAF数据路径
                "raf_fs": raf_fs,                    # 保存采样率
                "raf_max_len_s": raf_max_len_s,      # 保存最大长度
                "hop_len": hop_len,                  # 保存hop长度
                "max_len_frames": max_len_frames,    # 保存最大帧数
            }, os.path.join(audio_ckpt_dir, f"audio_{tag}.pth"))
            logger.info(f"[Audio] Saved audio checkpoint with full grid [35,128,128,128]: {os.path.join(audio_ckpt_dir, f'audio_{tag}.pth')}")
        except Exception as e:
            logger.info(f"[Audio] Failed to save audio checkpoint ({tag}): {e}")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    #检查点
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        #更新学习率
        gaussians.update_learning_rate(iteration)
        
        # 音频学习率更新移到声场训练循环内部

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        #加入一个相机
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        #渲染
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        
        #计算损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

        #反向传播
        loss.backward()
        
        # 立即更新光场，避免梯度污染
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        
        iter_end.record()
        
                # ---------------- Train audio after 15000 iterations ----------------
        if raf_loader is not None and iteration >= start_step_audio:   #训练声场
            # 声场训练：每次进入时训练n轮，n=1
            n = 1
            logger.info(f"[Training Mode] Iteration {iteration}: 开始声场训练（{n}轮）")
            for audio_step in range(n):
                # 更新音频学习率 - 基于全局音频步数
                global_audio_step += 1
                current_audio_lr = audio_scheduler_args(global_audio_step)
                for param_group in optimizer_audio.param_groups:
                    param_group['lr'] = current_audio_lr
                # 1) get a batch
                try:
                    batch_audio = next(raf_iter)
                except StopIteration:
                    raf_iter = iter(raf_loader)
                    batch_audio = next(raf_iter)

                # 2) build encodings - 严格匹配NeRAF的时间归一化
                # time in [0,1], shape [B, 1]
                time_query = batch_audio['time_query'].to('cuda').float().unsqueeze(-1)
                
                # NeRAF的时间归一化: time_query.float()/float(self.max_len - 1.0)
                t_norm = time_query.float() / float(max_len_frames - 1.0)
                t_feat = time_enc(t_norm)

                # positions -> normalize to [0,1] using current Gaussian AABB (min/max on anchors)
                anchors = gaussians.get_anchor.detach()
                min_bound = anchors.min(dim=0)[0]
                max_bound = anchors.max(dim=0)[0]
                extent = (max_bound - min_bound).clamp_min(1e-6)
                mic_pose = batch_audio['mic_pose'].to('cuda').float()
                src_pose = batch_audio['source_pose'].to('cuda').float()
                mic01 = ((mic_pose - min_bound) / extent).clamp(0.0, 1.0)
                src01 = ((src_pose - min_bound) / extent).clamp(0.0, 1.0)
                mic_feat = pos_enc(mic01)
                src_feat = pos_enc(src01)

                # rotation already in [0,1]
                rot = batch_audio['rot'].to('cuda').float()
                rot_feat = rot_enc(rot)

                # 3) grid feature from ScaffoldGS - 提取扬声器和听者周围9x9x9局部网格
                # 允许反向到 local_pooling_net（但不会反向到Gaussians几何）
                with torch.no_grad():
                    grid_feat = gaussians.get_feature(
                        speaker_pose=src_pose,
                        listener_pose=mic_pose
                    ).to('cuda')  # [B, 70]

                # 4) concat and forward audio field
                h = torch.cat([grid_feat, t_feat, mic_feat, src_feat, rot_feat], dim=-1)  # [B, in_size]
                pred = audio_field(h)            # [B, 1, F]
                pred = pred.squeeze(1)           # [B, F]  (log-magnitude predicted)

                # 5) compute audio loss using NeRAF STFTLoss
                gt = batch_audio['data'].to('cuda').float()       # [B, F]
                
                # 使用NeRAF的STFTLoss - 严格单帧STFT切片训练
                loss_dict = criterion_audio(pred, gt)
                
                # 按照NeRAF的权重组合损失
                audio_loss = 1.0 * loss_dict['audio_sc_loss'] + 1.0 * loss_dict['audio_mag_loss']
                
                # 调试：监控音频输出幅度范围和学习率
                if iteration % 100 == 0 and audio_step == 0:  # 只在第一次时打印调试信息
                    logger.info(f"[Audio Debug] Iter {iteration}, Audio Step {global_audio_step}: Pred range [{pred.min().item():.3f}, {pred.max().item():.3f}], GT range [{gt.min().item():.3f}, {gt.max().item():.3f}]")
                    logger.info(f"[Audio Debug] SC loss: {loss_dict['audio_sc_loss'].item():.6f}, Mag loss: {loss_dict['audio_mag_loss'].item():.6f}, Total loss: {audio_loss.item():.6f}")
                    logger.info(f"[Audio Debug] Audio learning rate: {current_audio_lr:.8f} (Global audio step: {global_audio_step})")
                    
                    # 检查预测的线性幅度
                    pred_linear = torch.exp(pred) - 1e-3
                    gt_linear = torch.exp(gt) - 1e-3
                    logger.info(f"[Audio Debug] Pred mag range [{pred_linear.min().item():.6f}, {pred_linear.max().item():.6f}], GT mag range [{gt_linear.min().item():.6f}, {gt_linear.max().item():.6f}]")

                # 6) optimize audio only
                optimizer_audio.zero_grad(set_to_none=True)
                audio_loss.backward()

                optimizer_audio.step()

                # 7) TB logging - 分别记录SC loss和Mag loss（只在最后一次记录）
                if audio_step == 9 and tb_writer:  # 只在第10次时记录到TensorBoard
                    tb_writer.add_scalar(f'{dataset_name}/audio_sc_loss', loss_dict['audio_sc_loss'].item(), iteration)
                    tb_writer.add_scalar(f'{dataset_name}/audio_mag_loss', loss_dict['audio_mag_loss'].item(), iteration)
                    tb_writer.add_scalar(f'{dataset_name}/audio_total_loss', audio_loss.item(), iteration)
                    # 记录当前音频学习率和全局音频步数
                    tb_writer.add_scalar(f'{dataset_name}/audio_learning_rate', current_audio_lr, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/global_audio_step', global_audio_step, iteration)
                    
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                
                        # -------- Periodic eval: vision + audio (every 100 steps after 2k) --------
            if iteration >= 10000 and (iteration % 100 == 0):
                # Vision quick eval: sample a few views (train/test各取若干)
                try:
                    scene.gaussians.eval()
                    num_views = 3  # 评估少量相机，加快速度
                    views_eval = []
                    trcams = scene.getTrainCameras()
                    tecams = scene.getTestCameras()
                    if trcams:
                        views_eval.extend(trcams[:min(num_views, len(trcams))])
                    if tecams:
                        views_eval.extend(tecams[:min(num_views, len(tecams))])
                    l1_sum = 0.0
                    psnr_sum = 0.0
                    cnt = 0
                    for v in views_eval:
                        voxel_visible_mask = prefilter_voxel(v, gaussians, pipe, background)
                        pkg = render(v, gaussians, pipe, background, visible_mask=voxel_visible_mask)
                        render_im = torch.clamp(pkg["render"], 0.0, 1.0)
                        gt_im = torch.clamp(v.original_image.to("cuda"), 0.0, 1.0)
                        l1_sum += l1_loss(render_im, gt_im).mean().item()
                        psnr_sum += psnr(render_im, gt_im).mean().item()
                        cnt += 1
                    if cnt > 0:
                        vision_l1 = l1_sum / cnt
                        vision_psnr = psnr_sum / cnt
                        logger.info(f"[Eval/Vision] iter {iteration}: L1 {vision_l1:.6f}, PSNR {vision_psnr:.3f}")
                        if tb_writer:
                            tb_writer.add_scalar(f'{dataset_name}/eval_vision_l1', vision_l1, iteration)
                            tb_writer.add_scalar(f'{dataset_name}/eval_vision_psnr', vision_psnr, iteration)
                finally:
                    scene.gaussians.train()

                # Audio quick eval: take one eval batch
                if raf_eval_loader is not None:
                    try:
                        try:
                            eval_batch = next(raf_eval_iter)
                        except StopIteration:
                            raf_eval_iter = iter(raf_eval_loader)
                            eval_batch = next(raf_eval_iter)

                        # build features (同训练，但 no grad) - 严格匹配NeRAF时间归一化
                        time_query = eval_batch['time_query'].to('cuda').float().unsqueeze(-1)
                        # NeRAF的时间归一化: time_query.float()/float(self.max_len - 1.0)
                        t_norm = time_query.float() / float(max_len_frames - 1.0)
                        t_feat = time_enc(t_norm)

                        anchors = gaussians.get_anchor.detach()
                        min_bound = anchors.min(dim=0)[0]
                        max_bound = anchors.max(dim=0)[0]
                        extent = (max_bound - min_bound).clamp_min(1e-6)

                        mic_pose = eval_batch['mic_pose'].to('cuda').float()
                        src_pose = eval_batch['source_pose'].to('cuda').float()
                        mic01 = ((mic_pose - min_bound) / extent).clamp(0.0, 1.0)
                        src01 = ((src_pose - min_bound) / extent).clamp(0.0, 1.0)
                        mic_feat = pos_enc(mic01)
                        src_feat = pos_enc(src01)

                        rot = eval_batch['rot'].to('cuda').float()
                        rot_feat = rot_enc(rot)

                        with torch.no_grad():
                            grid_feat = gaussians.get_feature(
                                    speaker_pose=src_pose,  # [B, 3]
                                    listener_pose=mic_pose   # [B, 3]
                                ).to('cuda')  # [B, 70] - 局部特征

                        h = torch.cat([grid_feat, t_feat, mic_feat, src_feat, rot_feat], dim=-1)
                        with torch.no_grad():
                            pred = audio_field(h).squeeze(1)  # [B, F], log-magnitude
                        gt = eval_batch['data'].to('cuda').float()
                        
                        # 计算评估损失（使用STFTLoss）
                        eval_loss_dict = criterion_audio(pred, gt)
                        eval_total_loss = 1.0 * eval_loss_dict['audio_sc_loss'] + 1.0 * eval_loss_dict['audio_mag_loss']
                        
                        logger.info(f"[Eval/Audio] iter {iteration}: SC loss {eval_loss_dict['audio_sc_loss'].item():.6f}, Mag loss {eval_loss_dict['audio_mag_loss'].item():.6f}, Total {eval_total_loss.item():.6f}")
                        if tb_writer:
                            tb_writer.add_scalar(f'{dataset_name}/eval_audio_sc_loss', eval_loss_dict['audio_sc_loss'].item(), iteration)
                            tb_writer.add_scalar(f'{dataset_name}/eval_audio_mag_loss', eval_loss_dict['audio_mag_loss'].item(), iteration)
                            tb_writer.add_scalar(f'{dataset_name}/eval_audio_total_loss', eval_total_loss.item(), iteration)
                    except Exception as e:
                        logger.info(f"[Eval/Audio] eval failed at iter {iteration}: {e}")
            if iteration == opt.iterations:
                progress_bar.close()
                
            #打印保存raf loader 信息
                        # ---- RAF loader sanity log (every 100 iterations) ----
            # if raf_loader is not None and (iteration % 100 == 0 or iteration == first_iter):
            #     try:
            #         batch = next(raf_iter)
            #     except StopIteration:
            #         raf_iter = iter(raf_loader)
            #         batch = next(raf_iter)
            #     try:
            #         data_tensor = batch.get('data', None)
            #         time_q = batch.get('time_query', None)
            #         mic_pose = batch.get('mic_pose', None)
            #         src_pose = batch.get('source_pose', None)
            #         rot = batch.get('rot', None)
            #         data_shape = tuple(data_tensor.shape) if isinstance(data_tensor, torch.Tensor) else 'N/A'
            #         time_q_info = (tuple(time_q.shape), int(time_q.min()), int(time_q.max())) if isinstance(time_q, torch.Tensor) else 'N/A'
            #         mic_shape = tuple(mic_pose.shape) if isinstance(mic_pose, torch.Tensor) else 'N/A'
            #         src_shape = tuple(src_pose.shape) if isinstance(src_pose, torch.Tensor) else 'N/A'
            #         rot_shape = tuple(rot.shape) if isinstance(rot, torch.Tensor) else 'N/A'
            #         logger.info(f"[RAF] sample -> data{data_shape}, time{time_q_info}, mic{mic_shape}, src{src_shape}, rot{rot_shape}")
            #     except Exception as e:
            #         logger.info(f"[RAF] logging batch failed: {e}")


            #保存记录报告
            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                save_audio_ckpt(f"iter_{iteration}")
                
                if iteration >= 10000 and iteration % 1000 == 0:
                    logger.info(f"\n[ITER {iteration}] Starting periodic audio evaluation...")
                    try:
                        evaluate_audio_field_periodic(
                            dataset.model_path, raf_data_root, raf_fs, raf_max_len_s, 
                            hop_len, max_len_frames, iteration, logger
                        )
                        logger.info(f"[ITER {iteration}] Periodic audio evaluation complete.")
                    except Exception as e:
                        logger.info(f"[ITER {iteration}] Periodic audio evaluation failed: {e}")
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
                  
            
                
            # Optimizer step - 已移到光场训练后立即执行，避免梯度污染
            # if iteration < opt.iterations:
            #     gaussians.optimizer.step()
            #     gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                save_audio_ckpt(f"chkpnt_{iteration}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        # Clean dataset_name for TensorBoard compatibility - remove special characters
        clean_name = dataset_name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('-', '_')
        
        # Add debug logging to verify values
        #logger.info(f"[TB Debug] Logging - L1: {Ll1.item():.6f}, Total: {loss.item():.6f}, Iter: {iteration}")
        
        tb_writer.add_scalar(f'{clean_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{clean_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{clean_name}/iter_time', elapsed, iteration)
        
        # Force flush to ensure data is written
        tb_writer.flush()

    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)

def evaluate_audio_field_periodic(model_path, raf_data_root, raf_fs, raf_max_len_s, hop_len, max_len_frames, iteration, logger):
    """训练过程中的定期音频评估（简化版，只计算STFT相关指标）"""
    try:
        # 导入必要的模块
        from arguments import RAFDataParserConfig, RAFDataParser, RAFDataset
        from torch.utils.data import DataLoader
        import numpy as np
        
        logger.info(f"[Periodic Audio Eval] Starting evaluation at iteration {iteration}...")
        logger.info(f"[Periodic Audio Eval] Model path: {model_path}")
        logger.info(f"[Periodic Audio Eval] RAF data: {raf_data_root}")
        
        # 定义STFTLoss用于评估
        criterion_audio = STFTLoss(loss_type='mse')
        
        # 1. 加载当前迭代的音频检查点
        audio_ckpt_dir = os.path.join(model_path, "audio_ckpts")
        ckpt_path = os.path.join(audio_ckpt_dir, f"audio_iter_{iteration}.pth")
        
        if not os.path.exists(ckpt_path):
            logger.warning(f"[Periodic Audio Eval] Checkpoint not found: {ckpt_path}")
            return
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        logger.info(f"[Periodic Audio Eval] Loaded checkpoint: audio_iter_{iteration}.pth")
        
        # 2. 重建音频网络
        from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
        import torch.nn as nn
        import torch.nn.functional as F
        
        time_enc = NeRFEncoding(in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
        pos_enc = NeRFEncoding(in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
        rot_enc = SHEncoding(levels=4, implementation="tcnn")
        
        time_dim = time_enc.get_out_dim()
        pos_dim = pos_enc.get_out_dim()
        rot_dim = rot_enc.get_out_dim()
        
        audio_W = 512
        audio_F = 513
        grid_feat_dim = 70  # 局部网格特征维度
        in_size = grid_feat_dim + time_dim + 2 * pos_dim + rot_dim
        
        
        
        audio_field = NeRAFAudioSoundField(in_size=in_size, W=audio_W, sound_rez=1, N_frequencies=audio_F)
        
        # 加载权重
        if checkpoint.get("time_enc") is not None:
            time_enc.load_state_dict(checkpoint["time_enc"])
        if checkpoint.get("pos_enc") is not None:
            pos_enc.load_state_dict(checkpoint["pos_enc"])
        if checkpoint.get("rot_enc") is not None:
            rot_enc.load_state_dict(checkpoint["rot_enc"])
        
        audio_field.load_state_dict(checkpoint["audio_field"])
        
        # 移动到设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        time_enc = time_enc.to(device)
        pos_enc = pos_enc.to(device)
        rot_enc = rot_enc.to(device)
        audio_field = audio_field.to(device)
        audio_field.eval()
        
        # 3. 加载评估数据（使用STFT切片评估）
        dp_cfg = RAFDataParserConfig(data=Path(raf_data_root))
        dp = RAFDataParser(dp_cfg)
        dpo_test = dp.get_dataparser_outputs(split="test")
        
        # 创建STFT切片评估数据集
        raf_eval_ds = RAFDataset(
            dataparser_outputs=dpo_test,
            mode='eval',  # STFT切片评估
            max_len=max_len_frames,
            max_len_time=raf_max_len_s,
            wav_path=os.path.join(raf_data_root, 'data'),
            fs=raf_fs, 
            hop_len=hop_len,
        )
        
        # 评估更多样本以加快速度（STFT切片评估较快）
        eval_samples = min(100, len(raf_eval_ds))
        
        # 创建子集数据集
        eval_subset = torch.utils.data.Subset(raf_eval_ds, list(range(eval_samples)))
        eval_loader = DataLoader(eval_subset, batch_size=16, shuffle=False, num_workers=0)
        
        # 4. 执行评估
        logger.info(f"[Periodic Audio Eval] Evaluating {eval_samples} STFT slices...")
        
        all_metrics = []
        
        for batch_idx, batch in enumerate(eval_loader):
            try:
                with torch.no_grad():
                    # 预测STFT切片
                    time_query = batch['time_query'].to(device).float().unsqueeze(-1)
                    t_norm = time_query.float() / float(max_len_frames - 1.0)
                    t_feat = time_enc(t_norm)
                    
                    # 使用保存的边界信息
                    min_bound = torch.tensor(checkpoint["aabb_min"]).to(device)
                    max_bound = torch.tensor(checkpoint["aabb_max"]).to(device)
                    extent = (max_bound - min_bound).clamp_min(1e-6)
                    
                    mic_pose = batch['mic_pose'].to(device).float()
                    src_pose = batch['source_pose'].to(device).float()
                    mic01 = ((mic_pose - min_bound) / extent).clamp(0.0, 1.0)
                    src01 = ((src_pose - min_bound) / extent).clamp(0.0, 1.0)
                    mic_feat = pos_enc(mic01)
                    src_feat = pos_enc(src01)
                    
                    rot = batch['rot'].to(device).float()
                    rot_feat = rot_enc(rot)
                    
                    # 使用保存的网格特征
                    grid_feat = torch.tensor(checkpoint["grid_feature"]).to(device)
                    B = t_feat.shape[0]
                    grid_feat = grid_feat.unsqueeze(0).expand(B, -1)
                    
                    h = torch.cat([grid_feat, t_feat, mic_feat, src_feat, rot_feat], dim=-1)
                    field_outputs = audio_field(h)  # [B, C, F] where B=batch_size, C=1, F=frequencies
                    
                    # 对于STFT切片模式，输出应该是 [B, 1, F]，我们需要 [B, F]
                    pred_log_stft = field_outputs.squeeze(1)  # [B, F] log-magnitude
                    
                    # 获取真实数据
                    gt_log_stft = batch['data'].to(device).float()
                    
                    # 计算STFT损失
                    loss_dict = criterion_audio(pred_log_stft, gt_log_stft)
                    total_loss = 1.0 * loss_dict['audio_sc_loss'] + 1.0 * loss_dict['audio_mag_loss']
                    
                    metrics = {
                        'stft_sc_loss': loss_dict['audio_sc_loss'].item(),
                        'stft_mag_loss': loss_dict['audio_mag_loss'].item(),
                        'stft_total_loss': total_loss.item(),
                    }
                    
                    all_metrics.append(metrics)
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"[Periodic Audio Eval] Processed {batch_idx + 1}/{len(eval_loader)} batches")
                        
            except Exception as e:
                logger.error(f"[Periodic Audio Eval] Error evaluating batch {batch_idx}: {e}")
                continue
        
        # 5. 计算平均指标
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values)
            
            # 6. 保存结果
            results_dir = os.path.join(model_path, "audio_evaluation_results")
            os.makedirs(results_dir, exist_ok=True)
            
            periodic_file = os.path.join(results_dir, f"periodic_metrics_iter_{iteration}.json")
            with open(periodic_file, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            
            # 打印结果
            logger.info("=" * 60)
            logger.info(f"PERIODIC AUDIO EVALUATION RESULTS (Iteration {iteration})")
            logger.info("=" * 60)
            
            for key, value in avg_metrics.items():
                if not key.endswith('_std'):
                    std_key = f"{key}_std"
                    std_value = avg_metrics.get(std_key, 0)
                    logger.info(f"{key:30s}: {value:.6f} ± {std_value:.6f}")
            
            logger.info("=" * 60)
            logger.info(f"[Periodic Audio Eval] Results saved to: {periodic_file}")
            
        else:
            logger.error("[Periodic Audio Eval] No valid STFT slices evaluated")
            
    except Exception as e:
        logger.error(f"[Periodic Audio Eval] Periodic audio evaluation failed: {e}")
        raise

def evaluate_audio_field(model_path, raf_data_root, raf_fs, raf_max_len_s, hop_len, max_len_frames, logger):
    """评估音频场的STFT指标（简化版）"""
    try:
        # 导入必要的模块
        from arguments import RAFDataParserConfig, RAFDataParser, RAFDataset
        from torch.utils.data import DataLoader
        from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
        import torch.nn as nn
        import torch.nn.functional as F
        import numpy as np
        
        logger.info(f"[Audio Eval] Starting audio field evaluation...")
        logger.info(f"[Audio Eval] Model path: {model_path}")
        logger.info(f"[Audio Eval] RAF data: {raf_data_root}")
        
        criterion_audio = STFTLoss(loss_type='mse')
        
        # 1. 加载音频检查点
        audio_ckpt_dir = os.path.join(model_path, "audio_ckpts")
        audio_ckpt_files = [f for f in os.listdir(audio_ckpt_dir) if f.startswith('audio_') and f.endswith('.pth')]
        audio_ckpt_files.sort()
        latest_ckpt = audio_ckpt_files[-1]
        ckpt_path = os.path.join(audio_ckpt_dir, latest_ckpt)
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        logger.info(f"[Audio Eval] Loaded checkpoint: {latest_ckpt}")
        
        # 2. 重建音频网络
        time_enc = NeRFEncoding(in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
        pos_enc = NeRFEncoding(in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
        rot_enc = SHEncoding(levels=4, implementation="tcnn")
        
        time_dim = time_enc.get_out_dim()
        pos_dim = pos_enc.get_out_dim()
        rot_dim = rot_enc.get_out_dim()
        
        audio_W = 512
        audio_F = 513
        grid_feat_dim = 70  # 局部网格特征维度
        in_size = grid_feat_dim + time_dim + 2 * pos_dim + rot_dim
        
    
        
        audio_field = NeRAFAudioSoundField(in_size=in_size, W=audio_W, sound_rez=1, N_frequencies=audio_F)
        
        # 加载权重
        if checkpoint.get("time_enc") is not None:
            time_enc.load_state_dict(checkpoint["time_enc"])
        if checkpoint.get("pos_enc") is not None:
            pos_enc.load_state_dict(checkpoint["pos_enc"])
        if checkpoint.get("rot_enc") is not None:
            rot_enc.load_state_dict(checkpoint["rot_enc"])
        
        audio_field.load_state_dict(checkpoint["audio_field"])
        
        # 移动到设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        time_enc = time_enc.to(device)
        pos_enc = pos_enc.to(device)
        rot_enc = rot_enc.to(device)
        audio_field = audio_field.to(device)
        audio_field.eval()
        
        # 3. 加载评估数据
        dp_cfg = RAFDataParserConfig(data=Path(raf_data_root))
        dp = RAFDataParser(dp_cfg)
        dpo_test = dp.get_dataparser_outputs(split="test")
        
        # 创建STFT切片评估数据集
        raf_eval_ds = RAFDataset(
            dataparser_outputs=dpo_test,
            mode='eval',  # STFT切片评估
            max_len=max_len_frames,
            max_len_time=raf_max_len_s,
            wav_path=os.path.join(raf_data_root, 'data'),
            fs=raf_fs, 
            hop_len=hop_len,
        )
        
        eval_loader = DataLoader(raf_eval_ds, batch_size=16, shuffle=False, num_workers=2)
        
        # 4. 执行评估
        logger.info(f"[Audio Eval] Evaluating {len(raf_eval_ds)} STFT slices...")
        
        all_metrics = []
        
        for batch_idx, batch in enumerate(eval_loader):
            try:
                with torch.no_grad():
                    # 预测STFT切片
                    time_query = batch['time_query'].to(device).float().unsqueeze(-1)
                    t_norm = time_query.float() / float(max_len_frames - 1.0)
                    t_feat = time_enc(t_norm)
                    
                    # 使用保存的边界信息
                    min_bound = torch.tensor(checkpoint["aabb_min"]).to(device)
                    max_bound = torch.tensor(checkpoint["aabb_max"]).to(device)
                    extent = (max_bound - min_bound).clamp_min(1e-6)
                    
                    mic_pose = batch['mic_pose'].to(device).float()
                    src_pose = batch['source_pose'].to(device).float()
                    mic01 = ((mic_pose - min_bound) / extent).clamp(0.0, 1.0)
                    src01 = ((src_pose - min_bound) / extent).clamp(0.0, 1.0)
                    mic_feat = pos_enc(mic01)
                    src_feat = pos_enc(src01)
                    
                    rot = batch['rot'].to(device).float()
                    rot_feat = rot_enc(rot)
                    
                    # 使用保存的网格特征
                    grid_feat = torch.tensor(checkpoint["grid_feature"]).to(device)
                    B = t_feat.shape[0]
                    grid_feat = grid_feat.unsqueeze(0).expand(B, -1)
                    
                    h = torch.cat([grid_feat, t_feat, mic_feat, src_feat, rot_feat], dim=-1)
                    field_outputs = audio_field(h)  # [B, C, F] where B=batch_size, C=1, F=frequencies
                    
                    # 对于STFT切片模式，输出应该是 [B, 1, F]，我们需要 [B, F]
                    pred_log_stft = field_outputs.squeeze(1)  # [B, F] log-magnitude
                    
                    # 获取真实数据
                    gt_log_stft = batch['data'].to(device).float()
                    
                    # 计算STFT损失
                    loss_dict = criterion_audio(pred_log_stft, gt_log_stft)
                    total_loss =1.0 * loss_dict['audio_sc_loss'] + 1.0 * loss_dict['audio_mag_loss']
                    
                    metrics = {
                        'stft_sc_loss': loss_dict['audio_sc_loss'].item(),
                        'stft_mag_loss': loss_dict['audio_mag_loss'].item(),
                        'stft_total_loss': total_loss.item(),
                    }
                    
                    all_metrics.append(metrics)
                    
                    if batch_idx % 100 == 0:
                        logger.info(f"[Audio Eval] Processed {batch_idx + 1}/{len(eval_loader)} STFT slices")
                        
            except Exception as e:
                logger.error(f"[Audio Eval] Error evaluating STFT slice {batch_idx}: {e}")
                continue
        
        # 5. 计算平均指标
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                avg_metrics[key] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values)
            
            # 6. 保存和打印结果
            results_dir = os.path.join(model_path, "audio_evaluation_results")
            os.makedirs(results_dir, exist_ok=True)
            
            avg_file = os.path.join(results_dir, "average_metrics.json")
            with open(avg_file, 'w') as f:
                json.dump(avg_metrics, f, indent=2)
            
            all_file = os.path.join(results_dir, "all_samples_metrics.json")
            with open(all_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            # 打印结果
            logger.info("=" * 60)
            logger.info("SCAFFOLDGS + RAF AUDIO FIELD EVALUATION RESULTS")
            logger.info("=" * 60)
            
            for key, value in avg_metrics.items():
                if not key.endswith('_std'):
                    std_key = f"{key}_std"
                    std_value = avg_metrics.get(std_key, 0)
                    logger.info(f"{key:30s}: {value:.6f} ± {std_value:.6f}")
            
            logger.info("=" * 60)
            logger.info(f"[Audio Eval] Results saved to: {results_dir}")
            
        else:
            logger.error("[Audio Eval] No valid STFT slices evaluated")
            
    except Exception as e:
        logger.error(f"[Audio Eval] Audio field evaluation failed: {e}")
        raise
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[20_000,30000,40000,50_000,60000,70000,80_000,90000,100000,110000,120_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000,30000,40000,50_000,60000,70000,80_000,90000,100000,110000,120_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")
   

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
    
    # Audio field evaluation
    logger.info("\n Starting audio field evaluation...")
    try:
        # 获取RAF数据路径
        raf_data_root = getattr(lp.extract(args), 'raf_data', '') if hasattr(lp.extract(args), 'raf_data') else ''
        raf_fs = int(getattr(lp.extract(args), 'raf_fs', 48000))
        raf_max_len_s = float(getattr(lp.extract(args), 'raf_max_len', 0.32))
        hop_len = 256 if raf_fs == 48000 else 128
        max_len_frames = max(1, int(raf_max_len_s * raf_fs / hop_len))
        
        if isinstance(raf_data_root, str) and len(raf_data_root) > 0 and os.path.isdir(raf_data_root):
            evaluate_audio_field(args.model_path, raf_data_root, raf_fs, raf_max_len_s, hop_len, max_len_frames, logger)
            logger.info("\nAudio field evaluation complete.")
        else:
            logger.info(f"\nSkipping audio field evaluation: RAF data not found or not configured")
    except Exception as e:
        logger.info(f"\nAudio field evaluation failed: {e}")
