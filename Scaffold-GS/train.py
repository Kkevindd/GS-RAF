# #
# # Copyright (C) 2023, Inria
# # GRAPHDECO research group, https://team.inria.fr/graphdeco
# # All rights reserved.
# #
# # This software is free for non-commercial, research and evaluation use 
# # under the terms of the LICENSE.md file.
# #
# # For inquiries contact  george.drettakis@inria.fr
# #

# import os
# import numpy as np

# import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# os.system('echo $CUDA_VISIBLE_DEVICES')


# import torch
# import torchvision
# import json
# import wandb
# import time
# from os import makedirs
# import shutil, pathlib
# from pathlib import Path
# from PIL import Image
# import torchvision.transforms.functional as tf
# # from lpipsPyTorch import lpips
# import lpips
# from random import randint
# from utils.loss_utils import l1_loss, ssim
# from gaussian_renderer import prefilter_voxel, render, network_gui
# import sys
# from scene import Scene, GaussianModel
# from utils.general_utils import safe_state
# import uuid
# from tqdm import tqdm
# from utils.image_utils import psnr
# from argparse import ArgumentParser, Namespace
# from arguments import ModelParams, PipelineParams, OptimizationParams

# from torch.utils.data import DataLoader
# from arguments import RAFDataParserConfig, RAFDataParser, RAFDataset
# from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
# import torch.nn as nn
# import torch.nn.functional as F


# # torch.set_num_threads(32)
# lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

# try:
#     from torch.utils.tensorboard import SummaryWriter
#     TENSORBOARD_FOUND = True
#     print("found tf board")
# except ImportError:
#     TENSORBOARD_FOUND = False
#     print("not found tf board")

# def saveRuntimeCode(dst: str) -> None:
#     additionalIgnorePatterns = ['.git', '.gitignore']
#     ignorePatterns = set()
#     ROOT = '.'
#     with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
#         for line in gitIgnoreFile:
#             if not line.startswith('#'):
#                 if line.endswith('\n'):
#                     line = line[:-1]
#                 if line.endswith('/'):
#                     line = line[:-1]
#                 ignorePatterns.add(line)
#     ignorePatterns = list(ignorePatterns)
#     for additionalPattern in additionalIgnorePatterns:
#         ignorePatterns.append(additionalPattern)

#     log_dir = pathlib.Path(__file__).parent.resolve()


#     shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
#     print('Backup Finished!')


# def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
#     print("\n\nsaving_iterations: ", saving_iterations)
#     first_iter = 0
#     tb_writer = prepare_output_and_logger(dataset)
    
#     #高斯模型和场景
#     gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
#                               dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
#     scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    
#     #参数
#     gaussians.training_setup(opt)
#     if checkpoint:
#         (model_params, first_iter) = torch.load(checkpoint)
#         gaussians.restore(model_params, opt)

#     #添加raf dataset setup
#         # ---------------- Audio modules (persistent) ----------------
#     # encodings
#     time_enc = NeRFEncoding(in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
#     pos_enc = NeRFEncoding(in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
#     rot_enc = SHEncoding(levels=4, implementation="tcnn")

#     time_dim = time_enc.get_out_dim()
#     pos_dim = pos_enc.get_out_dim()
#     rot_dim = rot_enc.get_out_dim()

#     # simple sound field head (same as NeRAF NeRAFAudioSoundField)
#     class NeRAFAudioSoundField(nn.Module):
#         def __init__(self, in_size, W, sound_rez=1, N_frequencies=513):
#             super().__init__()
#             self.soundfield = nn.ModuleList([
#                 nn.Linear(in_size, 5096), nn.Linear(5096, 2048),
#                 nn.Linear(2048, 1024), nn.Linear(1024, 1024),
#                 nn.Linear(1024, W)
#             ])
#             self.STFT_linear = nn.ModuleList([nn.Linear(W, N_frequencies) for _ in range(sound_rez)])
#         def forward(self, h):
#             for layer in self.soundfield:
#                 h = F.leaky_relu(layer(h), negative_slope=0.1)
#             output = []
#             for layer in self.STFT_linear:
#                 y = torch.tanh(layer(h)) * 10
#                 output.append(y.unsqueeze(1))
#             return torch.cat(output, dim=1)  # [B, sound_rez, F]

#     # model heads
#     audio_W = 512
#     audio_F = 513  # RAF fs=48k -> 513 bins for n_fft=1024
#     in_size = 1024 + time_dim + 2 * pos_dim + rot_dim
#     audio_field = NeRAFAudioSoundField(in_size=in_size, W=audio_W, sound_rez=1, N_frequencies=audio_F).cuda()

#     # optimizer for audio only
#     audio_lr = 1e-4
#     optimizer_audio = torch.optim.Adam(
#         list(audio_field.parameters()),
#         lr=audio_lr, eps=1e-8
#     )
#     criterion_audio = nn.MSELoss(reduction='mean')

#     # ---------------- RAF dataset (optional) setup ----------------
#     raf_loader = None
#     raf_eval_loader = None
#     raf_iter = None
#     raf_eval_iter = None
#     try:
#         raf_data_root = getattr(dataset, 'raf_data', '') if hasattr(dataset, 'raf_data') else ''
#         raf_fs = int(getattr(dataset, 'raf_fs', 48000))
#         raf_max_len_s = float(getattr(dataset, 'raf_max_len', 0.32))
#         if isinstance(raf_data_root, str) and len(raf_data_root) > 0 and os.path.isdir(raf_data_root):
#             logger.info(f"[RAF] Using raf_data={raf_data_root}, fs={raf_fs}, max_len_s={raf_max_len_s}")
#             dp_cfg = RAFDataParserConfig(data=Path(raf_data_root))
#             dp = RAFDataParser(dp_cfg)
#             dpo_train = dp.get_dataparser_outputs(split="train")
#             dpo_test  = dp.get_dataparser_outputs(split="test")
#             hop_len = 256 if raf_fs == 48000 else 128
#             max_len_frames = max(1, int(raf_max_len_s * raf_fs / hop_len))
#             raf_train_ds = RAFDataset(
#                 dataparser_outputs=dpo_train,
#                 mode='train',
#                 max_len=max_len_frames,
#                 max_len_time=raf_max_len_s,
#                 wav_path=os.path.join(raf_data_root, 'data'),
#                 fs=raf_fs, hop_len=hop_len,
#             )
#             raf_eval_ds = RAFDataset(
#                 dataparser_outputs=dpo_test,
#                 mode='eval',                     # 单帧切片评估；若要整段，改 'eval' 为你定义的整段模式
#                 max_len=max_len_frames,
#                 max_len_time=raf_max_len_s,
#                 wav_path=os.path.join(raf_data_root, 'data'),
#                 fs=raf_fs, hop_len=hop_len,
#             )
#             raf_loader = DataLoader(raf_train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
#             raf_eval_loader = DataLoader(raf_eval_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
#             raf_iter = iter(raf_loader)
#             raf_eval_iter = iter(raf_eval_loader)
#             logger.info(f"[RAF] Train items: {len(raf_train_ds)} | Eval items: {len(raf_eval_ds)} | max_len_frames: {max_len_frames} | hop_len: {hop_len}")
#         else:
#             if isinstance(raf_data_root, str) and len(raf_data_root) > 0:
#                 logger.info(f"[RAF] Provided raf_data not found or not a directory: {raf_data_root}")
#     except Exception as e:
#         logger.info(f"[RAF] Failed to initialize RAF loader: {e}")
        
#         # -------- audio checkpoint helpers --------
#     audio_ckpt_dir = os.path.join(dataset.model_path, "audio_ckpts")
#     os.makedirs(audio_ckpt_dir, exist_ok=True)

#     def save_audio_ckpt(tag: str):
#         try:
#             torch.save({
#                 "audio_field": audio_field.state_dict(),
#                 "time_enc": time_enc.state_dict() if hasattr(time_enc, "state_dict") else None,
#                 "pos_enc": pos_enc.state_dict() if hasattr(pos_enc, "state_dict") else None,
#                 "rot_enc": rot_enc.state_dict() if hasattr(rot_enc, "state_dict") else None,
#                 "optimizer_audio": optimizer_audio.state_dict(),
#             }, os.path.join(audio_ckpt_dir, f"audio_{tag}.pth"))
#             logger.info(f"[Audio] Saved audio checkpoint: {os.path.join(audio_ckpt_dir, f'audio_{tag}.pth')}")
#         except Exception as e:
#             logger.info(f"[Audio] Failed to save audio checkpoint ({tag}): {e}")

#     iter_start = torch.cuda.Event(enable_timing = True)
#     iter_end = torch.cuda.Event(enable_timing = True)


#     #检查点
#     viewpoint_stack = None
#     ema_loss_for_log = 0.0
#     progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
#     first_iter += 1
#     for iteration in range(first_iter, opt.iterations + 1):        
#         # network gui not available in scaffold-gs yet
#         if network_gui.conn == None:
#             network_gui.try_connect()
#         while network_gui.conn != None:
#             try:
#                 net_image_bytes = None
#                 custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
#                 if custom_cam != None:
#                     net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
#                     net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
#                 network_gui.send(net_image_bytes, dataset.source_path)
#                 if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
#                     break
#             except Exception as e:
#                 network_gui.conn = None

#         iter_start.record()
#         #更新学习率
#         gaussians.update_learning_rate(iteration)

#         bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#         #加入一个相机
#         # Pick a random Camera
#         if not viewpoint_stack:
#             viewpoint_stack = scene.getTrainCameras().copy()
#         viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
#         #渲染
#         # Render
#         if (iteration - 1) == debug_from:
#             pipe.debug = True
        
#         voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
#         retain_grad = (iteration < opt.update_until and iteration >= 0)
#         render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
#         image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        
#         #计算损失
#         gt_image = viewpoint_cam.original_image.cuda()
#         Ll1 = l1_loss(image, gt_image)

#         ssim_loss = (1.0 - ssim(image, gt_image))
#         scaling_reg = scaling.prod(dim=1).mean()
#         loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

#         #反向传播
#         loss.backward()
        
#         iter_end.record()
        
#                 # ---------------- Train audio after 2000 iterations ----------------
#         if raf_loader is not None and iteration >= 2000:
#             # 1) get a batch
#             try:
#                 batch_audio = next(raf_iter)
#             except StopIteration:
#                 raf_iter = iter(raf_loader)
#                 batch_audio = next(raf_iter)

#             # 2) build encodings
#             # time in [0,1], shape [B, 1]
#             time_query = batch_audio['time_query'].to('cuda').float().unsqueeze(-1)
#             # normalize time index to [0,1] using per-dataset max frame count
#             # here we approximate using batch max to stay simple
#             t_norm = (time_query / (time_query.max().clamp_min(1.0)))
#             t_feat = time_enc(t_norm)

#             # positions -> normalize to [0,1] using current Gaussian AABB (min/max on anchors)
#             anchors = gaussians.get_anchor.detach()
#             min_bound = anchors.min(dim=0)[0]
#             max_bound = anchors.max(dim=0)[0]
#             extent = (max_bound - min_bound).clamp_min(1e-6)
#             mic_pose = batch_audio['mic_pose'].to('cuda').float()
#             src_pose = batch_audio['source_pose'].to('cuda').float()
#             mic01 = ((mic_pose - min_bound) / extent).clamp(0.0, 1.0)
#             src01 = ((src_pose - min_bound) / extent).clamp(0.0, 1.0)
#             mic_feat = pos_enc(mic01)
#             src_feat = pos_enc(src01)

#             # rotation already in [0,1]
#             rot = batch_audio['rot'].to('cuda').float()
#             rot_feat = rot_enc(rot)

#             # 3) grid feature from ScaffoldGS
#             #    returns flattened 1024-d feature (no grad into gaussians)
#             with torch.no_grad():
#                 grid_feat = gaussians.get_feature().to('cuda')  # [1024]
#             B = t_feat.shape[0]
#             grid_feat = grid_feat.unsqueeze(0).expand(B, -1)   # [B, 1024]

#             # 4) concat and forward audio field
#             h = torch.cat([grid_feat, t_feat, mic_feat, src_feat, rot_feat], dim=-1)  # [B, in_size]
#             pred = audio_field(h)            # [B, 1, F]
#             pred = pred.squeeze(1)           # [B, F]  (log-magnitude predicted)

#             # 5) compute audio loss in log domain (RAFDataset returns log-magnitude slices)
#             gt = batch_audio['data'].to('cuda').float()       # [B, F]
#             audio_loss = criterion_audio(pred, gt)

#             # 6) optimize audio only
#             optimizer_audio.zero_grad(set_to_none=True)
#             audio_loss.backward()
#             optimizer_audio.step()

#             # 7) TB logging
#             if tb_writer:
#                 tb_writer.add_scalar(f'{dataset_name}/audio_train_loss', audio_loss.item(), iteration)

#         with torch.no_grad():
#             # Progress bar
#             ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

#             if iteration % 10 == 0:
#                 progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
#                 progress_bar.update(10)
                
#                         # -------- Periodic eval: vision + audio (every 100 steps after 2k) --------
#             if iteration >= 2000 and (iteration % 100 == 0):
#                 # Vision quick eval: sample a few views (train/test各取若干)
#                 try:
#                     scene.gaussians.eval()
#                     num_views = 3  # 评估少量相机，加快速度
#                     views_eval = []
#                     trcams = scene.getTrainCameras()
#                     tecams = scene.getTestCameras()
#                     if trcams:
#                         views_eval.extend(trcams[:min(num_views, len(trcams))])
#                     if tecams:
#                         views_eval.extend(tecams[:min(num_views, len(tecams))])
#                     l1_sum = 0.0
#                     psnr_sum = 0.0
#                     cnt = 0
#                     for v in views_eval:
#                         voxel_visible_mask = prefilter_voxel(v, gaussians, pipe, background)
#                         pkg = render(v, gaussians, pipe, background, visible_mask=voxel_visible_mask)
#                         render_im = torch.clamp(pkg["render"], 0.0, 1.0)
#                         gt_im = torch.clamp(v.original_image.to("cuda"), 0.0, 1.0)
#                         l1_sum += l1_loss(render_im, gt_im).mean().item()
#                         psnr_sum += psnr(render_im, gt_im).mean().item()
#                         cnt += 1
#                     if cnt > 0:
#                         vision_l1 = l1_sum / cnt
#                         vision_psnr = psnr_sum / cnt
#                         logger.info(f"[Eval/Vision] iter {iteration}: L1 {vision_l1:.6f}, PSNR {vision_psnr:.3f}")
#                         if tb_writer:
#                             tb_writer.add_scalar(f'{dataset_name}/eval_vision_l1', vision_l1, iteration)
#                             tb_writer.add_scalar(f'{dataset_name}/eval_vision_psnr', vision_psnr, iteration)
#                 finally:
#                     scene.gaussians.train()

#                 # Audio quick eval: take one eval batch
#                 if raf_eval_loader is not None:
#                     try:
#                         try:
#                             eval_batch = next(raf_eval_iter)
#                         except StopIteration:
#                             raf_eval_iter = iter(raf_eval_loader)
#                             eval_batch = next(raf_eval_iter)

#                         # build features (同训练，但 no grad)
#                         time_query = eval_batch['time_query'].to('cuda').float().unsqueeze(-1)
#                         # 用已知 max_len_frames 归一化时间
#                         # 若上面保留了 hop_len/max_len_frames，可复用；否则临时计算
#                         # 这里为了简单防御，用批内最大时间
#                         t_norm = (time_query / (time_query.max().clamp_min(1.0))).clamp(0.0, 1.0)
#                         t_feat = time_enc(t_norm)

#                         anchors = gaussians.get_anchor.detach()
#                         min_bound = anchors.min(dim=0)[0]
#                         max_bound = anchors.max(dim=0)[0]
#                         extent = (max_bound - min_bound).clamp_min(1e-6)

#                         mic_pose = eval_batch['mic_pose'].to('cuda').float()
#                         src_pose = eval_batch['source_pose'].to('cuda').float()
#                         mic01 = ((mic_pose - min_bound) / extent).clamp(0.0, 1.0)
#                         src01 = ((src_pose - min_bound) / extent).clamp(0.0, 1.0)
#                         mic_feat = pos_enc(mic01)
#                         src_feat = pos_enc(src01)

#                         rot = eval_batch['rot'].to('cuda').float()
#                         rot_feat = rot_enc(rot)

#                         with torch.no_grad():
#                             grid_feat = gaussians.get_feature().to('cuda')  # [1024]
#                         B = t_feat.shape[0]
#                         grid_feat = grid_feat.unsqueeze(0).expand(B, -1)

#                         h = torch.cat([grid_feat, t_feat, mic_feat, src_feat, rot_feat], dim=-1)
#                         with torch.no_grad():
#                             pred = audio_field(h).squeeze(1)  # [B, F], log-magnitude
#                         gt = eval_batch['data'].to('cuda').float()
#                         audio_eval_mse = torch.mean((pred - gt) ** 2).item()
#                         logger.info(f"[Eval/Audio] iter {iteration}: MSE(log-mag) {audio_eval_mse:.6f}")
#                         if tb_writer:
#                             tb_writer.add_scalar(f'{dataset_name}/eval_audio_mse', audio_eval_mse, iteration)
#                     except Exception as e:
#                         logger.info(f"[Eval/Audio] eval failed at iter {iteration}: {e}")
#             if iteration == opt.iterations:
#                 progress_bar.close()
                
#             #打印保存raf loader 信息
#                         # ---- RAF loader sanity log (every 100 iterations) ----
#             # if raf_loader is not None and (iteration % 100 == 0 or iteration == first_iter):
#             #     try:
#             #         batch = next(raf_iter)
#             #     except StopIteration:
#             #         raf_iter = iter(raf_loader)
#             #         batch = next(raf_iter)
#             #     try:
#             #         data_tensor = batch.get('data', None)
#             #         time_q = batch.get('time_query', None)
#             #         mic_pose = batch.get('mic_pose', None)
#             #         src_pose = batch.get('source_pose', None)
#             #         rot = batch.get('rot', None)
#             #         data_shape = tuple(data_tensor.shape) if isinstance(data_tensor, torch.Tensor) else 'N/A'
#             #         time_q_info = (tuple(time_q.shape), int(time_q.min()), int(time_q.max())) if isinstance(time_q, torch.Tensor) else 'N/A'
#             #         mic_shape = tuple(mic_pose.shape) if isinstance(mic_pose, torch.Tensor) else 'N/A'
#             #         src_shape = tuple(src_pose.shape) if isinstance(src_pose, torch.Tensor) else 'N/A'
#             #         rot_shape = tuple(rot.shape) if isinstance(rot, torch.Tensor) else 'N/A'
#             #         logger.info(f"[RAF] sample -> data{data_shape}, time{time_q_info}, mic{mic_shape}, src{src_shape}, rot{rot_shape}")
#             #     except Exception as e:
#             #         logger.info(f"[RAF] logging batch failed: {e}")


#             #保存记录报告
#             # Log and save
#             training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
#             if (iteration in saving_iterations):
#                 logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
#                 scene.save(iteration)
#                 save_audio_ckpt(f"iter_{iteration}")
            
#             # densification
#             if iteration < opt.update_until and iteration > opt.start_stat:
#                 # add statis
#                 gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
#                 # densification
#                 if iteration > opt.update_from and iteration % opt.update_interval == 0:
#                     gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
#             elif iteration == opt.update_until:
#                 del gaussians.opacity_accum
#                 del gaussians.offset_gradient_accum
#                 del gaussians.offset_denom
#                 torch.cuda.empty_cache()
                    
                  
            
                
#             # Optimizer step
#             if iteration < opt.iterations:
#                 gaussians.optimizer.step()
#                 gaussians.optimizer.zero_grad(set_to_none = True)
#             if (iteration in checkpoint_iterations):
#                 logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
#                 torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
#                 save_audio_ckpt(f"chkpnt_{iteration}")

# def prepare_output_and_logger(args):    
#     if not args.model_path:
#         if os.getenv('OAR_JOB_ID'):
#             unique_str=os.getenv('OAR_JOB_ID')
#         else:
#             unique_str = str(uuid.uuid4())
#         args.model_path = os.path.join("./output/", unique_str[0:10])
        
#     # Set up output folder
#     print("Output folder: {}".format(args.model_path))
#     os.makedirs(args.model_path, exist_ok = True)
#     with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
#         cfg_log_f.write(str(Namespace(**vars(args))))

#     # Create Tensorboard writer
#     tb_writer = None
#     if TENSORBOARD_FOUND:
#         tb_writer = SummaryWriter(args.model_path)
#     else:
#         print("Tensorboard not available: not logging progress")
#     return tb_writer

# # def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
# #     if tb_writer:
# #         tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
# #         tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
# #         tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)
# def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
#     if tb_writer:
#         # Clean dataset_name for TensorBoard compatibility - remove special characters
#         clean_name = dataset_name.replace('/', '_').replace('\\', '_').replace(' ', '_').replace('-', '_')
        
#         # Add debug logging to verify values
#         #logger.info(f"[TB Debug] Logging - L1: {Ll1.item():.6f}, Total: {loss.item():.6f}, Iter: {iteration}")
        
#         tb_writer.add_scalar(f'{clean_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
#         tb_writer.add_scalar(f'{clean_name}/train_loss_patches/total_loss', loss.item(), iteration)
#         tb_writer.add_scalar(f'{clean_name}/iter_time', elapsed, iteration)
        
#         # Force flush to ensure data is written
#         tb_writer.flush()

#     if wandb is not None:
#         wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })
    
#     # Report test and samples of training set
#     if iteration in testing_iterations:
#         scene.gaussians.eval()
#         torch.cuda.empty_cache()
#         validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
#                               {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

#         for config in validation_configs:
#             if config['cameras'] and len(config['cameras']) > 0:
#                 l1_test = 0.0
#                 psnr_test = 0.0
                
#                 if wandb is not None:
#                     gt_image_list = []
#                     render_image_list = []
#                     errormap_list = []

#                 for idx, viewpoint in enumerate(config['cameras']):
#                     voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
#                     image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
#                     gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
#                     if tb_writer and (idx < 30):
#                         tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
#                         tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

#                         if wandb:
#                             render_image_list.append(image[None])
#                             errormap_list.append((gt_image[None]-image[None]).abs())
                            
#                         if iteration == testing_iterations[0]:
#                             tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
#                             if wandb:
#                                 gt_image_list.append(gt_image[None])

#                     l1_test += l1_loss(image, gt_image).mean().double()
#                     psnr_test += psnr(image, gt_image).mean().double()

                
                
#                 psnr_test /= len(config['cameras'])
#                 l1_test /= len(config['cameras'])          
#                 logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
#                 if tb_writer:
#                     tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
#                     tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
#                 if wandb is not None:
#                     wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

#         if tb_writer:
#             # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
#             tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
#         torch.cuda.empty_cache()

#         scene.gaussians.train()

# def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
#     render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
#     error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
#     gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
#     makedirs(render_path, exist_ok=True)
#     makedirs(error_path, exist_ok=True)
#     makedirs(gts_path, exist_ok=True)
    
#     t_list = []
#     visible_count_list = []
#     name_list = []
#     per_view_dict = {}
#     for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
#         torch.cuda.synchronize();t_start = time.time()
        
#         voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
#         render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
#         torch.cuda.synchronize();t_end = time.time()

#         t_list.append(t_end - t_start)

#         # renders
#         rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
#         visible_count = (render_pkg["radii"] > 0).sum()
#         visible_count_list.append(visible_count)


#         # gts
#         gt = view.original_image[0:3, :, :]
        
#         # error maps
#         errormap = (rendering - gt).abs()


#         name_list.append('{0:05d}'.format(idx) + ".png")
#         torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
#         torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
#         torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
#         per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
#     with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
#             json.dump(per_view_dict, fp, indent=True)
    
#     return t_list, visible_count_list

# def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
#     with torch.no_grad():
#         gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
#                               dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
#         scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
#         gaussians.eval()

#         bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
#         background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
#         if not os.path.exists(dataset.model_path):
#             os.makedirs(dataset.model_path)

#         if not skip_train:
#             t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
#             train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
#             logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
#             if wandb is not None:
#                 wandb.log({"train_fps":train_fps.item(), })

#         if not skip_test:
#             t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
#             test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
#             logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
#             if tb_writer:
#                 tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
#             if wandb is not None:
#                 wandb.log({"test_fps":test_fps, })
    
#     return visible_count


# def readImages(renders_dir, gt_dir):
#     renders = []
#     gts = []
#     image_names = []
#     for fname in os.listdir(renders_dir):
#         render = Image.open(renders_dir / fname)
#         gt = Image.open(gt_dir / fname)
#         renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
#         gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
#         image_names.append(fname)
#     return renders, gts, image_names


# def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

#     full_dict = {}
#     per_view_dict = {}
#     full_dict_polytopeonly = {}
#     per_view_dict_polytopeonly = {}
#     print("")
    
#     scene_dir = model_paths
#     full_dict[scene_dir] = {}
#     per_view_dict[scene_dir] = {}
#     full_dict_polytopeonly[scene_dir] = {}
#     per_view_dict_polytopeonly[scene_dir] = {}

#     test_dir = Path(scene_dir) / "test"

#     for method in os.listdir(test_dir):

#         full_dict[scene_dir][method] = {}
#         per_view_dict[scene_dir][method] = {}
#         full_dict_polytopeonly[scene_dir][method] = {}
#         per_view_dict_polytopeonly[scene_dir][method] = {}

#         method_dir = test_dir / method
#         gt_dir = method_dir/ "gt"
#         renders_dir = method_dir / "renders"
#         renders, gts, image_names = readImages(renders_dir, gt_dir)

#         ssims = []
#         psnrs = []
#         lpipss = []

#         for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
#             ssims.append(ssim(renders[idx], gts[idx]))
#             psnrs.append(psnr(renders[idx], gts[idx]))
#             lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
#         if wandb is not None:
#             wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
#             wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
#             wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

#         logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
#         logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
#         logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
#         logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
#         print("")


#         if tb_writer:
#             tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
#             tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
#             tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
#             tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
#         full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
#                                                 "PSNR": torch.tensor(psnrs).mean().item(),
#                                                 "LPIPS": torch.tensor(lpipss).mean().item()})
#         per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
#                                                     "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
#                                                     "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
#                                                     "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

#     with open(scene_dir + "/results.json", 'w') as fp:
#         json.dump(full_dict[scene_dir], fp, indent=True)
#     with open(scene_dir + "/per_view.json", 'w') as fp:
#         json.dump(per_view_dict[scene_dir], fp, indent=True)
    
# def get_logger(path):
#     import logging

#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO) 
#     fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
#     fileinfo.setLevel(logging.INFO) 
#     controlshow = logging.StreamHandler()
#     controlshow.setLevel(logging.INFO)
#     formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
#     fileinfo.setFormatter(formatter)
#     controlshow.setFormatter(formatter)

#     logger.addHandler(fileinfo)
#     logger.addHandler(controlshow)

#     return logger

# if __name__ == "__main__":
#     # Set up command line argument parser
#     parser = ArgumentParser(description="Training script parameters")
#     lp = ModelParams(parser)
#     op = OptimizationParams(parser)
#     pp = PipelineParams(parser)
#     parser.add_argument('--ip', type=str, default="127.0.0.1")
#     parser.add_argument('--port', type=int, default=6009)
#     parser.add_argument('--debug_from', type=int, default=-1)
#     parser.add_argument('--detect_anomaly', action='store_true', default=False)
#     parser.add_argument('--warmup', action='store_true', default=False)
#     parser.add_argument('--use_wandb', action='store_true', default=False)
#     # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
#     # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
#     parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
#     parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000,4_000,5_000,6_000,7_000,8_000,9_000,10_0000,11_000,12_000,13_000,14_000,15_000,16_000,17_000,18_000,19_000,20_000,21_000,22_000,23_000,24_000,25_000,26_000,27_000,28_000,29_000,30_000])
#     parser.add_argument("--quiet", action="store_true")
#     parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
#     parser.add_argument("--start_checkpoint", type=str, default = None)
#     parser.add_argument("--gpu", type=str, default = '-1')
#     args = parser.parse_args(sys.argv[1:])
#     args.save_iterations.append(args.iterations)

    
#     # enable logging
    
#     model_path = args.model_path
#     os.makedirs(model_path, exist_ok=True)

#     logger = get_logger(model_path)


#     logger.info(f'args: {args}')

#     if args.gpu != '-1':
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
#         os.system("echo $CUDA_VISIBLE_DEVICES")
#         logger.info(f'using GPU {args.gpu}')

    

#     try:
#         saveRuntimeCode(os.path.join(args.model_path, 'backup'))
#     except:
#         logger.info(f'save code failed~')
        
#     dataset = args.source_path.split('/')[-1]
#     exp_name = args.model_path.split('/')[-2]
    
#     if args.use_wandb:
#         wandb.login()
#         run = wandb.init(
#             # Set the project where this run will be logged
#             project=f"Scaffold-GS-{dataset}",
#             name=exp_name,
#             # Track hyperparameters and run metadata
#             settings=wandb.Settings(start_method="fork"),
#             config=vars(args)
#         )
#     else:
#         wandb = None
    
#     logger.info("Optimizing " + args.model_path)

#     # Initialize system state (RNG)
#     safe_state(args.quiet)

#     # Start GUI server, configure and run training
#     network_gui.init(args.ip, args.port)
#     torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
#     # training
#     training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
#     if args.warmup:
#         logger.info("\n Warmup finished! Reboot from last checkpoints")
#         new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
#         training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

#     # All done
#     logger.info("\nTraining complete.")
   

#     # rendering
#     logger.info(f'\nStarting Rendering~')
#     visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
#     logger.info("\nRendering complete.")

#     # calc metrics
#     logger.info("\n Starting evaluation...")
#     evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
#     logger.info("\nEvaluating complete.")


#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

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
from scene import Scene, GaussianModel
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

    # simple sound field head (same as NeRAF NeRAFAudioSoundField)
    class NeRAFAudioSoundField(nn.Module):
        def __init__(self, in_size, W, sound_rez=1, N_frequencies=513):
            super().__init__()
            self.soundfield = nn.ModuleList([
                nn.Linear(in_size, 5096), nn.Linear(5096, 2048),
                nn.Linear(2048, 1024), nn.Linear(1024, 1024),
                nn.Linear(1024, W)
            ])
            self.STFT_linear = nn.ModuleList([nn.Linear(W, N_frequencies) for _ in range(sound_rez)])
        def forward(self, h):
            for layer in self.soundfield:
                h = F.leaky_relu(layer(h), negative_slope=0.1)
            output = []
            for layer in self.STFT_linear:
                y = torch.tanh(layer(h)) * 10
                output.append(y.unsqueeze(1))
            return torch.cat(output, dim=1)  # [B, sound_rez, F]

    # model heads
    audio_W = 512
    audio_F = 513  # RAF fs=48k -> 513 bins for n_fft=1024
    in_size = 1024 + time_dim + 2 * pos_dim + rot_dim
    audio_field = NeRAFAudioSoundField(in_size=in_size, W=audio_W, sound_rez=1, N_frequencies=audio_F).cuda()

    # optimizer for audio only
    audio_lr = 1e-4
    optimizer_audio = torch.optim.Adam(
        list(audio_field.parameters()),
        lr=audio_lr, eps=1e-8
    )
    criterion_audio = nn.MSELoss(reduction='mean')

    # ---------------- RAF dataset (optional) setup ----------------
    raf_loader = None
    raf_eval_loader = None
    raf_iter = None
    raf_eval_iter = None
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
                mode='eval',                     # 单帧切片评估；若要整段，改 'eval' 为你定义的整段模式
                max_len=max_len_frames,
                max_len_time=raf_max_len_s,
                wav_path=os.path.join(raf_data_root, 'data'),
                fs=raf_fs, hop_len=hop_len,
            )
            raf_loader = DataLoader(raf_train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
            raf_eval_loader = DataLoader(raf_eval_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
            raf_iter = iter(raf_loader)
            raf_eval_iter = iter(raf_eval_loader)
            logger.info(f"[RAF] Train items: {len(raf_train_ds)} | Eval items: {len(raf_eval_ds)} | max_len_frames: {max_len_frames} | hop_len: {hop_len}")
        else:
            if isinstance(raf_data_root, str) and len(raf_data_root) > 0:
                logger.info(f"[RAF] Provided raf_data not found or not a directory: {raf_data_root}")
    except Exception as e:
        logger.info(f"[RAF] Failed to initialize RAF loader: {e}")
        
        # -------- audio checkpoint helpers --------
    audio_ckpt_dir = os.path.join(dataset.model_path, "audio_ckpts")
    os.makedirs(audio_ckpt_dir, exist_ok=True)

    def save_audio_ckpt(tag: str):
        try:
            with torch.no_grad():
                current_grid_feat = gaussians.get_feature().detach().cpu().numpy()  # [1024]
                anchors = gaussians.get_anchor.detach()
                min_bound = anchors.min(dim=0)[0].cpu().numpy()
                max_bound = anchors.max(dim=0)[0].cpu().numpy()

            torch.save({
                "audio_field": audio_field.state_dict(),
                "time_enc": time_enc.state_dict() if hasattr(time_enc, "state_dict") else None,
                "pos_enc": pos_enc.state_dict() if hasattr(pos_enc, "state_dict") else None,
                "rot_enc": rot_enc.state_dict() if hasattr(rot_enc, "state_dict") else None,
                "optimizer_audio": optimizer_audio.state_dict(),
                "grid_feature": current_grid_feat,   # [1024]
                "aabb_min": min_bound,               # [3]
                "aabb_max": max_bound,               # [3]
            }, os.path.join(audio_ckpt_dir, f"audio_{tag}.pth"))
            logger.info(f"[Audio] Saved audio checkpoint: {os.path.join(audio_ckpt_dir, f'audio_{tag}.pth')}")
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
        
        iter_end.record()
        
                # ---------------- Train audio after 2000 iterations ----------------
        if raf_loader is not None and iteration >= 2000:
            # 1) get a batch
            try:
                batch_audio = next(raf_iter)
            except StopIteration:
                raf_iter = iter(raf_loader)
                batch_audio = next(raf_iter)

            # 2) build encodings
            # time in [0,1], shape [B, 1]
            time_query = batch_audio['time_query'].to('cuda').float().unsqueeze(-1)
            # normalize time index to [0,1] using per-dataset max frame count
            # here we approximate using batch max to stay simple
            t_norm = (time_query / (time_query.max().clamp_min(1.0)))
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

            # 3) grid feature from ScaffoldGS
            #    returns flattened 1024-d feature (no grad into gaussians)
            with torch.no_grad():
                grid_feat = gaussians.get_feature().to('cuda')  # [1024]
            B = t_feat.shape[0]
            grid_feat = grid_feat.unsqueeze(0).expand(B, -1)   # [B, 1024]

            # 4) concat and forward audio field
            h = torch.cat([grid_feat, t_feat, mic_feat, src_feat, rot_feat], dim=-1)  # [B, in_size]
            pred = audio_field(h)            # [B, 1, F]
            pred = pred.squeeze(1)           # [B, F]  (log-magnitude predicted)

            # 5) compute audio loss in log domain (RAFDataset returns log-magnitude slices)
            gt = batch_audio['data'].to('cuda').float()       # [B, F]
            audio_loss = criterion_audio(pred, gt)

            # 6) optimize audio only
            optimizer_audio.zero_grad(set_to_none=True)
            audio_loss.backward()
            optimizer_audio.step()

            # 7) TB logging
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/audio_train_loss', audio_loss.item(), iteration)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
                
                        # -------- Periodic eval: vision + audio (every 100 steps after 2k) --------
            if iteration >= 2000 and (iteration % 100 == 0):
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

                        # build features (同训练，但 no grad)
                        time_query = eval_batch['time_query'].to('cuda').float().unsqueeze(-1)
                        # 用已知 max_len_frames 归一化时间
                        # 若上面保留了 hop_len/max_len_frames，可复用；否则临时计算
                        # 这里为了简单防御，用批内最大时间
                        t_norm = (time_query / (time_query.max().clamp_min(1.0))).clamp(0.0, 1.0)
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
                            grid_feat = gaussians.get_feature().to('cuda')  # [1024]
                        B = t_feat.shape[0]
                        grid_feat = grid_feat.unsqueeze(0).expand(B, -1)

                        h = torch.cat([grid_feat, t_feat, mic_feat, src_feat, rot_feat], dim=-1)
                        with torch.no_grad():
                            pred = audio_field(h).squeeze(1)  # [B, F], log-magnitude
                        gt = eval_batch['data'].to('cuda').float()
                        audio_eval_mse = torch.mean((pred - gt) ** 2).item()
                        logger.info(f"[Eval/Audio] iter {iteration}: MSE(log-mag) {audio_eval_mse:.6f}")
                        if tb_writer:
                            tb_writer.add_scalar(f'{dataset_name}/eval_audio_mse', audio_eval_mse, iteration)
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
                    
                  
            
                
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000,4_000,5_000,6_000,7_000,8_000,9_000,10_0000,11_000,12_000,13_000,14_000,15_000,16_000,17_000,18_000,19_000,20_000,21_000,22_000,23_000,24_000,25_000,26_000,27_000,28_000,29_000,30_000])
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
