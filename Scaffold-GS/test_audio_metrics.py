#!/usr/bin/env python3
"""
独立测试脚本：按照NeRAF_evaluator.py方法评估完整声学指标
使用NeRAF评估器计算T60、C50、EDT等声学指标，采用Griffin-Lim重建
"""

import os
import sys
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from arguments import RAFDataParserConfig, RAFDataParser, RAFDataset
from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding
import torch.nn as nn
import torch.nn.functional as F
from scene import NeRAFAudioSoundField

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


# 导入声学指标计算函数
try:
    import pyroomacoustics
    from scipy.signal import hilbert
    PRA_AVAILABLE = True
except ImportError:
    PRA_AVAILABLE = False
    print("Warning: pyroomacoustics not available. Some acoustic metrics may not work.")

def setup_logger(log_path):
    """设置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def measure_rt60_advance(signal, sr, decay_db=10, cutoff_freq=200):
    """高级T60测量（来自NeRAF）"""
    if not PRA_AVAILABLE:
        return -1
    
    signal = torch.from_numpy(signal)
    signal = torchaudio.functional.highpass_biquad(
        waveform=signal,
        sample_rate=sr,
        cutoff_freq=cutoff_freq
    )
    signal = signal.cpu().numpy()
    try:
        rt60 = pyroomacoustics.experimental.measure_rt60(signal, sr, decay_db=decay_db, plot=False)
        return rt60
    except:
        return -1

def compute_t60(true_in, gen_in, fs, advanced=True):
    """计算T60指标"""
    ch = true_in.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        try:
            if advanced: 
                true = measure_rt60_advance(true_in[c], sr=fs)
                gen = measure_rt60_advance(gen_in[c], sr=fs)
            else:
                if PRA_AVAILABLE:
                    true = pyroomacoustics.experimental.measure_rt60(true_in[c], fs=fs, decay_db=30)
                    gen = pyroomacoustics.experimental.measure_rt60(gen_in[c], fs=fs, decay_db=30)
                else:
                    true = -1
                    gen = -1
        except:
            true = -1
            gen = -1
        gt.append(true)
        pred.append(gen)
    return np.array(gt), np.array(pred)

def measure_clarity(signal, time=50, fs=44100):
    """测量C50清晰度"""
    h2 = signal**2
    t = int((time/1000)*fs + 1) 
    return 10*np.log10(np.sum(h2[:t])/np.sum(h2[t:]))

def evaluate_clarity(pred_ir, gt_ir, fs):
    """评估C50清晰度"""
    ch = gt_ir.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        pred_clarity = measure_clarity(pred_ir[c,...], fs=fs)
        gt_clarity = measure_clarity(gt_ir[c,...], fs=fs)
        gt.append(gt_clarity)
        pred.append(pred_clarity)
    return np.array(gt), np.array(pred)

def measure_edt(h, fs=44100, decay_db=10):
    """测量EDT早期衰减时间"""
    h = np.array(h)
    fs = float(fs)
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]
    if np.all(energy == 0):
        return np.nan
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]
    i_decay = np.min(np.where(- decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs
    decay_time = t_decay
    est_edt = (60 / decay_db) * decay_time 
    return est_edt

def evaluate_edt(pred_ir, gt_ir, fs):
    """评估EDT早期衰减时间"""
    ch = gt_ir.shape[0]
    gt = []
    pred = []
    for c in range(ch):
        pred_edt = measure_edt(pred_ir[c], fs=fs)
        gt_edt = measure_edt(gt_ir[c], fs=fs)
        gt.append(gt_edt)
        pred.append(pred_edt)
    return np.array(gt), np.array(pred)

def create_visualization_images(pred_stft, gt_stft, pred_waveform, gt_waveform, sample_idx, output_dir, logger):
    """创建GT和预测结果的可视化图像（按照NeRAF_model.py的方法）"""
    try:
        # 确保输出目录存在
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. STFT频谱图可视化
        # 转换为numpy格式
        pred_stft_np = pred_stft.cpu().numpy()  # [F, T]
        gt_stft_np = gt_stft.cpu().numpy()  # [F, T]
        
        # 确保STFT幅度谱不为负（转换为幅度谱进行可视化）
        pred_mag_stft = np.maximum(np.exp(pred_stft_np) - 1e-3, 1e-6)  # 转换为幅度谱
        gt_mag_stft = np.maximum(np.exp(gt_stft_np) - 1e-3, 1e-6)  # 转换为幅度谱
        
        # 计算归一化范围（按照NeRAF方法）
        min_val = min(pred_mag_stft.min(), gt_mag_stft.min())
        max_val = max(pred_mag_stft.max(), gt_mag_stft.max())
        
        # 归一化到[0,1]
        pred_norm = (pred_mag_stft - min_val) / (max_val - min_val)
        gt_norm = (gt_mag_stft - min_val) / (max_val - min_val)
        
        # 应用viridis颜色映射
        pred_colored = cm.viridis(pred_norm)[..., :3]  # [F, T, 3]
        gt_colored = cm.viridis(gt_norm)[..., :3]  # [F, T, 3]
        
        # 计算差异图（基于幅度谱）
        diff_mag = pred_mag_stft - gt_mag_stft
        diff_norm = (diff_mag - min_val) / (max_val - min_val)
        diff_colored = cm.viridis(diff_norm)[..., :3]  # [F, T, 3]
        
        # 创建对比图（预测|GT）
        comparison_stft = np.concatenate([pred_colored, gt_colored], axis=1)  # [F, 2T, 3]
        
        # 保存STFT对比图
        stft_path = os.path.join(vis_dir, f"sample_{sample_idx:04d}_stft_comparison.png")
        plt.figure(figsize=(12, 8))
        plt.imshow(comparison_stft.transpose(1, 0, 2))  # 转置以正确显示
        plt.title(f"Sample {sample_idx}: STFT Comparison (Predicted | Ground Truth)")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(label="Log Magnitude")
        plt.tight_layout()
        plt.savefig(stft_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存差异图
        diff_path = os.path.join(vis_dir, f"sample_{sample_idx:04d}_stft_diff.png")
        plt.figure(figsize=(8, 6))
        plt.imshow(diff_colored.transpose(1, 0, 2))
        plt.title(f"Sample {sample_idx}: STFT Difference (Predicted - Ground Truth)")
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar(label="Log Magnitude Difference")
        plt.tight_layout()
        plt.savefig(diff_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. 波形可视化
        # 转换为numpy格式
        pred_wav_np = pred_waveform.cpu().numpy() if torch.is_tensor(pred_waveform) else pred_waveform
        gt_wav_np = gt_waveform.cpu().numpy() if torch.is_tensor(gt_waveform) else gt_waveform
        
        # 创建时间轴
        fs = 48000  # 假设采样率
        time_axis = np.arange(len(pred_wav_np)) / fs
        
        # 波形对比图
        plt.figure(figsize=(15, 10))
        
        # 子图1：预测波形
        plt.subplot(3, 1, 1)
        plt.plot(time_axis, pred_wav_np, 'b-', linewidth=0.8)
        plt.title(f"Sample {sample_idx}: Predicted Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        # 子图2：GT波形
        plt.subplot(3, 1, 2)
        plt.plot(time_axis, gt_wav_np, 'r-', linewidth=0.8)
        plt.title(f"Sample {sample_idx}: Ground Truth Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        # 子图3：对比图
        plt.subplot(3, 1, 3)
        plt.plot(time_axis, pred_wav_np, 'b-', linewidth=0.8, label='Predicted', alpha=0.7)
        plt.plot(time_axis, gt_wav_np, 'r-', linewidth=0.8, label='Ground Truth', alpha=0.7)
        plt.title(f"Sample {sample_idx}: Waveform Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        waveform_path = os.path.join(vis_dir, f"sample_{sample_idx:04d}_waveform_comparison.png")
        plt.savefig(waveform_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. 频谱分析图
        plt.figure(figsize=(12, 8))
        
        # 计算FFT
        pred_fft = np.fft.fft(pred_wav_np)
        gt_fft = np.fft.fft(gt_wav_np)
        freqs = np.fft.fftfreq(len(pred_wav_np), 1/fs)
        
        # 只取正频率部分
        pos_freqs = freqs[:len(freqs)//2]
        pred_mag = np.abs(pred_fft[:len(freqs)//2])
        gt_mag = np.abs(gt_fft[:len(freqs)//2])
        
        # 确保幅度谱不为负（虽然FFT幅度通常不为负，但为了安全起见）
        pred_mag = np.maximum(pred_mag, 1e-6)
        gt_mag = np.maximum(gt_mag, 1e-6)
        
        mag_mse = np.mean((pred_mag - gt_mag) ** 2)
        print("mag_mse:",mag_mse)
        
        plt.subplot(2, 1, 1)
        plt.semilogy(pos_freqs, pred_mag, 'b-', linewidth=0.8, label='Predicted')
        plt.semilogy(pos_freqs, gt_mag, 'r-', linewidth=0.8, label='Ground Truth')
        plt.title(f"Sample {sample_idx}: Frequency Spectrum Comparison")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 相位谱
        pred_phase = np.angle(pred_fft[:len(freqs)//2])
        gt_phase = np.angle(gt_fft[:len(freqs)//2])
        
        phase_mse = np.mean((pred_phase - gt_phase) ** 2)
        print("phase_mse:",phase_mse)
        
        plt.subplot(2, 1, 2)
        plt.plot(pos_freqs, pred_phase, 'b-', linewidth=0.8, label='Predicted')
        plt.plot(pos_freqs, gt_phase, 'r-', linewidth=0.8, label='Ground Truth')
        plt.title(f"Sample {sample_idx}: Phase Spectrum Comparison")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase (rad)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        spectrum_path = os.path.join(vis_dir, f"sample_{sample_idx:04d}_spectrum_comparison.png")
        plt.savefig(spectrum_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization images saved for sample {sample_idx}")
        return {
            'stft_comparison': stft_path,
            'stft_diff': diff_path,
            'waveform_comparison': waveform_path,
            'spectrum_comparison': spectrum_path
        }
        
    except Exception as e:
        logger.warning(f"Failed to create visualization for sample {sample_idx}: {e}")
        return None

def extract_local_features_from_grid(full_grid, speaker_pose, listener_pose, min_bound, max_bound, pooling_net=None, local_size=9, device='cuda'):
    """从完整网格中提取局部特征
    
    参数:
        full_grid: [35, 128, 128, 128] 完整网格
        speaker_pose: [3] 扬声器位置
        listener_pose: [3] 听者位置  
        min_bound: [3] 网格最小边界
        max_bound: [3] 网格最大边界
        pooling_net: 池化网络（可选，如果None则使用平均池化）
        local_size: int, 局部窗口大小（默认9）
        device: 设备
    
    返回:
        [70] 局部特征（扬声器9³ + 听者9³拼接后池化）
    """
    grid_size = 128
    half_size = local_size // 2
    
    # 计算体素大小
    voxel_size = (max_bound - min_bound) / grid_size
    
    # 将位置转换为网格索引
    speaker_idx = ((speaker_pose - min_bound) / voxel_size).long().clamp(0, grid_size - 1)
    listener_idx = ((listener_pose - min_bound) / voxel_size).long().clamp(0, grid_size - 1)
    
    # 提取扬声器周围9×9×9
    speaker_min = (speaker_idx - half_size).clamp(0, grid_size - local_size)
    speaker_max = speaker_min + local_size
    speaker_local = full_grid[
        :,
        speaker_min[0]:speaker_max[0],
        speaker_min[1]:speaker_max[1],
        speaker_min[2]:speaker_max[2]
    ]  # [35, 9, 9, 9]
    
    # 提取听者周围9×9×9
    listener_min = (listener_idx - half_size).clamp(0, grid_size - local_size)
    listener_max = listener_min + local_size
    listener_local = full_grid[
        :,
        listener_min[0]:listener_max[0],
        listener_min[1]:listener_max[1],
        listener_min[2]:listener_max[2]
    ]  # [35, 9, 9, 9]
    
    # 拼接：[70, 9, 9, 9]
    combined = torch.cat([speaker_local, listener_local], dim=0)
    
    # 池化
    if pooling_net is not None:
        # 使用训练的池化网络
        combined_batch = combined.unsqueeze(0)  # [1, 70, 9, 9, 9]
        pooled = pooling_net(combined_batch)  # [1, 70, 1, 1, 1]
        pooled = pooled.view(-1)  # [70]
    else:
        # 简单平均池化：[70, 9, 9, 9] -> [70]
        pooled = combined.mean(dim=[1, 2, 3])  # [70]
    
    return pooled

def load_audio_model(checkpoint_path, device, use_local_features=True):
    """加载音频模型
    
    参数:
        checkpoint_path: 音频检查点路径
        device: 设备
        use_local_features: 是否使用局部特征提取（从完整网格中提取）
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 重建编码器
    time_enc = NeRFEncoding(in_dim=1, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
    pos_enc = NeRFEncoding(in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True)
    rot_enc = SHEncoding(levels=4, implementation="tcnn")
    
    time_dim = time_enc.get_out_dim()
    pos_dim = pos_enc.get_out_dim()
    rot_dim = rot_enc.get_out_dim()
    
    # 根据是否使用局部特征确定grid_feat_dim
    pooling_net = None
    if use_local_features:
        grid_feat_dim = 70  # 局部特征维度
        print(f"✓ Using LOCAL features (70-dim) extracted from full grid")
        if checkpoint.get("local_pooling_net") is not None:
                # 需要导入LocalPoolingNet类
                try:
                    from scene.gaussian_model import LocalPoolingNet
                    pooling_net = LocalPoolingNet().to(device)
                    pooling_net.load_state_dict(checkpoint["local_pooling_net"])
                    pooling_net.eval()
                    print(f"✓ Loaded local_pooling_net from checkpoint")
                except Exception as e:
                    print(f"⚠ Failed to load LocalPoolingNet class: {e}")
                    print(f"⚠ Will use average pooling instead")
                    pooling_net = None
        else:
            print(f"⚠ local_pooling_net not found in checkpoint, will use average pooling")
    else:
        # 使用完整网格特征
        full_grid = checkpoint["grid_feature"]
        if isinstance(full_grid, np.ndarray):
            full_grid_shape = full_grid.shape
        else:
            full_grid_shape = tuple(full_grid.size()) if hasattr(full_grid, 'size') else 'unknown'
        
        # 如果是完整网格[35,128,128,128]，flatten它
        if full_grid_shape == (35, 128, 128, 128):
            grid_feat_dim = 35 * 128 * 128 * 128
            print(f"✓ Using full grid (will be flattened from [35,128,128,128] to {grid_feat_dim}-dim)")
        else:
            grid_feat_dim = full_grid.size if isinstance(full_grid, np.ndarray) else len(full_grid)
            print(f"✓ Using GLOBAL features from checkpoint ({grid_feat_dim}-dim)")
    
    # 重建音频网络
    audio_W = 512
    audio_F = 513
    in_size = grid_feat_dim + time_dim + 2 * pos_dim + rot_dim
    
    print(f"Audio network input size: {in_size} (grid_feat: {grid_feat_dim}, time: {time_dim}, pos: {pos_dim}×2, rot: {rot_dim})")
    
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
    time_enc = time_enc.to(device)
    pos_enc = pos_enc.to(device)
    rot_enc = rot_enc.to(device)
    audio_field = audio_field.to(device)
    audio_field.eval()
    
    return time_enc, pos_enc, rot_enc, audio_field, checkpoint, pooling_net

def evaluate_audio_field(model_path, raf_data_root, checkpoint_name=None, max_samples=None, logger=None, save_visualizations=True, max_vis_samples=10, use_local_features=True):
    """按照NeRAF_evaluator.py方法评估完整声学指标
    
    参数:
        use_local_features: 是否使用局部特征提取（默认True）
    """
    
    if logger is None:
        logger = setup_logger(os.path.join(model_path, "audio_metrics_test.log"))
    
    logger.info("=" * 80)
    logger.info("AUDIO FIELD EVALUATION (NeRAF Method)")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"RAF data: {raf_data_root}")
    logger.info(f"Use local features: {use_local_features}")
    
    # 1. 查找音频检查点
    audio_ckpt_dir = os.path.join(model_path, "audio_ckpts")
    if not os.path.exists(audio_ckpt_dir):
        logger.error(f"Audio checkpoint directory not found: {audio_ckpt_dir}")
        return
    
    audio_ckpt_files = [f for f in os.listdir(audio_ckpt_dir) if f.startswith('audio_') and f.endswith('.pth')]
    if not audio_ckpt_files:
        logger.error(f"No audio checkpoint files found in {audio_ckpt_dir}")
        return
    
    # 选择检查点
    if checkpoint_name:
        ckpt_path = os.path.join(audio_ckpt_dir, checkpoint_name)
        if not os.path.exists(ckpt_path):
            logger.error(f"Specified checkpoint not found: {ckpt_path}")
            return
    else:
        # 选择最新的检查点
        audio_ckpt_files.sort()
        latest_ckpt = audio_ckpt_files[-1]
        ckpt_path = os.path.join(audio_ckpt_dir, latest_ckpt)
    
    logger.info(f"Using checkpoint: {os.path.basename(ckpt_path)}")
    
    # 2. 加载模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    try:
        time_enc, pos_enc, rot_enc, audio_field, checkpoint, pooling_net = load_audio_model(
            ckpt_path, device, use_local_features=use_local_features
        )
        logger.info("Model loaded successfully")
        if pooling_net is not None:
            logger.info("✓ Using trained local_pooling_net for feature extraction")
        elif use_local_features:
            logger.info("⚠ Using simple average pooling (local_pooling_net not found)")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # 3. 获取参数
    raf_fs = checkpoint.get("raf_fs", 48000)
    raf_max_len_s = checkpoint.get("raf_max_len_s", 0.32)
    hop_len = checkpoint.get("hop_len", 256)
    max_len_frames = checkpoint.get("max_len_frames", 60)
    
    logger.info(f"RAF parameters: fs={raf_fs}, max_len_s={raf_max_len_s}, hop_len={hop_len}")
    
    # 4. 设置STFT参数（按照NeRAF方法）
    if raf_fs == 48000:
        n_fft = 1024
        win_length = 512
        hop_length = 256
    elif raf_fs == 16000:
        n_fft = 512
        win_length = 256
        hop_length = 128
    else:
        logger.error(f"Unsupported sample rate: {raf_fs}")
        return
    
    # 5. 创建NeRAF评估器
    try:
        # 导入NeRAF评估器
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'NeRAF'))
        from NeRAF.NeRAF_evaluator import RAFEvaluator
        from NeRAF.NeRAF_evaluator import STFTLoss
        
        evaluator = RAFEvaluator(fs=raf_fs)
        logger.info("NeRAF evaluator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize NeRAF evaluator: {e}")
        return
    
    # 6. 加载评估数据 - 使用完整音频评估
    try:
        dp_cfg = RAFDataParserConfig(data=Path(raf_data_root))
        dp = RAFDataParser(dp_cfg)
        dpo_test = dp.get_dataparser_outputs(split="test")
        
        # 创建完整音频评估数据集
        raf_eval_ds = RAFDataset(
            dataparser_outputs=dpo_test,
            mode='eval_image',  # 完整音频评估，包含波形数据
            max_len=max_len_frames,
            max_len_time=raf_max_len_s,
            wav_path=os.path.join(raf_data_root, 'data'),
            fs=raf_fs, 
            hop_len=hop_len,
        )
        
        logger.info(f"Loaded evaluation dataset: {len(raf_eval_ds)} samples")
        
    except Exception as e:
        logger.error(f"Failed to load evaluation dataset: {e}")
        return
    
    
    # 7. 创建数据加载器
    if max_samples is None:
        # 评估所有样本
        eval_samples = len(raf_eval_ds)
        eval_loader = DataLoader(raf_eval_ds, batch_size=1, shuffle=False, num_workers=0)
        logger.info(f"Evaluating ALL {eval_samples} samples...")
    else:
        # 评估指定数量的样本
        eval_samples = min(max_samples, len(raf_eval_ds))
        eval_subset = torch.utils.data.Subset(raf_eval_ds, list(range(eval_samples)))
        eval_loader = DataLoader(eval_subset, batch_size=1, shuffle=False, num_workers=0)
        logger.info(f"Evaluating {eval_samples} samples (out of {len(raf_eval_ds)} total)...")
    
    # 8. 执行评估 - 按照NeRAF方法计算完整声学指标
    all_metrics = []
    
    for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating audio metrics")):
        try:
            with torch.no_grad():
                # 获取真实数据
                gt_stft = batch['data'].to(device).float()  # [1, F, T]
                gt_waveform = batch['waveform'].to(device).float()  # [1, T]
                
                # 重建完整STFT
                # max_len = gt_stft.shape[-1]
                # time_query = torch.arange(0, max_len, 1, device=device).unsqueeze(-1).float() / float(max_len - 1.0)
                
                max_len_frames = checkpoint.get("max_len_frames", 60)  # 从checkpoint获取
                time_query = torch.arange(0, max_len_frames, 1, device=device).unsqueeze(-1).float() / float(max_len_frames - 1.0)
                
                t_feat = time_enc(time_query)
                
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
                
                # 提取网格特征：局部特征或全局特征
                if use_local_features:
                    # 从完整网格中提取局部特征
                    # 加载完整网格 [35, 128, 128, 128]
                    full_grid = torch.tensor(checkpoint["grid_feature"]).to(device)
                    # 注意：checkpoint保存的可能是flattened的，需要reshape
                    if full_grid.dim() == 1:
                        # 如果是flatten的，reshape回[35, 128, 128, 128]
                        full_grid = full_grid.view(35, 128, 128, 128)
                    
                    min_bound = torch.tensor(checkpoint["aabb_min"]).to(device)
                    max_bound = torch.tensor(checkpoint["aabb_max"]).to(device)
                    
                    # 提取局部特征 [70]（使用训练的pooling_net或平均池化）
                    local_feat = extract_local_features_from_grid(
                        full_grid,
                        src_pose.squeeze(0),  # [1, 3] -> [3]
                        mic_pose.squeeze(0),  # [1, 3] -> [3]
                        min_bound,
                        max_bound,
                        pooling_net=pooling_net,  # 传递pooling网络
                        local_size=9,
                        device=device
                    )  # [70]
                    
                    # 扩展到所有时间帧
                    B = t_feat.shape[0]
                    grid_feat = local_feat.unsqueeze(0).expand(B, -1)  # [1, 70] -> [B, 70]
                else:
                    # 使用保存的全局特征
                    grid_feat = torch.tensor(checkpoint["grid_feature"]).to(device)
                    # 如果是多维的，flatten
                    if grid_feat.dim() > 1:
                        grid_feat = grid_feat.flatten()
                    B = t_feat.shape[0]
                    grid_feat = grid_feat.unsqueeze(0).expand(B, -1)  # [grid_dim] -> [B, grid_dim]
                
                # 扩展其他特征到所有时间帧
                mic_feat = mic_feat.expand(B, -1)
                src_feat = src_feat.expand(B, -1)
                rot_feat = rot_feat.expand(B, -1)
                
                h = torch.cat([grid_feat, t_feat, mic_feat, src_feat, rot_feat], dim=-1)
                field_outputs = audio_field(h)  # [B, C, F] where B=time_frames, C=channels, F=frequencies
                
                # 转换为正确的格式 [C, F, T]（按照NeRAF的实现方式）
                # field_outputs: [T, C, F] -> permute to [C, F, T]
                pred_log_stft = field_outputs.permute(1, 2, 0)  # [T, C, F] -> [C, F, T]
                
                # GT STFT: [1, 1, 513, 60] -> squeeze to [513, 60] -> unsqueeze to [1, 513, 60]
                gt_stft_squeezed = gt_stft.squeeze(0).squeeze(0)  # [513, 60]
                gt_stft_tensor = gt_stft_squeezed.unsqueeze(0)  # [1, 513, 60] -> [C, F, T]
                
                # 计算STFT误差（在[C, F, T]格式下）
                stft_mse = torch.mean((pred_log_stft - gt_stft_tensor) ** 2).item()
                stft_l1 = torch.mean(torch.abs(pred_log_stft - gt_stft_tensor)).item()
                
                
                
                # 重建波形用于声学指标计算（按照NeRAF方法）
                # 确保幅度谱不为负
                pred_mag_stft = torch.clamp(torch.exp(pred_log_stft) - 1e-3, min=1e-6, max=10000.0)  # [C, F, T]
                
                # 使用Griffin-Lim重建波形（完全按照NeRAF的参数设置）
                # 关键修复：NeRAF使用 n_fft = (N_freq_stft-1)*2 = (513-1)*2 = 1024
                griffin_lim = torchaudio.transforms.GriffinLim(
                    n_fft=(513-1)*2,  # 修复：使用 (513-1)*2 = 1024，而不是直接使用n_fft
                    win_length=win_length,
                    hop_length=hop_length,
                    power=1  # 关键：NeRAF使用power=1，不设置其他参数
                ).to(device)
                
                pred_waveform = griffin_lim(pred_mag_stft)
                pred_waveform = pred_waveform.squeeze(0).cpu().numpy()  # [T]
                
                # GT waveform: [1, 1, T] -> squeeze to [T]
                gt_waveform_squeezed = gt_waveform.squeeze(0).squeeze(0).cpu().numpy()  # [T]
                
                # 为了公平比较，GT波形也应该通过相同的Griffin-Lim重建过程
                # 这样确保预测和GT都经过相同的重建过程
                # 确保GT幅度谱也不为负
                gt_mag_stft = torch.clamp(torch.exp(gt_stft_tensor) - 1e-3, min=1e-6, max=10000.0)  # [C, F, T]
                
                gt_waveform_reconstructed = griffin_lim(gt_mag_stft)
                gt_waveform_reconstructed = gt_waveform_reconstructed.squeeze(0).cpu().numpy()  # [T]
                
                # 确保波形长度一致
                min_len = min(len(pred_waveform), len(gt_waveform_reconstructed))
                pred_waveform = pred_waveform[:min_len]
                gt_waveform_squeezed = gt_waveform_reconstructed[:min_len]  # 使用重建后的GT波形
                
                # 调试信息
                if batch_idx < 30:  # 只打印前5个样本的调试信息
                    logger.info(f"[Sample {batch_idx}] Pred waveform range: [{pred_waveform.min():.6f}, {pred_waveform.max():.6f}]")
                    logger.info(f"[Sample {batch_idx}] GT waveform (reconstructed) range: [{gt_waveform_squeezed.min():.6f}, {gt_waveform_squeezed.max():.6f}]")
                    # 对比原始GT波形和重建GT波形
                    gt_waveform_original = gt_waveform.squeeze(0).squeeze(0).cpu().numpy()
                    logger.info(f"[Sample {batch_idx}] GT waveform (original) range: [{gt_waveform_original.min():.6f}, {gt_waveform_original.max():.6f}]")
                    logger.info(f"[Sample {batch_idx}] GT reconstruction difference: {np.mean(np.abs(gt_waveform_original[:min_len] - gt_waveform_squeezed)):.6f}")
                
                # 创建可视化图像（仅对前几个样本）
                vis_paths = None
                if save_visualizations and batch_idx < max_vis_samples:
                    vis_paths = create_visualization_images(
                        pred_log_stft.squeeze(0), gt_stft_squeezed,  # 转换为[F, T]格式用于可视化
                        pred_waveform, gt_waveform_squeezed,  # 使用重建后的GT波形
                        batch_idx, model_path, logger
                    )
                
                # 使用NeRAF评估器计算完整声学指标
                try:
                    # 准备数据格式（按照NeRAF_model.py的要求）
                    # STFT数据已经是 [C, F, T] 格式
                    pred_log_stft_tensor = pred_log_stft  # [C, F, T]
                    gt_stft_tensor = gt_stft_tensor  # [C, F, T]
                    
                    # 计算幅度谱 - 确保不为负
                    mag_prd = torch.clamp(torch.exp(pred_log_stft_tensor) - 1e-3, min=1e-6, max=10000.0)  # [C, F, T]
                    mag_gt = torch.clamp(torch.exp(gt_stft_tensor) - 1e-3, min=1e-6, max=10000.0)  # [C, F, T]
                    
                    # 转换为numpy格式
                    mag_prd_np = mag_prd.cpu().numpy()
                    mag_gt_np = mag_gt.cpu().numpy()
                    
                    # 波形数据格式：[channels, time]
                    wav_pred_istft = pred_waveform.reshape(1, -1)  # [1, T]
                    wav_gt_istft = gt_waveform_squeezed.reshape(1, -1)  # [1, T] - 重建后的GT波形
                    # 对于声学指标计算，使用原始GT波形（按照NeRAF的做法）
                    gt_waveform_original = gt_waveform.squeeze(0).squeeze(0).cpu().numpy()
                    wav_gt_ff = gt_waveform_original.reshape(1, -1)  # [1, T] - 原始GT波形用于声学指标
                    
                    # log STFT格式 [C, F, T]
                    log_prd = pred_log_stft_tensor.cpu().numpy()  # [C, F, T]
                    log_gt = gt_stft_tensor.cpu().numpy()  # [C, F, T]
                    
                    # 使用NeRAF评估器
                    metrics = evaluator.get_full_metrics(
                        mag_prd_np, mag_gt_np, wav_gt_ff, wav_pred_istft, wav_gt_istft, log_prd, log_gt
                    )
                    
                    # 添加STFT误差
                    metrics['stft_mse'] = stft_mse
                    metrics['stft_l1'] = stft_l1
                    
                except Exception as e:
                    logger.warning(f"Error computing NeRAF metrics for sample {batch_idx}: {e}")
                    # 使用备用方法
                    metrics = {
                        'stft_mse': stft_mse,
                        'stft_l1': stft_l1,
                        'audio_T60': 100.0,
                        'audio_total_invalids_T60': 1,
                        'audio_stft_error': 1.0,
                        'audio_EDT': 1.0,
                        'audio_C50': 1.0,
                    }
                
                all_metrics.append(metrics)
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1}/{len(eval_loader)} samples")
                    
        except Exception as e:
            logger.error(f"Error evaluating sample {batch_idx}: {e}")
            continue
    
    # 9. 计算平均指标
    if all_metrics:
        avg_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)
        
        # 10. 保存结果
        results_dir = os.path.join(model_path, "audio_neraf_results")
        os.makedirs(results_dir, exist_ok=True)
        
        checkpoint_basename = os.path.basename(ckpt_path).replace('.pth', '')
        results_file = os.path.join(results_dir, f"neraf_metrics_{checkpoint_basename}.json")
        
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # 转换所有数据
        converted_avg_metrics = convert_numpy_types(avg_metrics)
        converted_all_samples = convert_numpy_types(all_metrics)
        
        with open(results_file, 'w') as f:
            json.dump({
                'checkpoint': checkpoint_basename,
                'num_samples': len(all_metrics),
                'avg_metrics': converted_avg_metrics,
                'all_samples': converted_all_samples
            }, f, indent=2)
        
        # 11. 打印结果
        logger.info("=" * 80)
        logger.info("AUDIO FIELD EVALUATION RESULTS (NeRAF Method)")
        logger.info("=" * 80)
        logger.info(f"Checkpoint: {checkpoint_basename}")
        logger.info(f"Number of samples: {len(all_metrics)}")
        logger.info(f"Evaluation method: NeRAF evaluator with Griffin-Lim reconstruction")
        if save_visualizations:
            logger.info(f"Visualizations saved for first {min(max_vis_samples, len(all_metrics))} samples")
        logger.info("-" * 80)
        
        for key, value in avg_metrics.items():
            if not key.endswith('_std'):
                std_key = f"{key}_std"
                std_value = avg_metrics.get(std_key, 0)
                logger.info(f"{key:30s}: {value:.6f} ± {std_value:.6f}")
        
        logger.info("=" * 80)
        logger.info(f"Results saved to: {results_file}")
        
        return avg_metrics
    else:
        logger.error("No valid samples evaluated")
        return None

def main():
    parser = argparse.ArgumentParser(description="Test audio metrics for trained models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory or checkpoint file")
    parser.add_argument("--raf_data", type=str, required=True, help="Path to RAF dataset")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint name (e.g., 'audio_iter_30000.pth')")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate (default: all samples)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--save_visualizations", action="store_true", default=True, help="Save visualization images")
    parser.add_argument("--max_vis_samples", type=int, default=10, help="Maximum number of samples to visualize")
    parser.add_argument("--use_local_features", action="store_true", default=True, help="Use local features (70-dim) instead of global (default: True)")
    parser.add_argument("--use_global_features", action="store_true", default=False, help="Use global features from checkpoint (overrides --use_local_features)")
    
    args = parser.parse_args()
    
    # 智能处理model_path：如果传入的是checkpoint文件，自动提取目录
    original_model_path = args.model_path
    if os.path.isfile(args.model_path) and args.model_path.endswith('.pth'):
        # 用户传入了checkpoint文件路径
        checkpoint_file = os.path.basename(args.model_path)
        # 获取模型目录（向上两级：audio_ckpts/ -> model_dir/）
        audio_ckpts_dir = os.path.dirname(args.model_path)
        args.model_path = os.path.dirname(audio_ckpts_dir)
        
        # 如果没有指定checkpoint，使用传入的文件
        if args.checkpoint is None:
            args.checkpoint = checkpoint_file
        
        print(f"📌 检测到checkpoint文件路径，自动转换：")
        print(f"   原始路径: {original_model_path}")
        print(f"   模型目录: {args.model_path}")
        print(f"   Checkpoint: {args.checkpoint}")
        print()
    
    # 确定是否使用局部特征
    use_local_features = args.use_local_features and not args.use_global_features
    
    # 设置输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, "audio_metrics_test.log")
    else:
        log_path = os.path.join(args.model_path, "audio_metrics_test.log")
    
    # 设置日志
    logger = setup_logger(log_path)
    
    # 执行评估
    try:
        results = evaluate_audio_field(
            model_path=args.model_path,
            raf_data_root=args.raf_data,
            checkpoint_name=args.checkpoint,
            max_samples=args.max_samples,
            logger=logger,
            save_visualizations=args.save_visualizations,
            max_vis_samples=args.max_vis_samples,
            use_local_features=use_local_features
        )
        
        if results:
            logger.info("Evaluation completed successfully!")
        else:
            logger.error("Evaluation failed!")
            
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
