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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        
        #新添加raf数据解析
        self.raf_data = ""          # RAF 数据集根目录，例如 /data2/jkx/NeRAF/data/RAF/mix/FurnishedRoom
        self.raf_fs = 48000          # 采样率
        self.raf_max_len = 0.32      # RIR 时长(秒)，仅用于构造 batch 长度参考
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


# # ===== Minimal RAF audio dataparser/dataset (no nerfstudio deps) =====
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import List, Dict, Literal, Optional
# import os, json
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import librosa
# from scipy.spatial.transform import Rotation as T

# @dataclass
# class RAFDataparserOutputs:
#     """Lightweight outputs for RAF; no SceneBox or nerfstudio types."""
#     audios_filenames: List[str]
#     microphone_poses: torch.Tensor     # [N, 3]
#     source_poses: torch.Tensor         # [N, 3]
#     source_rotations: torch.Tensor     # [N, 3] direction cosine in [0,1]
#     def as_dict(self) -> dict:
#         return {
#             "audios_filenames": self.audios_filenames,
#             "microphone_poses": self.microphone_poses,
#             "source_poses": self.source_poses,
#             "source_rotations": self.source_rotations,
#         }

# @dataclass
# class RAFDataParserConfig:
#     data: Path = Path("")

# class RAFDataParser:
#     """Parse RAF split and poses from /metadata/data-split.json and /data/<id>/{rx_pos.txt,tx_pos.txt}."""
#     def __init__(self, config: RAFDataParserConfig):
#         self.config = config

#     def get_dataparser_outputs(self, split: str = "train") -> RAFDataparserOutputs:
#         split_file = os.path.join(self.config.data, "metadata", "data-split.json")
#         with open(split_file) as f:
#             split_dict = json.load(f)
#         if split == "train":
#             split_files = split_dict["train"][0]
#         elif split in ["val", "validation"]:
#             split_files = split_dict["validation"][0]
#         else:
#             split_files = split_dict["test"][0]

#         poses = self._process_poses(split_files)
#         return RAFDataparserOutputs(
#             audios_filenames=split_files,
#             microphone_poses=torch.from_numpy(poses["mic_pose"]).float(),
#             source_poses=torch.from_numpy(poses["source_pose"]).float(),
#             source_rotations=torch.from_numpy(poses["rot"]).float(),
#         )

#     def _process_poses(self, files: List[str]) -> Dict[str, np.ndarray]:
#         mic_list, src_list, rot_list = [], [], []
#         for f in files:
#             rx_file = os.path.join(self.config.data, "data", f, "rx_pos.txt")
#             tx_file = os.path.join(self.config.data, "data", f, "tx_pos.txt")
#             with open(rx_file, "r") as fr:
#                 rx = fr.readlines()
#                 rx = [i.replace("\n","").split(",") for i in rx]
#                 rx = np.array([float(j) for j in rx[0]], dtype=np.float32)   # [x,y,z]
#             with open(tx_file, "r") as ft:
#                 tx = ft.readlines()
#                 tx = [i.replace("\n","").split(",") for i in tx]
#                 tx = np.array([float(j) for j in tx[0]], dtype=np.float32)   # [qx,qy,qz,qw, x,y,z]

#             quat = tx[:4]   # [qx,qy,qz,qw]
#             tx_pose = tx[4:]  # [x,y,z]
#             rx_pose = rx

#             # derive a horizontal direction from quaternion (around y-axis), map to [0,1]
#             r = T.from_quat([quat[0], quat[1], quat[2], quat[3]])
#             spk_rot_deg = r.as_euler('yxz', degrees=True)[0]  # only yaw
#             rad = np.deg2rad(spk_rot_deg)
#             dir_cos = np.array([np.cos(rad), 0.0, np.sin(rad)], dtype=np.float32)
#             dir_cos = (dir_cos + 1.0) / 2.0

#             mic_list.append(rx_pose[None, :3])
#             src_list.append(tx_pose[None, :3])
#             rot_list.append(dir_cos[None, :3])

#         mic_pose = np.concatenate(mic_list, axis=0)
#         source_pose = np.concatenate(src_list, axis=0)
#         rot = np.concatenate(rot_list, axis=0)
#         return {"mic_pose": mic_pose, "source_pose": source_pose, "rot": rot}

# class RAFDataset(Dataset):
#     """Return log-magnitude STFT slice per step for training/eval sampling.
#        Expects:
#          - {root}/metadata/data-split.json
#          - {root}/data/<id>/rir.wav
#          - {root}/data/<id>/{rx_pos.txt, tx_pos.txt}
#     """
#     exclude_batch_keys_from_device: List[str] = ["audios"]

#     def __init__(self,
#                  dataparser_outputs: RAFDataparserOutputs,
#                  mode: Literal["train","eval","inference"] = "train",
#                  max_len: int = 100,            # num STFT frames per audio used for slicing
#                  max_len_time: float = 0.32,    # seconds of RIR considered
#                  wav_path: Path = None,
#                  fs: int = 48000,
#                  hop_len: int = 256):
#         super().__init__()
#         self._dpo = dataparser_outputs
#         self.mode = mode
#         self.wav_path = wav_path
#         self.fs = int(fs)
#         self.hop_len = int(hop_len)
#         # STFT params (match RAF)
#         if self.fs == 48000:
#             self.n_fft = 1024
#             self.win_length = 512
#             self.hop_len = 256
#         elif self.fs == 16000:
#             self.n_fft = 512
#             self.win_length = 256
#             self.hop_len = 128
#         else:
#             raise ValueError("Unsupported fs for RAF: {}".format(self.fs))

#         self.max_len = int(max_len)
#         self.max_len_time = float(max_len_time)

#     def __len__(self):
#         if self.mode in ["train","eval"]:
#             return len(self._dpo.audios_filenames) * self.max_len
#         elif self.mode == "inference":
#             return len(self._dpo.audios_filenames)
#         return len(self._dpo.audios_filenames)

#     def _idx2pair(self, idx: int):
#         return idx // self.max_len, idx % self.max_len

#     def _load_rir(self, audio_id: str) -> np.ndarray:
#         wav_file = os.path.join(self.wav_path, audio_id, "rir.wav")
#         wav, sr = librosa.load(wav_file, sr=None, mono=True)
#         if sr != self.fs:
#             wav = librosa.resample(wav, orig_sr=sr, target_sr=self.fs)
#         # clip to max_len_time seconds
#         max_samps = int(self.max_len_time * self.fs)
#         if wav.shape[0] > max_samps:
#             wav = wav[:max_samps]
#         return wav.astype(np.float32)

#     def _stft_logmag(self, wav: np.ndarray) -> torch.Tensor:
#         # librosa.stft returns complex [F, T]
#         S = librosa.stft(wav, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_length, window="hann", center=True)
#         mag = np.abs(S) + 1e-3
#         logmag = np.log(mag).astype(np.float32)    # [F, T]
#         return torch.from_numpy(logmag)            # [F, T]

#     def _get_slice(self, stft_log: torch.Tensor, t_idx: int) -> torch.Tensor:
#         # stft_log: [F, T] -> slice [F]
#         if t_idx < stft_log.shape[1]:
#             return stft_log[:, t_idx]
#         else:
#             vmin = torch.min(stft_log)
#             return torch.ones(stft_log.shape[0], dtype=stft_log.dtype) * vmin

#     def __getitem__(self, index: int) -> Dict:
#         if self.mode == "inference":
#             audio_idx = index
#             audio_id = self._dpo.audios_filenames[audio_idx]
#             # zeros when no GT
#             stft = torch.zeros(( (self.n_fft // 2)+1, self.max_len ), dtype=torch.float32)
#             waveform = torch.zeros(( int(self.max_len * self.hop_len) ), dtype=torch.float32)
#             mic_pose = self._dpo.microphone_poses[audio_idx]
#             src_pose = self._dpo.source_poses[audio_idx]
#             rot = self._dpo.source_rotations[audio_idx]
#             return {"audio_idx": audio_idx, "data": stft, "waveform": waveform,
#                     "rot": rot, "mic_pose": mic_pose, "source_pose": src_pose}

#         # train/eval (slice by time)
#         stft_id, stft_tp = self._idx2pair(index)
#         audio_id = self._dpo.audios_filenames[stft_id]
#         wav = self._load_rir(audio_id)
#         stft_log = self._stft_logmag(wav)   # [F, T]
#         slice_ft = self._get_slice(stft_log, stft_tp)  # [F]

#         mic_pose = self._dpo.microphone_poses[stft_id]
#         src_pose = self._dpo.source_poses[stft_id]
#         rot = self._dpo.source_rotations[stft_id]

#         if self.mode == "train" or self.mode == "eval":
#             return {"audio_idx": stft_id, "data": slice_ft, "time_query": stft_tp,
#                     "rot": rot, "mic_pose": mic_pose, "source_pose": src_pose}
#         else:
#             # full STFT for evaluation of a complete audio
#             Tlen = min(stft_log.shape[1], self.max_len)
#             full = stft_log[:, :Tlen]
#             if full.shape[1] < self.max_len:
#                 pad = torch.full((full.shape[0], self.max_len - full.shape[1]), full.min())
#                 full = torch.cat([full, pad], dim=1)
#             waveform = torch.from_numpy(wav)
#             return {"audio_idx": stft_id, "data": full.unsqueeze(0), "waveform": waveform.unsqueeze(0),
#                     "rot": rot, "mic_pose": mic_pose, "source_pose": src_pose}
# # ===== end of RAF minimal loader =====

# ===== Minimal RAF audio dataparser/dataset (no nerfstudio deps) =====
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Literal, Optional
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
from scipy.spatial.transform import Rotation as T

@dataclass
class RAFDataparserOutputs:
    """Lightweight outputs for RAF; no SceneBox or nerfstudio types."""
    audios_filenames: List[str]
    microphone_poses: torch.Tensor     # [N, 3]
    source_poses: torch.Tensor         # [N, 3]
    source_rotations: torch.Tensor     # [N, 3] direction cosine in [0,1]
    def as_dict(self) -> dict:
        return {
            "audios_filenames": self.audios_filenames,
            "microphone_poses": self.microphone_poses,
            "source_poses": self.source_poses,
            "source_rotations": self.source_rotations,
        }

@dataclass
class RAFDataParserConfig:
    data: Path = Path("")

class RAFDataParser:
    """Parse RAF split and poses from /metadata/data-split.json and /data/<id>/{rx_pos.txt,tx_pos.txt}."""
    def __init__(self, config: RAFDataParserConfig):
        self.config = config

    def get_dataparser_outputs(self, split: str = "train") -> RAFDataparserOutputs:
        split_file = os.path.join(self.config.data, "metadata", "data-split.json")
        with open(split_file) as f:
            split_dict = json.load(f)
        if split == "train":
            split_files = split_dict["train"][0]
        elif split in ["val", "validation"]:
            split_files = split_dict["validation"][0]
        else:
            split_files = split_dict["test"][0]

        poses = self._process_poses(split_files)
        return RAFDataparserOutputs(
            audios_filenames=split_files,
            microphone_poses=torch.from_numpy(poses["mic_pose"]).float(),
            source_poses=torch.from_numpy(poses["source_pose"]).float(),
            source_rotations=torch.from_numpy(poses["rot"]).float(),
        )

    def _process_poses(self, files: List[str]) -> Dict[str, np.ndarray]:
        mic_list, src_list, rot_list = [], [], []
        for f in files:
            rx_file = os.path.join(self.config.data, "data", f, "rx_pos.txt")
            tx_file = os.path.join(self.config.data, "data", f, "tx_pos.txt")
            with open(rx_file, "r") as fr:
                rx = fr.readlines()
                rx = [i.replace("\n","").split(",") for i in rx]
                rx = np.array([float(j) for j in rx[0]], dtype=np.float32)   # [x,y,z]
            with open(tx_file, "r") as ft:
                tx = ft.readlines()
                tx = [i.replace("\n","").split(",") for i in tx]
                tx = np.array([float(j) for j in tx[0]], dtype=np.float32)   # [qx,qy,qz,qw, x,y,z]

            quat = tx[:4]   # [qx,qy,qz,qw]
            tx_pose = tx[4:]  # [x,y,z]
            rx_pose = rx

            # derive a horizontal direction from quaternion (around y-axis), map to [0,1]
            r = T.from_quat([quat[0], quat[1], quat[2], quat[3]])
            spk_rot_deg = r.as_euler('yxz', degrees=True)[0]  # only yaw
            rad = np.deg2rad(spk_rot_deg)
            dir_cos = np.array([np.cos(rad), 0.0, np.sin(rad)], dtype=np.float32)
            dir_cos = (dir_cos + 1.0) / 2.0

            mic_list.append(rx_pose[None, :3])
            src_list.append(tx_pose[None, :3])
            rot_list.append(dir_cos[None, :3])

        mic_pose = np.concatenate(mic_list, axis=0)
        source_pose = np.concatenate(src_list, axis=0)
        rot = np.concatenate(rot_list, axis=0)
        return {"mic_pose": mic_pose, "source_pose": source_pose, "rot": rot}

class RAFDataset(Dataset):
    """Return log-magnitude STFT slice per step for training/eval sampling.
       Expects:
         - {root}/metadata/data-split.json
         - {root}/data/<id>/rir.wav
         - {root}/data/<id>/{rx_pos.txt, tx_pos.txt}
    """
    exclude_batch_keys_from_device: List[str] = ["audios"]

    def __init__(self,
                 dataparser_outputs: RAFDataparserOutputs,
                 mode: Literal["train","train_full","eval","eval_image","inference"] = "train",
                 max_len: int = 100,            # num STFT frames per audio used for slicing
                 max_len_time: float = 0.32,    # seconds of RIR considered
                 wav_path: Path = None,
                 fs: int = 48000,
                 hop_len: int = 256):
        super().__init__()
        self._dpo = dataparser_outputs
        self.mode = mode
        self.wav_path = wav_path
        self.fs = int(fs)
        self.hop_len = int(hop_len)
        # STFT params (match RAF)
        if self.fs == 48000:
            self.n_fft = 1024
            self.win_length = 512
            self.hop_len = 256
        elif self.fs == 16000:
            self.n_fft = 512
            self.win_length = 256
            self.hop_len = 128
        else:
            raise ValueError("Unsupported fs for RAF: {}".format(self.fs))

        self.max_len = int(max_len)
        self.max_len_time = float(max_len_time)

    def __len__(self):
        if self.mode in ["train","eval"]:
            return len(self._dpo.audios_filenames) * self.max_len
        elif self.mode in ["train_full", "eval_full", "eval_image", "inference"]:
            return len(self._dpo.audios_filenames)
        return len(self._dpo.audios_filenames)

    def _idx2pair(self, idx: int):
        return idx // self.max_len, idx % self.max_len

    def _load_rir(self, audio_id: str) -> np.ndarray:
        wav_file = os.path.join(self.wav_path, audio_id, "rir.wav")
        wav, sr = librosa.load(wav_file, sr=None, mono=True)
        if sr != self.fs:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.fs)
        # clip to max_len_time seconds
        max_samps = int(self.max_len_time * self.fs)
        if wav.shape[0] > max_samps:
            wav = wav[:max_samps]
        return wav.astype(np.float32)

    def _stft_logmag(self, wav: np.ndarray) -> torch.Tensor:
        # librosa.stft returns complex [F, T]
        S = librosa.stft(wav, n_fft=self.n_fft, hop_length=self.hop_len, win_length=self.win_length, window="hann", center=True)
        mag = np.abs(S) + 1e-3
        logmag = np.log(mag).astype(np.float32)    # [F, T]
        return torch.from_numpy(logmag)            # [F, T]

    def _get_slice(self, stft_log: torch.Tensor, t_idx: int) -> torch.Tensor:
        # stft_log: [F, T] -> slice [F]
        if t_idx < stft_log.shape[1]:
            return stft_log[:, t_idx]
        else:
            vmin = torch.min(stft_log)
            return torch.ones(stft_log.shape[0], dtype=stft_log.dtype) * vmin

    def __getitem__(self, index: int) -> Dict:
        if self.mode == "inference":
            audio_idx = index
            audio_id = self._dpo.audios_filenames[audio_idx]
            # zeros when no GT
            stft = torch.zeros(( (self.n_fft // 2)+1, self.max_len ), dtype=torch.float32)
            waveform = torch.zeros(( int(self.max_len * self.hop_len) ), dtype=torch.float32)
            mic_pose = self._dpo.microphone_poses[audio_idx]
            src_pose = self._dpo.source_poses[audio_idx]
            rot = self._dpo.source_rotations[audio_idx]
            return {"audio_idx": audio_idx, "data": stft, "waveform": waveform,
                    "rot": rot, "mic_pose": mic_pose, "source_pose": src_pose}

        elif self.mode == "train_full":
            # 完整音频训练模式 - 返回完整 STFT 和波形，用于训练完整音频
            audio_idx = index
            audio_id = self._dpo.audios_filenames[audio_idx]
            wav = self._load_rir(audio_id)
            stft_log = self._stft_logmag(wav)   # [F, T]
            
            # 裁剪或填充到 max_len
            Tlen = min(stft_log.shape[1], self.max_len)
            full_stft = stft_log[:, :Tlen]
            if full_stft.shape[1] < self.max_len:
                pad = torch.full((full_stft.shape[0], self.max_len - full_stft.shape[1]), full_stft.min())
                full_stft = torch.cat([full_stft, pad], dim=1)
            
            # 返回完整波形和 STFT
            waveform = torch.from_numpy(wav)
            mic_pose = self._dpo.microphone_poses[audio_idx]
            src_pose = self._dpo.source_poses[audio_idx]
            rot = self._dpo.source_rotations[audio_idx]
            
            # 生成时间查询序列
            time_query = torch.arange(Tlen, dtype=torch.float32)
            
            return {
                "audio_idx": audio_idx, 
                "data": full_stft.unsqueeze(0),  # [1, F, T] 保持与 NeRAF 一致
                "waveform": waveform.unsqueeze(0),  # [1, T] 单通道
                "time_query": time_query,  # [T] 时间查询序列
                "rot": rot, 
                "mic_pose": mic_pose, 
                "source_pose": src_pose
            }

        elif self.mode == "eval_full":
            # 完整音频评估模式 - 返回完整 STFT 和波形，用于全音频评估
            audio_idx = index
            audio_id = self._dpo.audios_filenames[audio_idx]
            wav = self._load_rir(audio_id)
            stft_log = self._stft_logmag(wav)   # [F, T]
            
            # 裁剪或填充到 max_len
            Tlen = min(stft_log.shape[1], self.max_len)
            full_stft = stft_log[:, :Tlen]
            if full_stft.shape[1] < self.max_len:
                pad = torch.full((full_stft.shape[0], self.max_len - full_stft.shape[1]), full_stft.min())
                full_stft = torch.cat([full_stft, pad], dim=1)
            
            # 返回完整波形和 STFT
            waveform = torch.from_numpy(wav)
            mic_pose = self._dpo.microphone_poses[audio_idx]
            src_pose = self._dpo.source_poses[audio_idx]
            rot = self._dpo.source_rotations[audio_idx]
            
            # 生成时间查询序列
            time_query = torch.arange(Tlen, dtype=torch.float32)
            
            return {
                "audio_idx": audio_idx, 
                "data": full_stft.unsqueeze(0),  # [1, F, T] 保持与 NeRAF 一致
                "waveform": waveform.unsqueeze(0),  # [1, T] 单通道
                "time_query": time_query,  # [T] 时间查询序列
                "rot": rot, 
                "mic_pose": mic_pose, 
                "source_pose": src_pose
            }

        elif self.mode == "eval_image":
            # 完整音频评估模式 - 返回完整 STFT 和波形，用于计算 T60/EDT/C50
            audio_idx = index
            audio_id = self._dpo.audios_filenames[audio_idx]
            wav = self._load_rir(audio_id)
            stft_log = self._stft_logmag(wav)   # [F, T]
            
            # 裁剪或填充到 max_len
            Tlen = min(stft_log.shape[1], self.max_len)
            full_stft = stft_log[:, :Tlen]
            if full_stft.shape[1] < self.max_len:
                pad = torch.full((full_stft.shape[0], self.max_len - full_stft.shape[1]), full_stft.min())
                full_stft = torch.cat([full_stft, pad], dim=1)
            
            # 返回完整波形和 STFT
            waveform = torch.from_numpy(wav)
            mic_pose = self._dpo.microphone_poses[audio_idx]
            src_pose = self._dpo.source_poses[audio_idx]
            rot = self._dpo.source_rotations[audio_idx]
            
            return {
                "audio_idx": audio_idx, 
                "data": full_stft.unsqueeze(0),  # [1, F, T] 保持与 NeRAF 一致
                "waveform": waveform.unsqueeze(0),  # [1, T] 单通道
                "rot": rot, 
                "mic_pose": mic_pose, 
                "source_pose": src_pose
            }

        else:
            # train/eval (slice by time) - 保持原有逻辑
            stft_id, stft_tp = self._idx2pair(index)
            audio_id = self._dpo.audios_filenames[stft_id]
            wav = self._load_rir(audio_id)
            stft_log = self._stft_logmag(wav)   # [F, T]
            slice_ft = self._get_slice(stft_log, stft_tp)  # [F]

            mic_pose = self._dpo.microphone_poses[stft_id]
            src_pose = self._dpo.source_poses[stft_id]
            rot = self._dpo.source_rotations[stft_id]

            if self.mode == "train":
                return {"audio_idx": stft_id, "data": slice_ft, "time_query": stft_tp,
                        "rot": rot, "mic_pose": mic_pose, "source_pose": src_pose}
            else:  # eval mode
                # 在 eval 模式下，也返回完整波形以支持全指标计算
                waveform = torch.from_numpy(wav)
                return {
                    "audio_idx": stft_id, 
                    "data": slice_ft, 
                    "time_query": stft_tp,
                    "waveform": waveform,  # 添加完整波形
                    "rot": rot, 
                    "mic_pose": mic_pose, 
                    "source_pose": src_pose
                }
# ===== end of RAF minimal loader =====
