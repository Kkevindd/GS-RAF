     
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

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from datetime import datetime

from nerfstudio.field_components.encodings import NeRFEncoding, SHEncoding

class GaussianModel:

    #定义激活函数
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    #初始化输入，必要的参数，mlp网络结构
    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 device="cuda"
                 ):
        self.device = device
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
    

        #初始化参数
          
        self._anchor = torch.empty(0)           #锚点位置 [3,38205]
        self._offset = torch.empty(0)           #相对于锚点的偏移量  [38205,10,3]
        self._anchor_feat = torch.empty(0)      #锚点特征向量
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)           #高斯核缩放向量 [38205,6]
        self._rotation = torch.empty(0)          #高斯核旋转向量 [38205,4]
        self._opacity = torch.empty(0)           #高斯核不透明度 [38205,1]
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(                            #不透明度mlp输出网络结构
            nn.Linear(feat_dim+3+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(                                 #协方差mlp输出网络结构
            nn.Linear(feat_dim+3+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 7*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(                               #颜色mlp输出网络结构
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()
        
     
    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    
    
    def get_full_grid(self):
        """
        获取完整的128×128×128网格（用于保存checkpoint）
        
        返回:
            [35, 128, 128, 128] 完整网格
        """
        device = self.device
        grid_size = 128
        
        # 计算边界和体素大小
        min_bound = self._anchor.min(dim=0)[0]
        max_bound = self._anchor.max(dim=0)[0]
        voxel_size = (max_bound - min_bound) / grid_size
        
        # 初始化网格
        voxel_grid = torch.zeros((35, grid_size, grid_size, grid_size), device=device)
        
        # 计算体素索引
        indices = ((self._anchor - min_bound) / voxel_size).long().clamp(0, grid_size-1)
        d, h, w = indices[:, 0], indices[:, 1], indices[:, 2]
        
        # 创建线性索引
        linear_indices = d * (grid_size * grid_size) + h * grid_size + w
        
        # 使用bincount计算每个体素的锚点数量
        count_flat = torch.bincount(linear_indices, minlength=grid_size**3)
        count_grid = count_flat.view(grid_size, grid_size, grid_size)
        
        # 累加特征
        feat_flat = torch.zeros(32, grid_size**3, device=device)
        for c in range(32):
            feat_flat[c].scatter_add_(0, linear_indices, self._anchor_feat[:, c])
        
        feat_grid = feat_flat.view(32, grid_size, grid_size, grid_size)
        
        # 计算平均特征
        valid_mask = count_grid > 0
        for c in range(32):
            feat_grid[c] = torch.where(valid_mask, feat_grid[c] / count_grid.float(), feat_grid[c])
        
        voxel_grid[:32] = feat_grid
        
        # 计算体素中心坐标
        i, j, k = torch.meshgrid(
            torch.arange(0.5, grid_size, device=device),
            torch.arange(0.5, grid_size, device=device),
            torch.arange(0.5, grid_size, device=device),
            indexing='ij'
        )
        
        voxel_grid[32] = min_bound[0] + i * voxel_size[0]
        voxel_grid[33] = min_bound[1] + j * voxel_size[1]
        voxel_grid[34] = min_bound[2] + k * voxel_size[2]
        
        return voxel_grid  # [35, 128, 128, 128]
    
    def get_feature(self, speaker_pose=None, listener_pose=None, local_size=9):
        """
        提取网格特征
        
        参数:
            speaker_pose: [B, 3] 扬声器位置（可选）
            listener_pose: [B, 3] 听者位置（可选）
            local_size: int, 局部网格大小（默认9）
        
        返回:
            如果提供位姿: [B, 70] 局部特征（扬声器+听者拼接后池化）
            否则: [1024] 全局特征（向后兼容）
        """
        device = self.device
        N = self._anchor.shape[0]
        grid_size = 128
        
        # 计算边界和体素大小
        min_bound = self._anchor.min(dim=0)[0]
        max_bound = self._anchor.max(dim=0)[0]
        voxel_size = (max_bound - min_bound) / grid_size
        
        # 初始化网格（不需要梯度，仅作为local_pooling_net的输入）
        voxel_grid = torch.zeros((35, grid_size, grid_size, grid_size), device=device, requires_grad=False)
        
        # 计算体素索引
        indices = (((self._anchor.detach()) - min_bound.detach()) / voxel_size.detach()).long().clamp(0, grid_size-1)
        d, h, w = indices[:, 0], indices[:, 1], indices[:, 2]
        
        # 创建线性索引
        linear_indices = d * (grid_size * grid_size) + h * grid_size + w
        
        # 使用bincount计算每个体素的锚点数量
        count_flat = torch.bincount(linear_indices, minlength=grid_size**3)
        count_grid = count_flat.view(grid_size, grid_size, grid_size)
        
        # 累加特征（在无梯度模式下累加，避免leaf view就地写入问题）
        feat_flat = torch.zeros(32, grid_size**3, device=device, requires_grad=False)
        for c in range(32):
            feat_flat[c].scatter_add_(0, linear_indices, (self._anchor_feat.detach())[:, c])
        
        feat_grid = feat_flat.view(32, grid_size, grid_size, grid_size)
        
        # 计算平均特征
        valid_mask = count_grid > 0
        for c in range(32):
            feat_grid[c] = torch.where(valid_mask, feat_grid[c] / count_grid.float(), feat_grid[c])
        
        voxel_grid[:32] = feat_grid
        
        # 计算体素中心坐标
        i, j, k = torch.meshgrid(
            torch.arange(0.5, grid_size, device=device),
            torch.arange(0.5, grid_size, device=device),
            torch.arange(0.5, grid_size, device=device),
            indexing='ij'
        )
        
        voxel_grid[32] = min_bound[0] + i * voxel_size[0]
        voxel_grid[33] = min_bound[1] + j * voxel_size[1]
        voxel_grid[34] = min_bound[2] + k * voxel_size[2]
        
        # === 如果没有提供位姿，返回全局特征（向后兼容） ===
        if speaker_pose is None or listener_pose is None:
            # 初始化ResNet3D（全局模式）
            if not hasattr(self, 'resnet3d'):
                self.resnet3d = ResNet3D_helper(
                    in_channels=35, 
                    backbone='resnet152', 
                    pretrained=False, 
                    grid_step=1/grid_size, 
                    N_features=1024
                ).cuda()
            
            feat_grid_full = voxel_grid.unsqueeze(0)
            res_feat = self.resnet3d(feat_grid_full)
            feat_grid_full = res_feat.flatten()
            return feat_grid_full
        
        # === 提取局部特征 ===
        
        # 确保输入是批量的
        if speaker_pose.dim() == 1:
            speaker_pose = speaker_pose.unsqueeze(0)
        if listener_pose.dim() == 1:
            listener_pose = listener_pose.unsqueeze(0)
        
        batch_size = speaker_pose.shape[0]
        
        # 将位置转换为网格索引
        speaker_indices = ((speaker_pose - min_bound) / voxel_size).long()  # [B, 3]
        listener_indices = ((listener_pose - min_bound) / voxel_size).long()  # [B, 3]
        
        # 提取局部网格
        half_size = local_size // 2
        local_features = []
        
        for b in range(batch_size):
            # 扬声器周围的9x9x9网格
            speaker_idx = speaker_indices[b]  # [3]
            speaker_min = (speaker_idx - half_size).clamp(0, grid_size - local_size)
            speaker_max = speaker_min + local_size
            
            speaker_local = voxel_grid[
                :, 
                speaker_min[0]:speaker_max[0],
                speaker_min[1]:speaker_max[1],
                speaker_min[2]:speaker_max[2]
            ]  # [35, 9, 9, 9]
            
            # 听者周围的9x9x9网格
            listener_idx = listener_indices[b]  # [3]
            listener_min = (listener_idx - half_size).clamp(0, grid_size - local_size)
            listener_max = listener_min + local_size
            
            listener_local = voxel_grid[
                :, 
                listener_min[0]:listener_max[0],
                listener_min[1]:listener_max[1],
                listener_min[2]:listener_max[2]
            ]  # [35, 9, 9, 9]
            
            # 拼接扬声器和听者的局部网格 [70, 9, 9, 9]
            combined_local = torch.cat([speaker_local, listener_local], dim=0)
            
            local_features.append(combined_local)
        
        # 堆叠所有batch: [B, 70, 9, 9, 9]
        local_features = torch.stack(local_features, dim=0)
        
        # 初始化局部特征池化网络（保持通道数，只池化空间维度）
        if not hasattr(self, 'local_pooling_net'):
            self.local_pooling_net = LocalPoolingNet().cuda()
        
        # 通过网络：[B, 70, 9, 9, 9] -> [B, 70, 1, 1, 1]
        local_feat = self.local_pooling_net(local_features)
        
        # Flatten: [B, 70, 1, 1, 1] -> [B, 70]
        local_feat = local_feat.view(batch_size, -1)
        
        return local_feat
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size
        
        return data

    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """
        从点云初始化高斯锚点及其属性。
        参数：
            pcd: 输入的点云对象，包含点的坐标。
            spatial_lr_scale: 空间学习率缩放因子。
        """
        self.spatial_lr_scale = spatial_lr_scale  # 设置空间学习率缩放因子
        points = pcd.points[::self.ratio]  # 按照 ratio 下采样点云

        # 如果未指定体素大小，则根据点云的中位数距离自适应确定体素大小
        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()  # 计算点之间的距离
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))  # 取中位数
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')  # 打印初始体素大小

        # 对点云进行体素化采样，去除过于密集的点
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()  # 转为 torch 张量

        # 初始化偏移量张量，形状为 [点数, n_offsets, 3]
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        # 初始化锚点特征张量，形状为 [点数, feat_dim]
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])  # 打印初始化点数

        # 计算每个点的最近距离，用于初始化高斯尺度
        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)  # 6维缩放参数

        # 初始化旋转四元数，默认无旋转（w=1, x/y/z=0）
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 初始化不透明度参数，逆 Sigmoid 映射到 logits 空间
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 将各属性注册为可学习参数
        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))  # 锚点位置
        self._offset = nn.Parameter(offsets.requires_grad_(True))            # 偏移量
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))  # 特征
        self._scaling = nn.Parameter(scales.requires_grad_(True))            # 缩放
        self._rotation = nn.Parameter(rots.requires_grad_(False))            # 旋转
        self._opacity = nn.Parameter(opacities.requires_grad_(False))        # 不透明度
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")  # 2D最大半径


    
    #训练设置
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense  

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")   # get_anchor.shape [38205,3],opacity_accum [38205,1]

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")  #[382050,1]
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
            
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
            
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]
            

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)
        
        # 为ResNet3D添加学习率调度（如果存在）
        if hasattr(self, 'resnet3d'):
            self.resnet3d_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_lr * 0.1,
                                                        lr_final=training_args.feature_lr * 0.01,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)
        
        # 为局部池化网络添加学习率调度（如果存在）
        if hasattr(self, 'local_pooling_net'):
            self.local_pooling_net_scheduler_args = get_expon_lr_func(lr_init=training_args.feature_lr * 0.1,
                                                        lr_final=training_args.feature_lr * 0.01,
                                                        lr_delay_mult=training_args.position_lr_delay_mult,
                                                        max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            if hasattr(self, 'resnet3d') and param_group["name"] == "resnet3d":
                lr = self.resnet3d_scheduler_args(iteration)
                param_group['lr'] = lr
            if hasattr(self, 'local_pooling_net') and param_group["name"] == "local_pooling_net":
                lr = self.local_pooling_net_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    #加载ply
    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    # def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
    #     # update opacity stats
    #     temp_opacity = opacity.clone().view(-1).detach()
    #     temp_opacity[temp_opacity<0] = 0
        
    #     temp_opacity = temp_opacity.view([-1, self.n_offsets])
    #     self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
    #     # update anchor visiting statis
    #     self.anchor_demon[anchor_visible_mask] += 1

    #     # update neural gaussian statis
    #     anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
    #     combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
    #     combined_mask[anchor_visible_mask] = offset_selection_mask
    #     temp_mask = combined_mask.clone()
    #     combined_mask[temp_mask] = update_filter
        
    #     grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #     self.offset_gradient_accum[combined_mask] += grad_norm
    #     self.offset_denom[combined_mask] += 1
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.detach().view(-1)
        temp_opacity[temp_opacity < 0] = 0

        N = self.get_anchor.shape[0]
        K = self.n_offsets
        total = N * K

        # 如果 opacity 无法整齐 reshape 到 [N, K]，说明当前渲染得到的是子集或数量不一致，跳过本次统计避免崩溃
        if temp_opacity.numel() < total:
            return

        # 只取前 total 个，避免偶发比期望更长
        if temp_opacity.numel() > total:
            temp_opacity = temp_opacity[:total]

        # 现在严格 [N, K]
        temp_opacity = temp_opacity.view(N, K)

        # 仅对可见 anchor 累加（左右 shape 一致：sum 后是 [num_visible, 1]）
        vis_mask = anchor_visible_mask
        if vis_mask.dtype != torch.bool:
            vis_mask = vis_mask.bool()
        if vis_mask.numel() != N:
            # 避免无意义累加
            return

        self.opacity_accum[vis_mask] += temp_opacity[vis_mask].sum(dim=1, keepdim=True)

        # update anchor visiting stats
        self.anchor_demon[vis_mask] += 1

        # update neural gaussian stats (保持原逻辑，但加长度检查以防越界)
        anchor_visible_mask_flat = vis_mask.unsqueeze(1).repeat(1, K).view(-1)  # [N*K]
        if anchor_visible_mask_flat.numel() != (self.offset_gradient_accum.numel()):
            # 下面用到的 combined_mask 期望与 offset_* 的长度一致，不一致时跳过
            return

        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        # 注意：offset_selection_mask & update_filter 的长度应等于 N*K；否则需要在上游修复
        if offset_selection_mask.numel() != anchor_visible_mask_flat.numel() or update_filter.numel() != anchor_visible_mask_flat.numel():
            return

        combined_mask[anchor_visible_mask_flat] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        if viewspace_point_tensor.grad is None:
            return
        if viewspace_point_tensor.grad.shape[0] < combined_mask.sum():
            return

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1
        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors
    
    #候选锚点
    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    #生长新的锚点 
    def anchor_growing(self, grads, threshold, offset_mask):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets  #anchor【38205,3】，n_offsets=10,init_length=382050总共的高斯核数量
        for i in range(self.update_depth):    #update_depth=3
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)  #不同层级的阈值
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)  #梯度大于阈值的mask ，需要加更多的点
            candidate_mask = torch.logical_and(candidate_mask, offset_mask) 
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))  #随机掩码用来生长
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)  #计算逻辑和
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)  #高斯核位置
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)  #尺度因子 随层级加深而减小
            cur_size = self.voxel_size*size_factor   #基础体素尺寸 0.005616...乘以尺度因子
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            #候选锚点位置
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            #网格取整
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)

            #找到唯一的体素坐标
            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            #如果有候选锚点初始化他们：
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                #将新锚点参数添加进去
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                


    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
        
        
        
# Code is adapted from: https://github.com/DonGovi/pyramid-detection-3D

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['ResNet3D', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
'''
def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out
'''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #conv2_rep = out
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #conv3_rep = out
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet3D(nn.Module):
    def __init__(self, in_channels, block, layers, grid_step=None, N_features=1024):
        self.in_planes = 64
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=5, stride=2, padding=2, bias=False)  # 128 -> 64
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)    # 64 -> 32
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)    # 32 -> 16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)    # 16 -> 8

        assert N_features in [1024, 2048], 'N_features should be 1024 or 2048'
        self.N_features = N_features    
        if self.N_features == 2048:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)    # 8 -> 4

        # self.layer5 = self._make_layer(block, 256, layers[3], stride=2)    # 4 -> 2

        if grid_step is None:
            grid_step = 1/128

        if grid_step >= 1/64 - 1/512:
            print('grid_size == 1/64')
            if self.N_features == 2048:
                self.avgpool = nn.AvgPool3d(2, stride=1)
            else:
                self.avgpool = nn.AvgPool3d(4, stride=1)
        elif grid_step >= 1/128 - 1/512:
         #   print('grid_size == 1/128')
            if self.N_features == 2048:
                self.avgpool = nn.AvgPool3d(4, stride=1)
            else:
                self.avgpool = nn.AvgPool3d(8, stride=1)
        else:
            print('grid_size == 1/256')
            if self.N_features == 2048:
                self.avgpool = nn.AvgPool3d(8, stride=1)
            else:
                self.avgpool = nn.AvgPool3d(16, stride=1)
        '''
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        '''

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):       # 128 
        c1 = self.conv1(x)      # 64 --> 8 anchor_area
        c1 = self.bn1(c1)         
        c1 = self.relu(c1)
        c2 = self.maxpool(c1)   # 32

        c2 = self.layer1(c2)  # 32 --> 16 anchor_area
        c3 = self.layer2(c2)  # 16 --> 32 anchor_area
        c4 = self.layer3(c3)  # 8
        if self.N_features == 2048:
            c5 = self.layer4(c4) # 4
            out = self.avgpool(c5)
        else: 
            # 1024 features
            out = self.avgpool(c4)
        
        # c6 = self.layer5(c5)  # 4
        return out
        '''
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        '''
        # return c1, c2, c3, c4, c5, c6
        # return c1, c2, c3, c4, c5


def resnet18(in_channels=3, pretrained=False, grid_step=None,N_features=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D(in_channels, BasicBlock, [2, 2, 2, 2], grid_step=grid_step, N_features=N_features, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(in_channels=3, pretrained=False, grid_step=None, N_features=None, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D(in_channels, BasicBlock, [3, 4, 6, 3], grid_step=grid_step,N_features=N_features, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(in_channels=3, pretrained=False, grid_step=None, N_features=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D(in_channels, Bottleneck, [3, 4, 6, 3], grid_step=grid_step,N_features=N_features, **kwargs)
    if pretrained:
        print('Pre-trained')
        model.load_state_dict(torch.utils.model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(in_channels=3, pretrained=False, grid_step=None, N_features=None, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D(in_channels, Bottleneck, [3, 4, 23, 3], grid_step = grid_step, N_features=N_features, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(in_channels=3, pretrained=False, grid_step = None, N_features=None, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D(in_channels, Bottleneck, [3, 8, 36, 3], grid_step = grid_step, N_features=N_features,  **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class ResNet3D_helper(nn.Module):
    backbones = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152
        }
    def __init__(self, in_channels=3, backbone='resnet50', pretrained=False, grid_step=None, N_features=1024):
        super(ResNet3D_helper, self).__init__()

        self.backbone_net = ResNet3D_helper.backbones[backbone](
            in_channels=in_channels,
            pretrained=pretrained,
            grid_step=grid_step, 
            N_features=N_features
        )	

    def forward(self, x):
        return self.backbone_net(x)
        # return resnet_feature_1, resnet_feature_2, resnet_feature_3, resnet_feature_4, resnet_feature_5, resnet_feature_6


class LocalPoolingNet(nn.Module):
    def __init__(self):
        super(LocalPoolingNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(70, 70, kernel_size=3, padding=1, groups=70),
            nn.BatchNorm3d(70),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )

    def forward(self, x):
        return self.net(x)

class SimplePool3D(nn.Module):
    def __init__(self, input_channels=35, hidden_dim=256, output_channels=35, pool_size=8):
        """
        GridMLPWithPooling: 先进行空间下采样，再应用MLP，减少参数量
        
        Args:
            input_channels: 输入通道数，默认为35
            hidden_dim: 隐藏层维度，默认为256
            output_channels: 输出通道数，默认为35
            pool_size: 空间下采样后的尺寸，默认为8
        """
        super(SimplePool3D, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.pool_size = pool_size
        
        # 自适应平均池化，将空间维度降低到pool_size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((pool_size, pool_size, pool_size))
        
        # 计算池化后的输入维度
        self.pooled_dim = input_channels * pool_size * pool_size * pool_size
        
        # 两层MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.pooled_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_channels)
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [B, C, D, H, W]
            
        Returns:
            输出张量，形状为 [B, C, 1, 1, 1]
        """
        B, C, D, H, W = x.shape
        
        # 先进行空间下采样: [B, C, D, H, W] -> [B, C, pool_size, pool_size, pool_size]
        x_pooled = self.adaptive_pool(x)
        
        # 展平: [B, C*pool_size*pool_size*pool_size]
        x_flat = x_pooled.view(B, -1)
        
        # 通过MLP: [B, C*pool_size^3] -> [B, output_channels]
        x_mlp = self.mlp(x_flat)
        
        # 重塑为 [B, output_channels, 1, 1, 1]
        x_out = x_mlp.view(B, self.output_channels, 1, 1, 1)
        
        return x_out
    

    # simple sound field head (same as NeRAF NeRAFAudioSoundField)
class NeRAFAudioSoundField(nn.Module):

    def __init__(self,in_size,W,sound_rez=2,N_frequencies=257):
        super().__init__()
        # 5个全连接层，从输入维度，映射到W维，也就是config中的512维。
        self.soundfield = nn.ModuleList(
            [nn.Linear(in_size, 5096), nn.Linear(5096, 2048), nn.Linear(2048, 1024), nn.Linear(1024, 1024), nn.Linear(1024, W)])
        
        # 每个声道对应一个线性层，映射到N_frequencies维，也就是config中的257维。
        self.STFT_linear = nn.ModuleList(
            [nn.Linear(W, N_frequencies) for _ in range(sound_rez)])
        
    def forward(self, h):
        
        #特征提取
        for i, layer in enumerate(self.soundfield):
            h = layer(h)
            h = F.leaky_relu(h, negative_slope=0.1)

        output = []
        feat = h
        
        #STFT生成
        for i, layer in enumerate(self.STFT_linear):
            h = layer(feat)
            h = F.tanh(h)*10
            # h = F.leaky_relu(h, negative_slope=0.1) # if mag stft prediction
            mono = h.unsqueeze(1) 
            output.append(mono)

        output = torch.cat(output, dim=1) 

        return output
        
