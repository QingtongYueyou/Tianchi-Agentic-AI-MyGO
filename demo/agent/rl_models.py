"""
rl_models.py - 物流调度强化学习系统神经网络模型定义
==================================================

本文件定义了4个核心网络，用于辅助物流调度决策：

1. PolicyNetwork (Actor-Critic 共享骨干)
   - 集成位置: rl_integration.py 中的策略决策层
   - 功能: 输入49维状态向量，输出9维动作logits + 状态价值估计
   - 用于PPO算法训练，学习最优调度策略

2. PositionValueNetwork (位置价值预测网络)
   - 集成位置: 替代 ProfitSearchLayer._downstream_value() 的手工计算
   - 功能: 预测在给定(位置, 时间, 月度进度)条件下的未来期望收入
   - 输入10维特征向量，输出预测收入值

3. OrderScoringNetwork (订单评分网络)
   - 集成位置: 替代 HeuristicLayer.score_and_rank() 中的手工 true_net 公式
   - 功能: 学习更精确的订单边际价值评估
   - 双路输入: 司机状态(20维) + 货源特征(8维)，输出订单净收入边际贡献

4. DeadheadOptimizer (空驶预算优化器)
   - 集成位置: 有空驶上限约束的司机(如D003)决策过滤
   - 功能: 预测接该订单后是否会超出月度空驶预算
   - 输入5维特征，输出风险概率[0,1]

所有网络支持:
- GPU/CPU 自动检测
- batch 和 single-sample 推理
- He权重初始化
- state_dict + 超参数的保存/加载
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _get_device() -> torch.device:
    """自动检测可用设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _init_weights(module: nn.Module) -> None:
    """He初始化 (Kaiming Normal) 用于Linear层"""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PolicyNetwork - Actor-Critic 共享骨干网络
# ═══════════════════════════════════════════════════════════════════════════════


class PolicyNetwork(nn.Module):
    """
    Actor-Critic 共享骨干策略网络。
    
    输入: state向量 (49维 float32)
    输出:
      - policy_logits: (batch, 9) 动作概率对数
      - value: (batch, 1) 状态价值估计
    
    架构:
      shared_backbone:
        Linear(49, 256) -> LayerNorm(256) -> ReLU
        Linear(256, 256) -> LayerNorm(256) -> ReLU
        Linear(256, 128) -> ReLU
      policy_head: Linear(128, 9)
      value_head:  Linear(128, 1)
    """

    def __init__(self, state_dim: int = 49, action_dim: int = 9, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._state_dim = state_dim
        self._action_dim = action_dim
        self.device = device or _get_device()

        # 共享骨干网络
        self.shared_backbone = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 策略头: 输出动作logits
        self.policy_head = nn.Linear(128, action_dim)

        # 价值头: 输出状态价值
        self.value_head = nn.Linear(128, 1)

        # He初始化
        self.apply(_init_weights)

        # 移动到设备
        self.to(self.device)

        # 打印网络结构摘要
        total_params = _count_parameters(self)
        print(f"[PolicyNetwork] state_dim={state_dim}, action_dim={action_dim}, "
              f"params={total_params:,}, device={self.device}")

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Args:
            state: (batch, 49) 或 (49,) 状态向量
            
        Returns:
            logits: (batch, 9) 动作原始logits
            value: (batch, 1) 状态价值估计
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        features = self.shared_backbone(state)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据策略选择动作。
        
        Args:
            state: (batch, 49) 或 (49,) 状态向量
            deterministic: 是否确定性选择(argmax)
            
        Returns:
            action: (batch,) 选择的动作索引
            log_prob: (batch,) 动作的log概率
            value: (batch, 1) 状态价值
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估给定状态-动作对，用于PPO更新。
        
        Args:
            state: (batch, 49) 状态向量
            action: (batch,) 动作索引
            
        Returns:
            log_prob: (batch,) 动作的log概率
            entropy: (batch,) 策略熵
            value: (batch, 1) 状态价值
        """
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)

        log_prob = dist.log_prob(action.to(self.device))
        entropy = dist.entropy()
        return log_prob, entropy, value

    def save(self, path: str) -> None:
        """保存模型 state_dict + 超参数"""
        save_dict = {
            'state_dict': self.state_dict(),
            'hyperparams': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
            }
        }
        torch.save(save_dict, path)
        print(f"[PolicyNetwork] 模型已保存到 {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'PolicyNetwork':
        """从文件加载模型"""
        checkpoint = torch.load(path, map_location=device or _get_device())
        hyperparams = checkpoint['hyperparams']
        model = cls(
            state_dim=hyperparams['state_dim'],
            action_dim=hyperparams['action_dim'],
            device=device,
        )
        model.load_state_dict(checkpoint['state_dict'])
        print(f"[PolicyNetwork] 模型已从 {path} 加载")
        return model

    def export_numpy(self, path: str) -> None:
        """将网络权重导出为 numpy .npz 格式，供竞赛部署时无 PyTorch 环境使用。"""
        import numpy as np
        path = str(path)
        if not path.endswith(".npz"):
            path = path.replace(".pt", "").replace(".pth", "") + ".npz"
        
        weights = {}
        for name, param in self.named_parameters():
            # 统一 shared_backbone 前缀为 shared_，与 PolicyNetworkNumpy 中的 key 对齐
            if name.startswith("shared_backbone."):
                suffix = name[len("shared_backbone."):]   # "0.weight" -> "0_weight"
                key = "shared_" + suffix.replace(".", "_")
            else:
                key = name.replace(".", "_")
            weights[key] = param.detach().cpu().numpy()
        
        # 保存超参数
        weights["_state_dim"] = np.array([self._state_dim])
        weights["_action_dim"] = np.array([self._action_dim])
        
        np.savez(path, **weights)
        import logging
        logging.getLogger("agent.rl_models").info("Exported numpy weights to %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PositionValueNetwork - 位置价值预测网络
# ═══════════════════════════════════════════════════════════════════════════════


class PositionValueNetwork(nn.Module):
    """
    位置价值预测网络。
    
    预测在给定(位置, 时间, 月度进度)条件下的未来期望收入，
    替代 ProfitSearchLayer._downstream_value() 的手工计算。
    
    输入: 10维向量
      [0] lat_norm       (lat - 22.0) / 4.0
      [1] lng_norm       (lng - 110.0) / 8.0
      [2] time_of_day    当日时间比例 [0,1]
      [3] day_of_month   月度进度 [0,1]
      [4] income_so_far  已累计收入归一化
      [5] mileage_so_far 已累计里程归一化
      [6] is_weekend
      [7] deadhead_ratio 空驶比例
      [8] orders_pace    当前接单节奏
      [9] time_remaining 剩余月度时间比例
    
    输出: 1维 - 预测该位置的未来期望收入（已归一化）
    
    架构:
      Linear(10, 64) -> ReLU
      Linear(64, 64) -> ReLU
      Linear(64, 32) -> ReLU
      Linear(32, 1)
    """

    def __init__(self, input_dim: int = 10, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.device = device or _get_device()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # He初始化
        self.apply(_init_weights)

        # 移动到设备
        self.to(self.device)

        # 打印网络结构摘要
        total_params = _count_parameters(self)
        print(f"[PositionValueNetwork] input_dim={input_dim}, "
              f"params={total_params:,}, device={self.device}")

    def forward(self, pos_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            pos_features: (batch, 10) 或 (10,) 位置特征向量
            
        Returns:
            value: (batch, 1) 预测的未来期望收入
        """
        if pos_features.dim() == 1:
            pos_features = pos_features.unsqueeze(0)
        pos_features = pos_features.to(self.device)
        return self.network(pos_features)

    def predict(self, lat: float, lng: float, time_min: float, progress_min: float,
                state_dict: Optional[Dict[str, float]] = None) -> float:
        """
        便捷方法：直接接受原始坐标和时间，内部做特征提取。
        
        Args:
            lat: 纬度
            lng: 经度
            time_min: 当日已过分钟数 (0-1440)
            progress_min: 月度已过天数 (0-31)
            state_dict: 可选的额外状态信息字典，包含:
                - income_so_far: 已累计收入
                - mileage_so_far: 已累计里程
                - is_weekend: 是否周末
                - deadhead_ratio: 空驶比例
                - orders_pace: 接单节奏
                - time_remaining: 剩余时间比例
                
        Returns:
            预测的未来期望收入 (float)
        """
        state_dict = state_dict or {}

        # 特征提取
        features = np.zeros(self.input_dim, dtype=np.float32)
        features[0] = (lat - 22.0) / 4.0          # lat_norm
        features[1] = (lng - 110.0) / 8.0         # lng_norm
        features[2] = time_min / 1440.0            # time_of_day [0,1]
        features[3] = progress_min / 31.0          # day_of_month [0,1]
        features[4] = state_dict.get('income_so_far', 0.0)
        features[5] = state_dict.get('mileage_so_far', 0.0)
        features[6] = float(state_dict.get('is_weekend', 0))
        features[7] = state_dict.get('deadhead_ratio', 0.0)
        features[8] = state_dict.get('orders_pace', 0.0)
        features[9] = state_dict.get('time_remaining', 1.0)

        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.network(tensor)
        return float(value.item())

    def save(self, path: str) -> None:
        """保存模型 state_dict + 超参数"""
        save_dict = {
            'state_dict': self.state_dict(),
            'hyperparams': {
                'input_dim': self.input_dim,
            }
        }
        torch.save(save_dict, path)
        print(f"[PositionValueNetwork] 模型已保存到 {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'PositionValueNetwork':
        """从文件加载模型"""
        checkpoint = torch.load(path, map_location=device or _get_device())
        hyperparams = checkpoint['hyperparams']
        model = cls(
            input_dim=hyperparams['input_dim'],
            device=device,
        )
        model.load_state_dict(checkpoint['state_dict'])
        print(f"[PositionValueNetwork] 模型已从 {path} 加载")
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OrderScoringNetwork - 订单评分网络
# ═══════════════════════════════════════════════════════════════════════════════


class OrderScoringNetwork(nn.Module):
    """
    订单评分网络。
    
    替代 HeuristicLayer.score_and_rank() 中的手工 true_net 公式，
    学习更精确的订单边际价值评估。
    
    输入: 双路特征
      司机状态 (20维):
        [0-4]   时空特征 (lat, lng, time_of_day, day_of_month, is_weekend)
        [5-10]  累计状态 (income, mileage, deadhead, orders_today, rest, consecutive_rest)
        [11-15] 约束状态 (penalty, home_dist, visit_dist, urgent_countdown, rest_deficit)
        [16-19] 历史统计 (avg_income_per_order, deadhead_ratio, orders_pace, spatial_density)
      
      货源特征 (8维):
        [0] price_norm          price / 5000.0
        [1] deadhead_km_norm    deadhead / 200.0
        [2] haul_km_norm        haul_km / 1000.0
        [3] cost_time_norm      cost_time_minutes / 480.0
        [4] net_profit_norm     (price - cost) / 3000.0
        [5] end_value_norm      end_position_value / 10000.0
        [6] time_efficiency     price / cost_time_minutes (归一化)
        [7] is_preferred_category  是否为该司机偏好品类
    
    输出: 1维 - 预测该订单的月度净收入边际贡献（归一化，可正可负）
    
    架构:
      state_encoder:  Linear(20, 64) -> ReLU -> Linear(64, 64) -> ReLU
      cargo_encoder:  Linear(8, 32) -> ReLU -> Linear(32, 32) -> ReLU
      combined = concat(state_encoded, cargo_encoded)  # 96维
      scorer:
        Linear(96, 64) -> ReLU
        Dropout(0.1)
        Linear(64, 32) -> ReLU
        Linear(32, 1)
    """

    def __init__(self, state_dim: int = 20, cargo_dim: int = 8, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.cargo_dim = cargo_dim
        self.device = device or _get_device()

        # 司机状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 货源特征编码器
        self.cargo_encoder = nn.Sequential(
            nn.Linear(cargo_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # 评分器: 拼接后打分
        self.scorer = nn.Sequential(
            nn.Linear(96, 64),  # 64 + 32 = 96
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # He初始化
        self.apply(_init_weights)

        # 移动到设备
        self.to(self.device)

        # 打印网络结构摘要
        total_params = _count_parameters(self)
        print(f"[OrderScoringNetwork] state_dim={state_dim}, cargo_dim={cargo_dim}, "
              f"params={total_params:,}, device={self.device}")

    def forward(self, state_features: torch.Tensor, cargo_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            state_features: (batch, 20) 或 (20,) 司机状态特征
            cargo_features: (batch, 8) 或 (8,) 货源特征
            
        Returns:
            score: (batch, 1) 订单边际贡献评分
        """
        if state_features.dim() == 1:
            state_features = state_features.unsqueeze(0)
        if cargo_features.dim() == 1:
            cargo_features = cargo_features.unsqueeze(0)

        state_features = state_features.to(self.device)
        cargo_features = cargo_features.to(self.device)

        state_encoded = self.state_encoder(state_features)
        cargo_encoded = self.cargo_encoder(cargo_features)

        combined = torch.cat([state_encoded, cargo_encoded], dim=-1)
        score = self.scorer(combined)
        return score

    def score_order(self, state_vec: np.ndarray, cargo_dict: Dict[str, float],
                    cost_per_km: float = 3.0) -> float:
        """
        便捷方法：直接接受状态向量和货源字典，返回评分。
        
        Args:
            state_vec: 司机状态向量 (20维 numpy数组)
            cargo_dict: 货源信息字典，应包含:
                - price: 运价
                - deadhead_km: 空驶公里数
                - haul_km: 运输公里数
                - cost_time_minutes: 运输耗时(分钟)
                - end_position_value: 目的地位置价值
                - is_preferred_category: 是否偏好品类 (0/1)
            cost_per_km: 每公里成本
            
        Returns:
            订单评分 (float)
        """
        # 构建货源特征向量
        price = cargo_dict.get('price', 0.0)
        deadhead_km = cargo_dict.get('deadhead_km', 0.0)
        haul_km = cargo_dict.get('haul_km', 0.0)
        cost_time_minutes = cargo_dict.get('cost_time_minutes', 1.0)
        end_value = cargo_dict.get('end_position_value', 0.0)
        is_preferred = cargo_dict.get('is_preferred_category', 0.0)

        total_cost = (deadhead_km + haul_km) * cost_per_km
        net_profit = price - total_cost
        time_efficiency = price / max(cost_time_minutes, 1.0)

        cargo_features = np.array([
            price / 5000.0,                    # price_norm
            deadhead_km / 200.0,               # deadhead_km_norm
            haul_km / 1000.0,                  # haul_km_norm
            cost_time_minutes / 480.0,         # cost_time_norm
            net_profit / 3000.0,               # net_profit_norm
            end_value / 10000.0,               # end_value_norm
            time_efficiency / 10.0,            # time_efficiency (归一化)
            float(is_preferred),               # is_preferred_category
        ], dtype=np.float32)

        state_tensor = torch.from_numpy(np.atleast_2d(state_vec).astype(np.float32)).to(self.device)
        cargo_tensor = torch.from_numpy(cargo_features[np.newaxis, :]).to(self.device)

        with torch.no_grad():
            score = self.forward(state_tensor, cargo_tensor)
        return float(score.item())

    def save(self, path: str) -> None:
        """保存模型 state_dict + 超参数"""
        save_dict = {
            'state_dict': self.state_dict(),
            'hyperparams': {
                'state_dim': self.state_dim,
                'cargo_dim': self.cargo_dim,
            }
        }
        torch.save(save_dict, path)
        print(f"[OrderScoringNetwork] 模型已保存到 {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'OrderScoringNetwork':
        """从文件加载模型"""
        checkpoint = torch.load(path, map_location=device or _get_device())
        hyperparams = checkpoint['hyperparams']
        model = cls(
            state_dim=hyperparams['state_dim'],
            cargo_dim=hyperparams['cargo_dim'],
            device=device,
        )
        model.load_state_dict(checkpoint['state_dict'])
        print(f"[OrderScoringNetwork] 模型已从 {path} 加载")
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DeadheadOptimizer - 空驶预算优化器
# ═══════════════════════════════════════════════════════════════════════════════


class DeadheadOptimizer(nn.Module):
    """
    空驶预算优化器。
    
    预测接该订单后是否会超出月度空驶预算，
    用于 D003 等有空驶上限约束的司机。
    
    输入: 5维
      [0] current_deadhead_ratio   当前空驶/月限
      [1] remaining_days_ratio     剩余天数/31
      [2] cargo_deadhead_km_norm   本单空驶/200
      [3] avg_daily_deadhead_norm  历史日均空驶/100
      [4] urgency                  是否有紧急任务
    
    输出: 1维 [0,1] - 0表示安全，1表示危险（会超限）
    
    架构: 简单MLP
      Linear(5, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1) -> Sigmoid
    """

    def __init__(self, input_dim: int = 5, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.device = device or _get_device()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # He初始化 (注意: Sigmoid前的层仍可用He初始化)
        self.apply(_init_weights)

        # 移动到设备
        self.to(self.device)

        # 打印网络结构摘要
        total_params = _count_parameters(self)
        print(f"[DeadheadOptimizer] input_dim={input_dim}, "
              f"params={total_params:,}, device={self.device}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            features: (batch, 5) 或 (5,) 空驶特征向量
            
        Returns:
            risk_prob: (batch, 1) 超限风险概率 [0,1]
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        features = features.to(self.device)
        return self.network(features)

    def is_risky(self, current_deadhead_km: float, max_monthly_km: float,
                 remaining_days: float, cargo_deadhead_km: float,
                 avg_daily_deadhead: float = 50.0, urgency: float = 0.0,
                 threshold: float = 0.5) -> bool:
        """
        便捷方法：直接判断是否有超限风险。
        
        Args:
            current_deadhead_km: 当前已空驶公里数
            max_monthly_km: 月度空驶上限
            remaining_days: 剩余天数
            cargo_deadhead_km: 本单空驶公里数
            avg_daily_deadhead: 历史日均空驶公里数 (默认50)
            urgency: 是否有紧急任务 (0或1)
            threshold: 风险阈值 (默认0.5)
            
        Returns:
            True 表示有超限风险，建议拒绝
        """
        # 构建特征向量
        features = np.array([
            current_deadhead_km / max(max_monthly_km, 1.0),  # current_deadhead_ratio
            remaining_days / 31.0,                            # remaining_days_ratio
            cargo_deadhead_km / 200.0,                        # cargo_deadhead_km_norm
            avg_daily_deadhead / 100.0,                       # avg_daily_deadhead_norm
            float(urgency),                                   # urgency
        ], dtype=np.float32)

        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            risk_prob = self.network(tensor)
        return float(risk_prob.item()) > threshold

    def save(self, path: str) -> None:
        """保存模型 state_dict + 超参数"""
        save_dict = {
            'state_dict': self.state_dict(),
            'hyperparams': {
                'input_dim': self.input_dim,
            }
        }
        torch.save(save_dict, path)
        print(f"[DeadheadOptimizer] 模型已保存到 {path}")

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'DeadheadOptimizer':
        """从文件加载模型"""
        checkpoint = torch.load(path, map_location=device or _get_device())
        hyperparams = checkpoint['hyperparams']
        model = cls(
            input_dim=hyperparams['input_dim'],
            device=device,
        )
        model.load_state_dict(checkpoint['state_dict'])
        print(f"[DeadheadOptimizer] 模型已从 {path} 加载")
        return model


# ─────────────────────────────────────────────────────────────
# Numpy 推理包装类（竞赛部署用，无需 PyTorch）
# ─────────────────────────────────────────────────────────────


class PolicyNetworkNumpy:
    """PolicyNetwork 的纯 numpy 推理实现，从 .npz 权重文件加载。
    
    设计原则：竞赛环境可能没有 PyTorch，用 numpy 实现前向推理即可。
    训练时用 PolicyNetwork（PyTorch），部署时用 PolicyNetworkNumpy。
    """
    
    def __init__(self) -> None:
        self._weights: dict = {}
        self._loaded = False
    
    def load(self, path: str) -> None:
        """从 .npz 文件加载权重。"""
        import numpy as np
        path = str(path)
        if not path.endswith(".npz"):
            path = path + ".npz"
        
        try:
            data = np.load(path, allow_pickle=False)
            self._weights = {k: data[k] for k in data.files}
            self._loaded = True
            import logging
            logging.getLogger("agent.rl_models").info("PolicyNetworkNumpy loaded from %s", path)
        except Exception as e:
            import logging
            logging.getLogger("agent.rl_models").warning("PolicyNetworkNumpy load failed: %s", e)
            self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def _relu(self, x):
        import numpy as np
        return np.maximum(0, x)
    
    def _layer_norm(self, x, weight_key: str, bias_key: str):
        """LayerNorm 的 numpy 实现。"""
        import numpy as np
        eps = 1e-5
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        if weight_key in self._weights and bias_key in self._weights:
            x_norm = x_norm * self._weights[weight_key] + self._weights[bias_key]
        return x_norm
    
    def _linear(self, x, weight_key: str, bias_key: str):
        """Linear 层的 numpy 实现。"""
        W = self._weights.get(weight_key)
        b = self._weights.get(bias_key)
        if W is None:
            return x
        out = x @ W.T
        if b is not None:
            out = out + b
        return out
    
    def forward(self, state: "np.ndarray") -> tuple:
        """前向推理。
        
        Returns:
            (action_probs, value): action概率数组 (9,) 和 状态价值 float
        """
        import numpy as np
        if not self._loaded:
            # 未加载时返回均匀分布
            n_actions = 9
            return np.ones(n_actions) / n_actions, 0.0
        
        x = np.array(state, dtype=np.float32).flatten()
        
        try:
            # shared backbone: 3层 Linear + LayerNorm + ReLU
            # Layer 1
            x = self._linear(x, "shared_0_weight", "shared_0_bias")
            x = self._layer_norm(x, "shared_1_weight", "shared_1_bias")
            x = self._relu(x)
            # Layer 2
            x = self._linear(x, "shared_3_weight", "shared_3_bias")
            x = self._layer_norm(x, "shared_4_weight", "shared_4_bias")
            x = self._relu(x)
            # Layer 3
            x = self._linear(x, "shared_6_weight", "shared_6_bias")
            x = self._relu(x)
            
            # policy head
            logits = self._linear(x, "policy_head_weight", "policy_head_bias")
            logits = logits - logits.max()  # 数值稳定
            probs = np.exp(logits) / np.exp(logits).sum()
            
            # value head
            value = self._linear(x, "value_head_weight", "value_head_bias")
            value = float(value.flatten()[0]) if hasattr(value, 'flatten') else float(value)
            
            return probs, value
        except Exception:
            n_actions = int(self._weights.get("_action_dim", np.array([9]))[0])
            return np.ones(n_actions) / n_actions, 0.0
    
    def get_action(self, state: "np.ndarray", deterministic: bool = False) -> int:
        """获取动作。deterministic=True 时返回 argmax，否则采样。"""
        import numpy as np
        probs, _ = self.forward(state)
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))


class PositionValueNetworkNumpy:
    """​PositionValueNetwork 的纯 numpy 推理实现。"""
    
    def __init__(self) -> None:
        self._weights: dict = {}
        self._loaded = False
    
    def load(self, path: str) -> None:
        """从 .npz 文件加载权重。"""
        import numpy as np
        path = str(path)
        if not path.endswith(".npz"):
            path = path + ".npz"
        try:
            data = np.load(path, allow_pickle=False)
            self._weights = {k: data[k] for k in data.files}
            self._loaded = True
        except Exception as e:
            import logging
            logging.getLogger("agent.rl_models").warning("PositionValueNetworkNumpy load failed: %s", e)
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def _relu(self, x):
        import numpy as np
        return np.maximum(0, x)
    
    def _linear(self, x, weight_key: str, bias_key: str):
        import numpy as np
        W = self._weights.get(weight_key)
        b = self._weights.get(bias_key)
        if W is None:
            return x
        out = x @ W.T
        if b is not None:
            out = out + b
        return out
    
    def forward(self, pos_features: "np.ndarray") -> float:
        """前向推理，返回位置价值估计。"""
        import numpy as np
        if not self._loaded:
            return 0.0
        x = np.array(pos_features, dtype=np.float32).flatten()
        try:
            x = self._relu(self._linear(x, "encoder_0_weight", "encoder_0_bias"))
            x = self._relu(self._linear(x, "encoder_2_weight", "encoder_2_bias"))
            x = self._relu(self._linear(x, "encoder_4_weight", "encoder_4_bias"))
            x = self._linear(x, "encoder_6_weight", "encoder_6_bias")
            return float(x.flatten()[0])
        except Exception:
            return 0.0


class OrderScoringNetworkNumpy:
    """OrderScoringNetwork 的纯 numpy 推理实现。"""
    
    def __init__(self) -> None:
        self._weights: dict = {}
        self._loaded = False
    
    def load(self, path: str) -> None:
        import numpy as np
        path = str(path)
        if not path.endswith(".npz"):
            path = path + ".npz"
        try:
            data = np.load(path, allow_pickle=False)
            self._weights = {k: data[k] for k in data.files}
            self._loaded = True
        except Exception as e:
            import logging
            logging.getLogger("agent.rl_models").warning("OrderScoringNetworkNumpy load failed: %s", e)
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded
    
    def _relu(self, x):
        import numpy as np
        return np.maximum(0, x)
    
    def _linear(self, x, weight_key: str, bias_key: str):
        W = self._weights.get(weight_key)
        b = self._weights.get(bias_key)
        if W is None:
            return x
        out = x @ W.T
        if b is not None:
            out = out + b
        return out
    
    def forward(self, state_features: "np.ndarray", cargo_features: "np.ndarray") -> float:
        """前向推理，返回订单评分。"""
        import numpy as np
        if not self._loaded:
            return 0.0
        s = np.array(state_features, dtype=np.float32).flatten()
        c = np.array(cargo_features, dtype=np.float32).flatten()
        try:
            # state encoder
            s = self._relu(self._linear(s, "state_encoder_0_weight", "state_encoder_0_bias"))
            s = self._relu(self._linear(s, "state_encoder_2_weight", "state_encoder_2_bias"))
            # cargo encoder
            c = self._relu(self._linear(c, "cargo_encoder_0_weight", "cargo_encoder_0_bias"))
            c = self._relu(self._linear(c, "cargo_encoder_2_weight", "cargo_encoder_2_bias"))
            # combined
            x = np.concatenate([s, c])
            x = self._relu(self._linear(x, "scorer_0_weight", "scorer_0_bias"))
            x = self._linear(x, "scorer_3_weight", "scorer_3_bias")
            return float(x.flatten()[0])
        except Exception:
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 测试代码
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import torch

    print("=" * 60)
    print("=== PolicyNetwork 测试 ===")
    policy = PolicyNetwork(state_dim=49, action_dim=9)
    state = torch.randn(1, 49)
    logits, value = policy(state)
    print(f"  logits shape: {logits.shape}")  # (1, 9)
    print(f"  value shape: {value.shape}")    # (1, 1)

    # 测试 get_action
    action, log_prob, val = policy.get_action(state)
    print(f"  action: {action}, log_prob: {log_prob}, value shape: {val.shape}")

    # 测试 evaluate_actions
    lp, ent, v = policy.evaluate_actions(state, action)
    print(f"  evaluate_actions -> log_prob: {lp}, entropy: {ent}, value shape: {v.shape}")

    # 测试单sample输入
    state_single = torch.randn(49)
    logits_s, value_s = policy(state_single)
    print(f"  单sample -> logits: {logits_s.shape}, value: {value_s.shape}")

    print("\n=== OrderScoringNetwork 测试 ===")
    scorer = OrderScoringNetwork()
    s = torch.randn(1, 20)
    c = torch.randn(1, 8)
    score = scorer(s, c)
    print(f"  score shape: {score.shape}")    # (1, 1)

    # 测试 score_order 便捷方法
    state_vec = np.random.randn(20).astype(np.float32)
    cargo_dict = {
        'price': 2500.0,
        'deadhead_km': 30.0,
        'haul_km': 500.0,
        'cost_time_minutes': 360.0,
        'end_position_value': 5000.0,
        'is_preferred_category': 1.0,
    }
    order_score = scorer.score_order(state_vec, cargo_dict)
    print(f"  score_order: {order_score:.4f}")

    print("\n=== PositionValueNetwork 测试 ===")
    value_net = PositionValueNetwork()
    pos = torch.randn(1, 10)
    val = value_net(pos)
    print(f"  value shape: {val.shape}")      # (1, 1)

    # 测试 predict 便捷方法
    predicted = value_net.predict(lat=23.5, lng=113.2, time_min=720, progress_min=15)
    print(f"  predict: {predicted:.4f}")

    print("\n=== DeadheadOptimizer 测试 ===")
    optimizer = DeadheadOptimizer()
    feat = torch.randn(1, 5)
    risk = optimizer(feat)
    print(f"  risk_prob shape: {risk.shape}")  # (1, 1)
    print(f"  risk_prob value: {risk.item():.4f}")

    # 测试 is_risky 便捷方法
    risky = optimizer.is_risky(
        current_deadhead_km=800.0,
        max_monthly_km=1000.0,
        remaining_days=5.0,
        cargo_deadhead_km=50.0,
    )
    print(f"  is_risky: {risky}")

    # 测试批量推理
    print("\n=== 批量推理测试 ===")
    batch_state = torch.randn(32, 49)
    batch_logits, batch_value = policy(batch_state)
    print(f"  PolicyNetwork batch -> logits: {batch_logits.shape}, value: {batch_value.shape}")

    batch_s = torch.randn(32, 20)
    batch_c = torch.randn(32, 8)
    batch_score = scorer(batch_s, batch_c)
    print(f"  OrderScoringNetwork batch -> score: {batch_score.shape}")

    batch_pos = torch.randn(32, 10)
    batch_val = value_net(batch_pos)
    print(f"  PositionValueNetwork batch -> value: {batch_val.shape}")

    batch_feat = torch.randn(32, 5)
    batch_risk = optimizer(batch_feat)
    print(f"  DeadheadOptimizer batch -> risk: {batch_risk.shape}")

    print("\n所有网络测试通过!")
