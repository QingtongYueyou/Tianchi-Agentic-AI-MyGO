"""
RL训练好的模型权重存放目录。

训练完成后，此目录将包含：
- policy_best.pt          最优策略网络权重
- value_best.pt           最优价值网络权重
- scorer_best.pt          最优订单评分网络权重
- deadhead_optimizer.pt   空驶优化器权重
- rl_checkpoint_*.pt      训练中间检查点

使用方式：
    from demo.agent.rl_models import PolicyNetwork
    policy = PolicyNetwork.load("demo/agent/models/policy_best.pt")
"""
