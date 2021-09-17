from alg_net import CriticNet, ActorNet
from alg_constrants_amd_packages import *

action_net = ActorNet(10, 2)
dist = torch.distributions.Normal(0.0, 1.0)
print(f'dist sample: {dist.sample()}')
print(f'dist rsample: {dist.rsample()}')
print(f'dist log_prob: {dist.log_prob(1.0)}')
print(f'---')