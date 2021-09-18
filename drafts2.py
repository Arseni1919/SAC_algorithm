import matplotlib.pyplot as plt
import torch

from alg_net import CriticNet, ActorNet
from alg_constrants_amd_packages import *

action_net = ActorNet(10, 2)
dist = torch.distributions.Normal(torch.tensor([0.0, 30.0]), torch.tensor([0.2, 0.8]))
print(f'dist sample: {dist.sample()}')
print(f'dist rsample: {dist.rsample()}')
print(f'dist entropy: {dist.entropy()}')
list_of_xs = torch.tensor(np.arange(-0.5, 0.5, 0.1))
list_of_ys = []
for i in list_of_xs:
    list_of_ys.append(dist.log_prob(i).detach().numpy())
    # print(f'dist log_prob: {dist.log_prob(0.1)}')
print(f'---')

plt.plot(list_of_xs.detach().numpy(), list_of_ys)
plt.show()
