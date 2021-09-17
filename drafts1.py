import numpy as np
from alg_net import CriticNet
from alg_constrants_amd_packages import *


a1 = np.array([[1, 2], [3, 4]])
a2 = np.array([[1, 2], [3, 16]])
mse = (np.square(a1 - a2)).mean()
print(mse)
critic_net1 = CriticNet(10, 1)
critic_net2 = CriticNet(10, 1)
for c1, c2 in zip(critic_net1.parameters(), critic_net2.parameters()):
    mse = (np.square(c1.data.numpy() - c2.data.numpy())).mean()
    print(mse)


for i in range(10):
    b = np.array([0, 0])
    a = np.random.normal(0, ACT_NOISE, 2)
    c = b + a
    print(c)
    d = np.clip(a, -0.1, 0.1)
    print(d)
    print('---')