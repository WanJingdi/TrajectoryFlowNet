import os
import random

###########u_t 不添加到 totalloss 损失函数形式与原始PINNs相同##############
###########不收敛，每次随机结果不一#############################

import torch
import torch.nn as nn
import numpy as np
from numpy import genfromtxt
import time
import matplotlib.pyplot as plt
import scipy.io
from collections import OrderedDict
from pyDOE import lhs
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product, combinations
import pandas as pd
from more_itertools import flatten
from torch.utils.data import Dataset, DataLoader, TensorDataset
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
import cv2
import re
import matplotlib.colors as mcolors
import h5py

os.environ['CUDA_VISIBLE_DEVICES']='5'
np.random.seed(1234)
torch.cuda.manual_seed_all(1234)
torch.manual_seed(1234)
device = torch.device('cuda')

# 自定义第二个神经网络
class Net2(nn.Module):
    def __init__(self, layers_2):
        super(Net2, self).__init__()

        self.depth = len(layers_2) - 1  # 深度是10-1为9
        self.activation = nn.Tanh
        layers_list = list()

        for i in range(self.depth - 1):
            layers_list.append(
                ('layer_%d' % i, nn.Linear(layers_2[i], layers_2[i + 1]))  # 把每一层都加到list中
            )
            # layers_list.append(
            #     ('layer_norm%d' % i, nn.LayerNorm(layers_2[i + 1]))
            # )
            layers_list.append(
                ('activation_%d' % i, self.activation())  # 每一层都加激活函数Tanh
            )

        layers_list.append(
            ('layer_%d' % (self.depth - 1), nn.Linear(layers_2[-2], layers_2[-1]))  # 最后线性输出
        )
        layerDict = OrderedDict(layers_list)  # 用有序字典

        self.layers = nn.Sequential(layerDict)  # 用Sequential容器做一个网络

    def forward(self, x, y, t1, B):
        xyt = torch.cat([x, y, t1], dim=1)
        FF_xyt = input_mapping(xyt, B)
        u_v_p = self.layers(FF_xyt)
        return u_v_p

class TrajectoryNSNet():
    def __init__(self,B):
        self.layers_2 = layers_2
        self.B = B.to(device)

        #将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        # (net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        # 所以经过类型转换这个self.lambda变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

        self.net_2 = Net2(layers_2).to(device)

        self.optimizer = torch.optim.LBFGS(
            self.net_2.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(self.net_2.parameters(), lr=0.001)
        self.iter = 0
        self.net_2.load_state_dict(torch.load("suanli1nopretrainnet2.pth"))

    def net_x_psi_p(self, x, y, t1, B):
        psi_p = self.net_2(x, y, t1, B)
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]
        u = torch.autograd.grad(
            psi, y,
            grad_outputs=torch.ones_like(psi),
            retain_graph=True,  # 由于后面要求二阶导所以此处应设置为true
            create_graph=True
        )[0]
        v = - torch.autograd.grad(
            psi, x,
            grad_outputs=torch.ones_like(psi),
            retain_graph=True,  # 由于后面要求二阶导所以此处应设置为true
            create_graph=True
        )[0]
        # u = psi_p[:, 0:1]
        # v = psi_p[:, 1:2]
        # p = psi_p[:, 2:3]
        return u, v, p


    def net_f(self, x, y, t1, B):
        u, v, p = self.net_x_psi_p(x, y, t1, B)

        u_t = torch.autograd.grad(
            u, t1,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,  # 由于后面要求二阶导所以此处应设置为true
            create_graph=True
        )[0]
        # u对x求导
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_y = torch.autograd.grad(
            u, y,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        # u对x求二阶导
        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]
        u_yy = torch.autograd.grad(
            u_y, y,
            grad_outputs=torch.ones_like(u_y),
            retain_graph=True,
            create_graph=True
        )[0]

        v_t = torch.autograd.grad(
            v, t1,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,  # 由于后面要求二阶导所以此处应设置为true
            create_graph=True
        )[0]
        # u对x求导
        v_x = torch.autograd.grad(
            v, x,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]
        v_y = torch.autograd.grad(
            v, y,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]
        # u对x求二阶导
        v_xx = torch.autograd.grad(
            v_x, x,
            grad_outputs=torch.ones_like(v_x),
            retain_graph=True,
            create_graph=True
        )[0]
        v_yy = torch.autograd.grad(
            v_y, y,
            grad_outputs=torch.ones_like(v_y),
            retain_graph=True,
            create_graph=True
        )[0]

        p_x = torch.autograd.grad(
            p, x,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True
        )[0]
        p_y = torch.autograd.grad(
            p, y,
            grad_outputs=torch.ones_like(p),
            retain_graph=True,
            create_graph=True
        )[0]

        f_u = u_t + u * u_x + v * u_y + p_x - (0.001 / (1 * U * L)) * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y + p_y - (0.001 / (1 * U * L)) * (v_xx + v_yy)

        return f_u, f_v

    def closure(self):
        self.optimizer.zero_grad()
        self.u_pred, self.v_pred, self.p_pred = self.net_x_psi_p(self.x, self.y, self.t1, self.B)
        self.f_u_pred, self.f_v_pred = self.net_f(self.x_f_train, self.y_f_train, self.t1_f_train, self.B)
        # loss_x = torch.mean((self.x - self.x_pred) ** 2) + \
        #          torch.mean((self.y - self.y_pred) ** 2)
        loss_u = torch.mean((self.u - self.u_pred) ** 2) + \
                 torch.mean((self.v - self.v_pred) ** 2)
        # loss_e = torch.mean((self.E_x_pred) ** 2) + \
        #          torch.mean((self.E_y_pred) ** 2)
        loss_f = torch.mean((self.f_u_pred) ** 2) + \
                 torch.mean((self.f_v_pred) ** 2)

        total_loss = loss_u + loss_f

        self.iter += 1
        if self.iter % 10 == 0:
            print('It: %d, total_loss: %.5e, loss_u: %.5e, loss_f: %.5e' % (self.iter,
                                                                                      total_loss.item(),
                                                                                      loss_u.item(),
                                                                                      loss_f.item(),
                                                                                      ))

        total_loss.backward()

        return total_loss

    def train(self, traindata):
        self.net_2.train()
        for step, data in enumerate(traindata):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            self.x = inputs[:, 0:1].clone().detach().requires_grad_(True)
            self.y = inputs[:, 1:2].clone().detach().requires_grad_(True)
            self.t1 = inputs[:, 2:3].clone().detach().requires_grad_(True)
            # self.t2 = self.t1
            self.x_f_train = inputs[:, 3: 3 + multiple]
            self.y_f_train = inputs[:, 3 + multiple:3 + 2 * multiple]
            self.t_f_train = inputs[:, 3 + 2 * multiple:3 + 3 * multiple]
            self.x_f_train = torch.cat([self.x, self.x_f_train], dim=1)
            self.y_f_train = torch.cat([self.y, self.y_f_train], dim=1)
            self.t1_f_train = torch.cat([self.t1, self.t_f_train], dim=1)
            self.x_f_train = torch.flatten(self.x_f_train.T, start_dim=0, end_dim=1)[:,
                             None].clone().detach().requires_grad_(True)
            self.y_f_train = torch.flatten(self.y_f_train.T, start_dim=0, end_dim=1)[:,
                             None].clone().detach().requires_grad_(True)
            self.t1_f_train = torch.flatten(self.t1_f_train.T, start_dim=0, end_dim=1)[:,
                              None].clone().detach().requires_grad_(True)

            self.u = labels[:, 0:1]
            self.v = labels[:, 1:2]

            print(step)
            self.optimizer.step(closure=self.closure)#通过梯度下降执行下一步参数更新
        print('End')

    def predict_uv(self, x_star, y_star, t_star):
        self.x = torch.tensor(x_star, requires_grad=True).float().to(device)
        self.y = torch.tensor(y_star, requires_grad=True).float().to(device)
        self.t = torch.tensor(t_star, requires_grad=True).float().to(device) #将t分离出来
        self.net_2.eval() #设置predict模式
        u_star, v_star, p_star = self.net_x_psi_p(self.x, self.y, self.t, self.B)
        u_star = u_star.detach().cpu().numpy()
        v_star = v_star.detach().cpu().numpy()
        p_star = p_star.detach().cpu().numpy()

        return u_star, v_star, p_star



def input_mapping(x, B):
    # x = x.to('cuda')

    if B is None:
        return x
    else:
        # 将 B 移动到 GPU
        # B = B.to('cuda')

        # 计算 x_proj
        x_proj = (2. * torch.pi * x) @ B.T

        # 计算 sin 和 cos
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def addbatch(data_train,data_test,batchsize):
    """
    设置batch
    :param data_train: 输入
    :param data_test: 标签
    :param batchsize: 一个batch大小
    :return: 设置好batch的数据集
    """

    data = TensorDataset(data_train,data_test)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=True, pin_memory=True)#shuffle是是否打乱数据集，可自行设置

    return data_loader
        # torch.save(self.net_2.state_dict(), "suanli3net2pretrain_test_FFN.pth")


np.set_printoptions(precision=20)
B_gauss_1 = np.random.randn(50, 4)
B_gauss_1 = B_gauss_1 * 0.1
B_gauss_2 = np.random.randn(50, 3)
B_gauss_2 = B_gauss_2 * 0.1
layers_1 = [50 * 2, 40, 40, 40, 40, 2]
layers_2 = [50 * 2, 60, 60, 60, 60, 60, 60, 2]
# Load Data

filename = "fangqiang.csv"

x_Matrix = pd.read_csv(filename, header=3)
train_data = np.array(x_Matrix)

#print('从fluent导出的训练数据为：')
train_data = np.delete(train_data, 2, axis=1)
# train_data = np.delete(train_data, 4, axis=1)
# train_data = np.delete(train_data, 0, axis=0)
train_data = train_data.astype(float)
np.set_printoptions(suppress=True)
#print(train_data_test)
# index = np.argwhere(train_data[:, 3:4] == 0.500999987 or train_data[:, 3:4] == 0.550999999 or train_data[:, 3:4] == 0.550999999)[:, 0:1].flatten()
# N_p = 500
# idx_p = np.random.choice(index.shape[0], N_p, replace=False)
# index = index[idx_p]

id = train_data[:, 2:3]
x = train_data[:, 0:1]
y = train_data[:, 1:2]
t = train_data[:, 3:4]
p = train_data[:, 6:7]
u = train_data[:, 4:5]
v = train_data[:, 5:6]

extrater = []
for i in range(len(x)):
    if t[i]<22:
        extrater.append(i)

id = np.delete(id, extrater, axis=0)
x = np.delete(x, extrater, axis=0)
y = np.delete(y, extrater, axis=0)
t = np.delete(t, extrater, axis=0)
p = np.delete(p, extrater, axis=0)
u = np.delete(u, extrater, axis=0)
v = np.delete(v, extrater, axis=0)




def find_diff_positions(arr):
    diff_positions = []
    diff_positions.append(0)
    for i in range(1, len(arr)):
        if arr[i] - arr[i-1] != 0:
            diff_positions.append(i)
    return diff_positions
index = find_diff_positions(id)
index.append(int(len(id)))
# index = np.argwhere(train_data[:, 3:4] == 10.0200005)[:, 0:1].flatten()
# for i in index:
#     print(id[i])
# x = [i-0.01 for i in x]
# y = [i+0.02 for i in y]
# u_min = min(abs(u))
# v_min = min(abs(v))
# uv_min = float(min(u_min, v_min))

# x = [i+0.5 for i in x]
# y = [i+0.5 for i in y]
t = [i-22 for i in t]
# u = [i-uv_min for i in u]
# v = [i-uv_min for i in v]

x = np.array(x).flatten()[:, None]
y = np.array(y).flatten()[:, None]
p = np.array(p).flatten()[:, None]
u = np.array(u).flatten()[:, None]
v = np.array(v).flatten()[:, None]
t = np.array(t).flatten()[:, None]

x_ab = max(abs(x))
y_ab = max(abs(y))
L = float(max(x_ab, y_ab))
u_ab = max(abs(u))
v_ab = max(abs(v))
U = float(max(u_ab, v_ab))


##########################################
u_mag_cor = (u ** 2 + v ** 2) ** 0.5
u_mag_plot_cor = float(max(u_mag_cor))


##########################################
# U = 1
# L = 0.01
x_train = x / L
y_train = y / L
t_train = t * (U / L)
u_train = u / U
v_train = v / U
p_train = p / (1 * (U ** 2))


k = 0
x00_train = []
y00_train = []
x0_train = []
y0_train = []
x_net =[]
y_net =[]
t_net =[]
u_net =[]
v_net =[]
p_net =[]
# par_num = []
# delta_i = 1400

for i in index[:-1]:
    x0 = x_train[i]
    y0 = y_train[i]
    delta_i = index[k+1] - index[k]
    k = k+1
    for j in range(delta_i):
        x00_train.append(x0)
        y00_train.append(y0)

# k = 0
# for i in index:
index = np.array(index)
N_p = 200
N_p_test = 100
idx_p = np.random.choice(index.shape[0]-1, N_p, replace=False)
idx_test = np.random.choice(int(np.setdiff1d(index.shape[0]-1, idx_p)), N_p_test, replace=False)
idx_p1 = [i+1 for i in idx_p]
index_N = index[idx_p]
delta_index = index[idx_p1] - index_N
for i in range(int(len(index_N))):
    x0_train.append(x00_train[index_N[i]:index_N[i]+delta_index[i]])
    y0_train.append(y00_train[index_N[i]:index_N[i]+delta_index[i]])
    x_net.append(x_train[index_N[i]:index_N[i]+delta_index[i]])
    y_net.append(y_train[index_N[i]:index_N[i]+delta_index[i]])
    t_net.append(t_train[index_N[i]:index_N[i]+delta_index[i]])
    u_net.append(u_train[index_N[i]:index_N[i]+delta_index[i]])
    v_net.append(v_train[index_N[i]:index_N[i]+delta_index[i]])
    p_net.append(p_train[index_N[i]:index_N[i]+delta_index[i]])

# x_in = x_train
# y_in = y_train
# t_in = t_train
# u_in = u_train
# v_in = v_train
# p_in = p_train
# x_net.append(x_in)
# y_net.append(y_in)
# t_net.append(t_in)
# u_net.append(u_in)
# v_net.append(v_in)
# p_net.append(p_in)
x0_train = list(flatten(x0_train))
y0_train = list(flatten(y0_train))
x_net = list(flatten(x_net))
y_net = list(flatten(y_net))
u_net = list(flatten(u_net))
v_net = list(flatten(v_net))
p_net = list(flatten(p_net))
t_net = list(flatten(t_net))

x0_train = np.array(x0_train).flatten()[:, None]
y0_train = np.array(y0_train).flatten()[:, None]
x_net = np.array(x_net).flatten()[:, None]
y_net = np.array(y_net).flatten()[:, None]
t_net = np.array(t_net).flatten()[:, None]
u_net = np.array(u_net).flatten()[:, None]
v_net = np.array(v_net).flatten()[:, None]
p_net = np.array(p_net).flatten()[:, None]

# fig = plt.figure(figsize=(9, 6), dpi=300)
# ax = fig.add_subplot(111)
#
# plt.scatter(x_net,y_net,s =0.1, marker="o")
# # plt.scatter(xyt0_train[:,3:4],xyt0_train[:,6:7],s =0.1, marker="o",c='r')
# ax.set_xlabel('$x$', size=20)
# ax.set_ylabel('$y$', size=20)
# ax.tick_params(labelsize=15)
# plt.axis('equal')
# plt.show()


# cp_filename = "Cpnewelbow.csv"
#
# cp_Matrix = pd.read_csv(cp_filename, header=3)
# cp = np.array(cp_Matrix)
#
# #print('从fluent导出的训练数据为：')
# # train_data = np.delete(train_data, 2, axis=1)
# # train_data = np.delete(train_data, 4, axis=1)
# # train_data = np.delete(train_data, 0, axis=0)
# cp = cp.astype(float)
# np.set_printoptions(suppress=True)
# #print(train_data_test)
# # index = np.argwhere(train_data[:, 3:4] == 0.500999987 or train_data[:, 3:4] == 0.550999999 or train_data[:, 3:4] == 0.550999999)[:, 0:1].flatten()
# # N_p = 500
# # idx_p = np.random.choice(index.shape[0], N_p, replace=False)
# # index = index[idx_p]
#
#
# x_f_train = cp[:, 0:1]
# y_f_train = cp[:, 1:2]
# for i in range(4):
#     x_f_train = np.vstack([x_f_train,x_f_train])
#     y_f_train = np.vstack([y_f_train, y_f_train])
#
# x_f_train = x_f_train / L
# y_f_train = y_f_train / L



N_p = len(x_net)

# multiple_in = 1
# multiple_out = 2
multiple = 2
N_f = multiple * N_p
# N_f_in = multiple_in * N_p
# x_in_min = x_train.min(0)
# x_in_max = x_train.max(0)
# y_in_min = y_train.min(0)
# y_in_max = 0.15 / L
x_min = x_net.min(0)
x_max = x_net.max(0)
y_min = y_net.min(0)
y_max = y_net.max(0)
t_min = t_net.min(0)
t_max = t_net.max(0)

# N_f_out = multiple_out * N_p
# x_out_min = 0.5/L
# x_out_max = x_train.max(0)
# y_out_min = 0.15/L
# y_out_max = y_train.max(0)
# #print(x_min,x_max,y_min,y_max)
# x_f_train_in = x_in_min + (x_in_max - x_in_min) * lhs(1, N_f_in)
# y_f_train_in = y_in_min + (y_in_max - y_in_min) * lhs(1, N_f_in)
# t_f_train_in = t_min + (t_max - t_min) * lhs(1, N_f_in)
# x_f_train_out = x_out_min + (x_out_max - x_out_min) * lhs(1, N_f_out)
# y_f_train_out = y_out_min + (y_out_max - y_out_min) * lhs(1, N_f_out)
# idx_cp = np.random.choice(x_f_train.shape[0], N_f, replace=False)
x_f_train = x_min + (x_max - x_min) * lhs(1, N_f)
y_f_train = y_min + (y_max - y_min) * lhs(1, N_f)
t_f_train = t_min + (t_max - t_min) * lhs(1, N_f)



# extrater_cylinder = []
# for i in range(len(x_f_train)):
#     if x_f_train[i]>0.005/L and x_f_train[i]<0.015/L and y_f_train[i]>0.005/L and y_f_train[i]<0.015/L:
#         extrater_cylinder.append(i)
#
# x_f_train = np.delete(x_f_train, extrater_cylinder, axis=0)
# y_f_train = np.delete(y_f_train, extrater_cylinder, axis=0)
# t_f_train = np.delete(t_f_train, extrater_cylinder, axis=0)
#
# N_f_extra = int(len(extrater_cylinder))
# x_f_train_extra = 0.015/L + (x_max - 0.015/L)*lhs(1, N_f_extra)
# y_f_train_extra = y_min + (y_max - y_min)*lhs(1, N_f_extra)
# t_f_train_extra = t_min + (t_max - t_min)*lhs(1, N_f_extra)
#
# x_f_train = np.vstack((x_f_train,x_f_train_extra))
# y_f_train = np.vstack((y_f_train,y_f_train_extra))
# t_f_train = np.vstack((t_f_train,t_f_train_extra))

x0_f_train = np.array(x_f_train).flatten().tolist()
y0_f_train = np.array(y_f_train).flatten().tolist()
t_f_train = np.array(t_f_train).flatten().tolist()



x0_f_train = np.array(x_f_train).flatten()[:, None]
y0_f_train = np.array(y_f_train).flatten()[:, None]
t_f_train = np.array(t_f_train).flatten()[:, None]

# t_f_train = []
# t0_f_train = t_net.flatten().tolist()
#
# for i in range(2000):
#     t_f_train.append(t0_f_train[:delta_i])
# t_f_train = np.array(t_f_train).flatten()[:, None]

len_data = int(x_net.shape[0])
x0_f_train_batch = x0_f_train[:len_data,:]
y0_f_train_batch = y0_f_train[:len_data,:]
t0_f_train_batch = t_f_train[:len_data,:]

for i in range(multiple - 1):
    j = i + 1
    x0_f_train_batch = np.hstack((x0_f_train_batch, x0_f_train[j * len_data:(j + 1) * len_data, :]))
    y0_f_train_batch = np.hstack((y0_f_train_batch, y0_f_train[j * len_data:(j + 1) * len_data, :]))
    t0_f_train_batch = np.hstack((t0_f_train_batch, t_f_train[j * len_data:(j + 1) * len_data, :]))


xyt0_train = np.hstack([x_net, y_net, t_net, x0_f_train_batch, y0_f_train_batch, t0_f_train_batch])
xyuv_train = np.hstack([u_net, v_net, p_net])
#xyt0_star = np.hstack([x0_star, y0_star, t_star])
xyuv_test = np.hstack([xyt0_train,xyuv_train])


#jiangcaiyang
# M = []

# for i in range(int(len(x_net))):
#     if i % 100 != 0:
#         M.append(i)
#
# xyt0_train = np.delete(xyt0_train, M, axis=0)
# xyuv_train = np.delete(xyuv_train, M, axis=0)

N_u = 10000
idx = np.random.choice(xyt0_train.shape[0], N_u, replace=False)
xyt0_train = xyt0_train[idx, :]
xyuv_train = xyuv_train[idx, :]
# xyt0_train = torch.tensor(xyt0_train, requires_grad=True).float().to(device)
# xyuv_train = torch.tensor(xyuv_train, requires_grad=True).float().to(device)

# torch.set_default_tensor_type(torch.DoubleTensor)
xyt0_train = torch.tensor(xyt0_train).float()
xyuv_train = torch.tensor(xyuv_train).float()
B_gauss_1 = torch.tensor(B_gauss_1).float()
B_gauss_2 = torch.tensor(B_gauss_2).float()
train_data = addbatch(xyt0_train,xyuv_train, 50000)
model = TrajectoryNSNet(B_gauss_2)




##############测试集################
# filename_test = "F:/code/pycharm/reproductionPINN/export0905.csv"
#
# #train_data = np.genfromtxt(filename, delimiter=',', skip_header=True)
# x_Matrix_test = pd.read_csv(filename_test, header=3)
# train_data_test = np.array(x_Matrix_test)
#
# #print('从fluent导出的训练数据为：')
# train_data_test = np.delete(train_data_test, 2, axis=1)
# # train_data = np.delete(train_data, 0, axis=0)
# train_data_test = train_data_test.astype(float)
# np.set_printoptions(suppress=True)
# #print(train_data_test)
# index_test = np.argwhere(train_data_test[:, 3:4] == 10.0200005)[:, 0:1].flatten()
#
# x_test = train_data_test[:, 0:1]
# y_test = train_data_test[:, 1:2]
# t_test = train_data_test[:, 3:4]
# p_test = train_data_test[:, 4:5]
# u_test = train_data_test[:, 5:6]
# v_test = train_data_test[:, 6:7]
#
# x_test = [i+0.5 for i in x_test]
# y_test = [i+0.5 for i in y_test]
#
# x_test = np.array(x_test).flatten()[:, None]
# y_test = np.array(y_test).flatten()[:, None]
# p_test = np.array(p_test).flatten()[:, None]
# u_test = np.array(u_test).flatten()[:, None]
# v_test = np.array(v_test).flatten()[:, None]
# t_test = np.array(t_test).flatten()[:, None]
#
# x_test = x_test / L
# y_test = y_test / L
# t_test = t_test * (U / L)
# u_test = u_test / U
# v_test = v_test / U
# p_test = p_test / (1 * (U ** 2))
#
#
# #print(x_test, x_test.shape)
# # 初始时刻，x y值
# # 初始时刻，x y值
# #index = np.append(index, 685800)
# k = 0
# x0_test = []
# y0_test = []
# x_net_test =[]
# y_net_test =[]
# t_net_test =[]
# u_net_test =[]
# v_net_test =[]
# p_net_test =[]
# par_num = []
# delta_i = 1000
# for i in index_test:
#     x0 = x_test[i]
#     y0 = y_test[i]
#
#     k = k+1
#     for j in range(delta_i):
#         x0_test.append(x0)
#         y0_test.append(y0)
#         j = j + 1
# k = 0
# for i in index_test:
#     x_in_test = x_test[i:i + delta_i]
#     y_in_test = y_test[i:i + delta_i]
#     t_in_test = t_test[i:i + delta_i]
#     u_in_test = u_test[i:i + delta_i]
#     v_in_test = v_test[i:i + delta_i]
#     p_in_test = p_test[i:i + delta_i]
#     x_net_test.append(x_in_test)
#     y_net_test.append(y_in_test)
#     t_net_test.append(t_in_test)
#     u_net_test.append(u_in_test)
#     v_net_test.append(v_in_test)
#     p_net_test.append(p_in_test)
#
# x0_test = np.array(x0_test).flatten()[:, None]
# y0_test = np.array(y0_test).flatten()[:, None]
# x_net_test = np.array(x_net_test).flatten()[:, None]
# y_net_test = np.array(y_net_test).flatten()[:, None]
# t_net_test = np.array(t_net_test).flatten()[:, None]
# u_net_test = np.array(u_net_test).flatten()[:, None]
# v_net_test = np.array(v_net_test).flatten()[:, None]
# p_net_test = np.array(p_net_test).flatten()[:, None]
#
#
#
# M_test = []
#
# for i in range(67000):
#     if i % 5 != 0:
#         M_test.append(i)
#
# x0_test = np.delete(x0_test, M_test, axis=0)
# y0_test = np.delete(y0_test, M_test, axis=0)
# x_net_test = np.delete(x_net_test, M_test, axis=0)
# y_net_test = np.delete(y_net_test, M_test, axis=0)
# t_net_test = np.delete(t_net_test, M_test, axis=0)
# u_net_test = np.delete(u_net_test, M_test, axis=0)
# v_net_test = np.delete(v_net_test, M_test, axis=0)
# p_net_test = np.delete(p_net_test, M_test, axis=0)
# t_net = t_net[:67000,]
# t_net = np.delete(t_net, M_test, axis=0)
# u_test_pred, v_test_pred, p_test_pred = model.predict_uv(x_net_test, y_net_test, t_net)
# # error_x_test = np.linalg.norm(x_net_test - x_test_pred, 2) / np.linalg.norm(x_net_test, 2)
# # error_y_test = np.linalg.norm(y_net_test - y_test_pred, 2) / np.linalg.norm(y_net_test, 2)
#
# # u_net_test = u_net_test * U
# # u_test_pred = u_test_pred * U
# # v_net_test = v_net_test * U
# # v_net_pred = v_test_pred * U
# # p_net_test = p_net_test * (1 * (U ** 2))
# # p_test_pred = p_test_pred * (1 * (U ** 2))
#
#
# error_u_test = np.linalg.norm(u_net_test - u_test_pred, 2) / np.linalg.norm(u_net_test, 2)
# error_v_test = np.linalg.norm(v_net_test - v_test_pred, 2) / np.linalg.norm(v_net_test, 2)
# error_p_test = np.linalg.norm(p_net_test - p_test_pred, 2) / np.linalg.norm(p_net_test, 2)
#
# # print('Error x test: %e' % (error_x_test))
# # print('Error y test: %e' % (error_y_test))
# print('Error u test: %e' % (error_u_test))
# print('Error v test: %e' % (error_v_test))
# print('Error p test: %e' % (error_p_test))


########################################################


filename_test = "fangqiang1250.csv"

#train_data = np.genfromtxt(filename, delimiter=',', skip_header=True)
wangge = pd.read_csv(filename_test, header=3)
wangge = np.array(wangge)

#print('从fluent导出的训练数据为：')
wangge = np.delete(wangge, 2, axis=1)
# train_data = np.delete(train_data, 0, axis=0)
wangge = wangge.astype(float)
np.set_printoptions(suppress=True)


# wangge = np.around(wangge,3)
wangge_u = np.array(wangge[:,3:4]).flatten()[:, None]
wangge_x = np.array(wangge[:,0:1]).flatten()[:, None]
wangge_y = np.array(wangge[:,1:2]).flatten()[:, None]
wangge_p = np.array(wangge[:,2:3]).flatten()[:, None]
wangge_v = np.array(wangge[:,4:5]).flatten()[:, None]

# wangge_x = np.around(wangge_x,2)
# wangge_y = np.around(wangge_y,2)



wangge_x = np.array(wangge_x).flatten()[:, None]
wangge_y = np.array(wangge_y).flatten()[:, None]

wangge_x = wangge_x / L
wangge_y = wangge_y / L
# t_train = t * (U / L)
wangge_u  = wangge_u  / U
wangge_v = wangge_v / U
wangge_p = wangge_p / (1 * (U ** 2))

# X_true_star = np.hstack((wangge_x, wangge_y))
#
# wangge_y_in = []
# for i in range(101):
#     yy = wangge_y[i * 101]
#     wangge_y_in.append(yy)
#
# wangge_y = np.array(wangge_y_in).flatten()[:, None]
#
# X, Y = np.meshgrid(wangge_x[0:101], wangge_y)
#
# X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

# print(X_star)
t_plot = []
for _ in range(int(len(wangge_x))):
    t_plot.append(3.0 * (U / L))
t_plot = np.array(t_plot).flatten()[:, None]
#####################################

#####################################
u_test_pred, v_test_pred, p_test_pred = model.predict_uv(wangge_x, wangge_y, t_plot)

xiuzheng = wangge_p[100] - p_test_pred[100]
p_test_pred = p_test_pred + xiuzheng

wangge_u = wangge_u * U
u_test_pred = u_test_pred * U
wangge_v = wangge_v * U
v_test_pred = v_test_pred * U
wangge_p = wangge_p * (1 * (U ** 2))
p_test_pred = p_test_pred * (1 * (U ** 2))

error_u_test = np.linalg.norm(wangge_u - u_test_pred, 2) / np.linalg.norm(wangge_u, 2)
error_v_test = np.linalg.norm(wangge_v - v_test_pred, 2) / np.linalg.norm(wangge_v, 2)
error_p_test = np.linalg.norm(wangge_p - p_test_pred, 2) / np.linalg.norm(wangge_p, 2)
mask = (t_net > 0.290) & (t_net < 0.292)
filtered_data_x = x_net[mask]
filtered_data_y = y_net[mask]
# print('Error x test: %e' % (error_x_test))
# print('Error y test: %e' % (error_y_test))
wangge_u_pearsonr = wangge_u.flatten()
wangge_v_pearsonr = wangge_v.flatten()
wangge_p_pearsonr = wangge_p.flatten()
u_test_pred_pearsonr = u_test_pred.flatten()
v_test_pred_pearsonr = v_test_pred.flatten()
p_test_pred_pearsonr = p_test_pred.flatten()

coefficient_u_test ,_= stats.pearsonr(wangge_u_pearsonr,u_test_pred_pearsonr)
coefficient_v_test,_ = stats.pearsonr(wangge_v_pearsonr,v_test_pred_pearsonr)
coefficient_p_test,_ = stats.pearsonr(wangge_p_pearsonr,p_test_pred_pearsonr)
print('Error u test: %e' % (error_u_test))
print('Error v test: %e' % (error_v_test))
print('Error p test: %e' % (error_p_test))
print('Coefficient u test: %f' % (coefficient_u_test))
print('Coefficient v test: %f' % (coefficient_v_test))
print('Coefficient p test: %f' % (coefficient_p_test))

u_pred_mag = (u_test_pred ** 2 + v_test_pred ** 2) ** 0.5
wangge_u_mag = (wangge_u ** 2 + wangge_v ** 2) ** 0.5
fig = plt.figure(dpi=600,figsize=(9,4))
ax1= fig.add_subplot(121,aspect='equal')
ax2= fig.add_subplot(122,aspect='equal')
U_pred = ax2.scatter(wangge_x,wangge_y, c=u_pred_mag, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none',vmin=-1,vmax=1, rasterized=True)
U_true = ax1.scatter(wangge_x,wangge_y, c=wangge_u_mag, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none',vmin=-1,vmax=1, rasterized=True)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
ax1.set_title('u true t=400', fontsize = 10)
ax2.set_title('u predict t=400', fontsize = 10)

fig.subplots_adjust(hspace=10.0)
fig.colorbar(U_true, ax=[ax1,ax2], fraction=0.03, shrink=0.5)
fig.savefig('1_u_mag_50_25s'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(dpi=600,figsize=(9,4))
ax1= fig.add_subplot(121,aspect='equal')
ax2= fig.add_subplot(122,aspect='equal')
V_pred = ax2.scatter(wangge_x,wangge_y, c=v_test_pred, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none',vmin=-0.3,vmax=0.3, rasterized=True)
V_true = ax1.scatter(wangge_x,wangge_y, c=wangge_v, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none',vmin=-0.3,vmax=0.3, rasterized=True)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
ax1.set_title('v true t=400', fontsize = 10)
ax2.set_title('v predict t=400', fontsize = 10)

fig.subplots_adjust(hspace=10.0)
fig.colorbar(V_true, ax=[ax1,ax2], fraction=0.03, shrink=0.5)
fig.savefig('1_v_50_25s'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


fig = plt.figure(dpi=600,figsize=(9,4))
ax1= fig.add_subplot(121,aspect='equal')
ax2= fig.add_subplot(122,aspect='equal')
P_pred = ax2.scatter(wangge_x,wangge_y, c=p_test_pred, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none',vmin = -3, vmax = 3, rasterized=True)
P_true = ax1.scatter(wangge_x,wangge_y, c=wangge_p, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none',vmin = -3, vmax = 3, rasterized=True)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
ax1.set_title('p true t=400', fontsize = 10)
ax2.set_title('p predict t=400', fontsize = 10)

fig.subplots_adjust(hspace=10.0)
fig.colorbar(P_true, ax=[ax1,ax2], fraction=0.03, shrink=0.5)
fig.savefig('1_p_50_25s'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()
# fig = plt.figure(dpi=600, figsize=(4, 9))
# gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1], wspace=0.1, hspace=0.01)
#
# # 创建子图并指定它们在网格中的位置
# ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
# ax2 = fig.add_subplot(gs[1, 0], aspect='equal')
#
# # 绘制散点图
# U_true = ax1.scatter(wangge_x, wangge_y, c=wangge_u, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none')
# U_pred = ax2.scatter(wangge_x, wangge_y, c=u_test_pred, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none')
#
# # 设置标题
# # ax1.set_title('u true t=400', fontsize=10)
# # ax2.set_title('u predict t=400', fontsize=10)
#
# # 移除刻度线和坐标轴边框
# for ax in [ax1, ax2]:
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#
# # 添加跨越两行的颜色条
# cax = fig.add_subplot(gs[:, 1])  # 颜色条跨越所有行，占据第二列
# fig.colorbar(U_true, cax=cax)  # 可以选择 U_pred 或 U_true 作为颜色条的参考
# fig.savefig('1_u_50_25s'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()
#
# fig = plt.figure(dpi=600, figsize=(4, 9))
# gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1], wspace=0.1, hspace=0.01)
#
# # 创建子图并指定它们在网格中的位置
# ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
# ax2 = fig.add_subplot(gs[1, 0], aspect='equal')
#
# # 绘制散点图
# V_true = ax1.scatter(wangge_x, wangge_y, c=wangge_v, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none')
# V_pred = ax2.scatter(wangge_x, wangge_y, c=v_test_pred, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none')
#
# # 设置标题
# # ax1.set_title('u true t=400', fontsize=10)
# # ax2.set_title('u predict t=400', fontsize=10)
#
# # 移除刻度线和坐标轴边框
# for ax in [ax1, ax2]:
#     ax.set_xticks([])
#     ax.set_yticks([])
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#
# # 添加跨越两行的颜色条
# cax = fig.add_subplot(gs[:, 1])  # 颜色条跨越所有行，占据第二列
# fig.colorbar(V_true, cax=cax)  # 可以选择 U_pred 或 U_true 作为颜色条的参考
# fig.savefig('1_v_50_25s'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()


fig = plt.figure(dpi=600,figsize=(9,4))
ax1= fig.add_subplot(121,aspect='equal')
ax2= fig.add_subplot(122,aspect='equal')
P_pred = ax2.scatter(wangge_x,wangge_y, c=p_test_pred, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none', rasterized=True)
P_true = ax1.scatter(wangge_x,wangge_y, c=wangge_p, cmap='rainbow', s=7, marker='s', alpha=0.9, edgecolors='none', rasterized=True)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
ax1.set_title('p true t=400', fontsize = 10)
ax2.set_title('p predict t=400', fontsize = 10)

fig.subplots_adjust(hspace=10.0)
fig.colorbar(P_true, ax=[ax1,ax2], fraction=0.03, shrink=0.5)

plt.show()


######################################################
fig, ax = plt.subplots()
error_u = ax.scatter(wangge_x,wangge_y, c=u_pred_mag-wangge_u_mag, cmap='seismic', s=20, marker='o', alpha=0.9, edgecolors='none', vmin=-1, vmax=1, rasterized=True)
# ax.scatter(filtered_data_x, filtered_data_y, s=10, marker="x", c='gray', label='Particle position',zorder=10, alpha=0.6, rasterized=True)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(error_u, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
#                    borderaxespad=0.)
# ax.set_title('error_u t=400', fontsize = 10)
ax.set_aspect('equal')
fig.savefig('1_error_u_mag_50_25s'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig, ax = plt.subplots()
error_v = ax.scatter(wangge_x,wangge_y, c=v_test_pred-wangge_v, cmap='seismic', s=20, marker='o', alpha=0.9, edgecolors='none', vmin=-wangge_v.max(), vmax=wangge_v.max(), rasterized=True)
ax.scatter(filtered_data_x, filtered_data_y, s=10, marker="x", c='gray', label='Particle position',zorder=10, alpha=0.6, rasterized=True)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(error_v, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0),
                   borderaxespad=0.)
# ax.set_title('error_v t=400', fontsize = 10)
ax.set_aspect('equal')
fig.savefig('1_error_v_50_25s'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig, ax = plt.subplots()
error_p = ax.scatter(wangge_x,wangge_y, c=p_test_pred-wangge_p, cmap='seismic', s=20, marker='o', alpha=0.9, edgecolors='none', vmin=-3, vmax=3, rasterized=True)
# ax.scatter(filtered_data_x, filtered_data_y, s=10, marker="x", c='gray', label='Particle position',zorder=10, alpha=0.6, rasterized=True)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.10)
cbar = fig.colorbar(error_p, cax=cax)
cbar.ax.tick_params(labelsize=15)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
# ax.set_title('error_p t=400', fontsize = 10)
ax.set_aspect('equal')
fig.savefig('1_error_p_50_25s'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()
##########################################################################

from scipy.stats import gaussian_kde
from scipy import stats
def calculate_kde(data, n_points=1000):
    """计算数据的KDE曲线坐标"""
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), n_points)
    y = kde.evaluate(x)
    return x, y

u_mag_true = (wangge_u ** 2 + wangge_v ** 2 ) ** 0.5
u_mag_pred = (u_test_pred ** 2 + v_test_pred ** 2 ) ** 0.5
errors = u_mag_pred - u_mag_true
NAE = np.abs(u_mag_pred - u_mag_true) / u_mag_plot_cor
NAE = NAE.flatten()
u_mag_pred = u_mag_pred.flatten()
u_mag_true = u_mag_true.flatten()
errors = errors.flatten()
# 计算各数据集KDE
true_x, true_y = calculate_kde(u_mag_true)
pred_x, pred_y = calculate_kde(u_mag_pred)
error_x, error_y = calculate_kde(NAE)

df_true = pd.DataFrame({'x': true_x, 'density': true_y})
df_pred = pd.DataFrame({'x': pred_x, 'density': pred_y})
df_error = pd.DataFrame({'x': error_x, 'density': error_y})

df_true.to_csv('1true_velocity_kde.csv', index=False)
df_pred.to_csv('1predicted_velocity_kde.csv', index=False)
df_error.to_csv('1error_kde.csv', index=False)

True_mean=np.mean(u_mag_true)
True_std=np.std(u_mag_true)
True_skew=stats.skew(u_mag_true)
True_kurtosis=stats.kurtosis(u_mag_true)

Predicted_mean=np.mean(u_mag_pred)
Predicted_std=np.std(u_mag_pred)
Predicted_skew=stats.skew(u_mag_pred)
Predicted_kurtosis=stats.kurtosis(u_mag_pred)

RMSE=np.sqrt(np.mean(errors ** 2))
MAE=np.mean(np.abs(errors))
KS_stat=stats.ks_2samp(u_mag_true, u_mag_pred)[0]
KS_pvalue=stats.ks_2samp(u_mag_true, u_mag_pred)[1]

print('Delta mu = %.5f'% (np.mean(u_mag_pred) - np.mean(u_mag_true)))
print('Delta sigma = %.5f'% (np.std(u_mag_pred) - np.std(u_mag_true)))
print('KS p=%.5e'% stats.ks_2samp(u_mag_true, u_mag_pred)[1])

print('RMSE = %.5f'% RMSE)
print('MAE = %.5f'% MAE)
print('Skewness p=%.5e'% stats.skew(errors))
print('Skewness p=%.5e'% stats.kurtosis(errors))