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
import matplotlib.gridspec as gridspec
from itertools import product, combinations
import pandas as pd
from more_itertools import flatten
from torch.utils.data import Dataset, DataLoader, TensorDataset


os.environ['CUDA_VISIBLE_DEVICES']='3'
np.random.seed(1234)
torch.cuda.manual_seed_all(1234)
torch.manual_seed(1234)
device = torch.device('cuda')

# 自定义第二个神经网络
class Net1(nn.Module):
    def __init__(self, layers_1):
        super(Net1, self).__init__()

        self.depth = len(layers_1) - 1  # 深度是10-1为9
        self.activation = nn.Tanh

        layers_list = list()

        for i in range(self.depth - 1):
            layers_list.append(
                ('layer_%d' % i, nn.Linear(layers_1[i], layers_1[i + 1]))  # 把每一层都加到list中
            )
            layers_list.append(
                ('activation_%d' % i, self.activation())  # 每一层都加激活函数Tanh
            )

        layers_list.append(
            ('layer_%d' % (self.depth - 1), nn.Linear(layers_1[-2], layers_1[-1]))  # 最后线性输出
        )
        layerDict = OrderedDict(layers_list)  # 用有序字典

        self.layers = nn.Sequential(layerDict)  # 用Sequential容器做一个网络

    def forward(self, x0, y0, t1, t0, B):
        dt = t1 - t0
        xydtt0 = torch.cat([x0, y0, dt, t0], dim=1)
        FF_xydtt0 = input_mapping(xydtt0, B)
        delta_xy = self.layers(FF_xydtt0)
        xy = torch.add(delta_xy, torch.cat([x0, y0], dim=1))
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        return x, y, dt


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

            layers_list.append(
                ('activation_%d' % i, self.activation())  # 每一层都加激活函数Tanh
            )

        layers_list.append(
            ('layer_%d' % (self.depth - 1), nn.Linear(layers_2[-2], layers_2[-1]))  # 最后线性输出
        )
        layerDict = OrderedDict(layers_list)  # 用有序字典

        self.layers = nn.Sequential(layerDict)  # 用Sequential容器做一个网络

    def forward(self, x, y, t1, B1):
        xyt = torch.cat([x, y, t1], dim=1)
        FF_xyt = input_mapping(xyt, B1)
        u_v_p = self.layers(FF_xyt)
        return u_v_p


# 将两个神经网络串联起来
class Nets(nn.Module):
    def __init__(self, net1, net2):
        super(Nets, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x0, y0, t1, t0, t2, B2):
        x, y, dt = self.net1(x0, y0, t1, t0)
        psi_p = self.net2(x, y, t2, B2)
        return x, y, dt, psi_p


class TrajectoryNSNet():
    def __init__(self, B1, B2):
        self.layers_1 = layers_1
        self.layers_2 = layers_2
        self.B1 = B1.to(device)
        self.B2 = B2.to(device)
        self.net_1 = Net1(layers_1).to(device)
        self.net_2 = Net2(layers_2).to(device)

        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_1 = nn.Parameter(self.lambda_1)
        self.net_2 = Net2(layers_2).to(device)
        self.net_2.register_parameter('lambda_1', self.lambda_1)  # 将一个参数添加到模块中
        # self.net_1.load_state_dict(torch.load("suanli3net1pretrain_wang_10x.pth"))
        # self.net_2.load_state_dict(torch.load("suanli3net2pretrain_test_auto_manual_more0.pth"))
        # self.lambda_1 = self.lambda_1.requires_grad_(False)
        self.nets = Nets(self.net_1, self.net_2).to(device)

        self.optimizer = torch.optim.LBFGS(
            self.nets.parameters(),
            lr=1.0,
            max_iter=500000,
            max_eval=500000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(self.nets.parameters(), lr=1e-3)
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_Adam, gamma=0.99999996)
        self.iter = 0


    def net_x(self, x0, y0, t1, t0, B1):
        x, y, dt = self.net_1(x0, y0, t1, t0, B1)

        return x, y, dt

    def net_x_psi_p(self, x0, y0, t1, t0, t2, B1, B2):
        x, y, _ = self.net_x(x0, y0, t1, t0, B1)
        psi_p = self.net_2(x, y, t2, B2)
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

        return u, v, p

    def net_E(self, x0, y0, t1, t0, t2, B1, B2):
        x, y, dt = self.net_x(x0, y0, t1, t0, B1)
        u, v, _ = self.net_x_psi_p(x0, y0, t1, t0, t2, B1, B2)
        x_t = torch.autograd.grad(
            x, dt,
            grad_outputs=torch.ones_like(x),
            retain_graph=True,  # 由于后面要求二阶导所以此处应设置为true
            create_graph=True
        )[0]
        y_t = torch.autograd.grad(
            y, dt,
            grad_outputs=torch.ones_like(y),
            retain_graph=True,  # 由于后面要求二阶导所以此处应设置为true
            create_graph=True
        )[0]

        E_x = x_t - u
        E_y = y_t - v

        return E_x, E_y

    def net_f(self, x0, y0, t1, t0, t2, B1, B2):
        lambda_1 = self.lambda_1
        x, y, _ = self.net_x(x0, y0, t1, t0, B1)
        psi_p = self.net_2(x, y, t2, B2)

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

        u_t = torch.autograd.grad(
            u, t2,
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
            v, t2,
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

        self.RE1 = 0.0008  * torch.sigmoid(lambda_1)

        f_u = u_t + u * u_x + v * u_y + p_x - self.RE1 * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y + p_y - self.RE1 * (v_xx + v_yy)


        return f_u, f_v

    def closure(self):
        self.optimizer.zero_grad()
        self.x_pred, self.y_pred, _ = self.net_x(self.x0, self.y0, self.t1, self.t0, self.B1)
        self.u_pred, self.v_pred, self.p_pred = self.net_x_psi_p(self.x0, self.y0, self.t1, self.t0, self.t2,self.B1, self.B2)
        self.E_x_pred, self.E_y_pred = self.net_E(self.x0_f_train, self.y0_f_train, self.t1_f_train, self.t0_f_train,
                                                  self.t2_f_train, self.B1, self.B2)
        self.f_u_pred, self.f_v_pred = self.net_f(self.x0_f_train, self.y0_f_train, self.t1_f_train, self.t0_f_train,
                                                  self.t2_f_train, self.B1, self.B2)
        loss_x = torch.mean((self.x - self.x_pred) ** 2) + \
                 torch.mean((self.y - self.y_pred) ** 2)
        loss_u = torch.mean((self.u - self.u_pred) ** 2) + \
                 torch.mean((self.v - self.v_pred) ** 2)
        loss_e = torch.mean((self.E_x_pred) ** 2) + \
                 torch.mean((self.E_y_pred) ** 2)
        loss_f = torch.mean((self.f_u_pred) ** 2) + \
                 torch.mean((self.f_v_pred) ** 2)

        total_loss =  500 * loss_x +  loss_e +  loss_u +  0.1 * loss_f
        self.iter += 1
        if self.iter % 10 == 0:
            print(
                'It: %d, total_loss: %.5e, loss_x: %.5e, loss_u: %.5e, loss_e: %.5e, loss_f: %.5e, l1: %.6f' %
                (
                    self.iter,
                    total_loss.item(),
                    loss_x.item(),
                    loss_u.item(),
                    loss_e.item(),
                    loss_f.item(),
                    self.RE1
                )
            )

        total_loss.backward()
        return total_loss

    def train(self, traindata):
        self.nets.train()
        for step, data in enumerate(traindata):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            self.optimizer_Adam.zero_grad()
            self.x0 = inputs[:, 0:1].clone().detach().requires_grad_(True)
            self.y0 = inputs[:, 1:2].clone().detach().requires_grad_(True)
            self.t1 = inputs[:, 2:3].clone().detach().requires_grad_(True)
            self.t0 = inputs[:, 3:4].clone().detach().requires_grad_(True)
            self.t2 = self.t1.clone().detach().requires_grad_(True)
            self.x0_f_train = inputs[:, 4: 4 + multiple]
            self.y0_f_train = inputs[:, 4 + multiple:4 + 2 * multiple]
            self.t0_f_train = inputs[:, 4 + 2 * multiple:4 + 3 * multiple]
            self.t_f_train = inputs[:, 4 + 3 * multiple:4 + 4 * multiple]
            self.x0_f_train = torch.cat([self.x0, self.x0_f_train], dim=1)
            self.y0_f_train = torch.cat([self.y0, self.y0_f_train], dim=1)
            self.t1_f_train = torch.cat([self.t1, self.t_f_train], dim=1)
            self.t0_f_train = torch.cat([self.t0, self.t0_f_train], dim=1)
            self.x0_f_train = torch.flatten(self.x0_f_train.T, start_dim=0, end_dim=1)[:,
                              None].clone().detach().requires_grad_(True)
            self.y0_f_train = torch.flatten(self.y0_f_train.T, start_dim=0, end_dim=1)[:,
                              None].clone().detach().requires_grad_(True)
            self.t1_f_train = torch.flatten(self.t1_f_train.T, start_dim=0, end_dim=1)[:,
                              None].clone().detach().requires_grad_(True)
            self.t0_f_train = torch.flatten(self.t0_f_train.T, start_dim=0, end_dim=1)[:,
                              None].clone().detach().requires_grad_(True)
            self.t2_f_train = self.t1_f_train.clone().detach().requires_grad_(True)
            self.x = labels[:, 0:1].clone().detach().requires_grad_(True)
            self.y = labels[:, 1:2].clone().detach().requires_grad_(True)
            self.u = labels[:, 2:3].clone().detach().requires_grad_(True)
            self.v = labels[:, 3:4].clone().detach().requires_grad_(True)
            self.p = labels[:, 4:5].clone().detach().requires_grad_(True)

            self.optimizer.step(closure=self.closure)
        print('End')
        torch.save(self.net_1.state_dict(), "suanli3nopretrainnet1.pth")
        torch.save(self.net_2.state_dict(), "suanli3nopretrainnet2.pth")
        # torch.save(self.nets.state_dict(), "elbowwithcylinderallfinal.pth")

def addbatch(data_train,data_test,batchsize):

    data = TensorDataset(data_train,data_test)
    data_loader = DataLoader(data, batch_size=batchsize, shuffle=True)#shuffle是是否打乱数据集，可自行设置

    return data_loader

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



np.set_printoptions(precision=20)



filename = "Velocity_high.csv"

x_Matrix = pd.read_csv(filename, header=0)
train_data = np.array(x_Matrix)

#print('从fluent导出的训练数据为：')
# train_data = np.delete(train_data, 2, axis=1)
# train_data = np.delete(train_data, 4, axis=1)
# train_data = np.delete(train_data, 0, axis=0)
train_data = train_data.astype(float)
np.set_printoptions(suppress=True)
#print(train_data_test)
# index = np.argwhere(train_data[:, 3:4] == 0.500999987 or train_data[:, 3:4] == 0.550999999 or train_data[:, 3:4] == 0.550999999)[:, 0:1].flatten()
# N_p = 500
# idx_p = np.random.choice(index.shape[0], N_p, replace=False)
# index = index[idx_p]

id = train_data[:, 0:1]
x = train_data[:, 4:5]
y = train_data[:, 5:6]
t = train_data[:, 1:2]
u = train_data[:, 2:3]
v = train_data[:, 3:4]

def find_diff_positions(arr):
    diff_positions = []
    diff_positions.append(0)
    for i in range(1, len(arr)):
        if arr[i] - arr[i-1] != 0:
            diff_positions.append(i)
    return diff_positions
index = find_diff_positions(id)
index.append(int(len(id)))

y_max = max(y)
y = y_max-y
v = - v

x = 0.082 * 0.001 * x
y = 0.082 * 0.001 * y
t =  0.005 * t
u = ((0.082 * 0.001)/( 0.005)) * u
v = ((0.082 * 0.001)/( 0.005)) * v

########################################
data_auto = np.load('BULK_wang.npz')

# 访问保存的数组
x_auto = data_auto['x']
y_auto = data_auto['y']
t_auto = data_auto['t']
u_auto = data_auto['u']
v_auto = data_auto['v']



x_ab = max(abs(x))
y_ab = max(abs(y))
L = float(max(x_ab, y_ab))
u_ab = max(abs(u))
v_ab = max(abs(v))
U = float(max(u_ab, v_ab))


x_ab_auto = max(abs(x_auto))
y_ab_auto = max(abs(y_auto))
L_auto = float(max(x_ab_auto, y_ab_auto))
u_ab_auto = max(abs(u_auto))
v_ab_auto = max(abs(v_auto))
U_auto = float(max(u_ab_auto, v_ab_auto))


L = float(max(L, L_auto))
U = float(max(U, U_auto))
# U = 1
# L = 0.01
x_train = x / L
y_train = y / L
t_train = t * (U / L)
u_train = u / U
v_train = v / U

k = 0
x00_train = []
y00_train = []
t00_train = []
x0_train = []
y0_train = []
t0_train = []
x_net =[]
y_net =[]
t_net =[]
u_net =[]
v_net =[]
# p_net =[]
# par_num = []
# delta_i = 1400

for i in index[:-1]:
    x0 = x_train[i]
    y0 = y_train[i]
    t0 = t_train[i]
    delta_i = index[k+1] - index[k]
    k = k+1
    for j in range(delta_i):
        x00_train.append(x0)
        y00_train.append(y0)
        t00_train.append(t0)

# for i in index:
index = np.array(index)
N_p = int(0.8 * len(index))
N_p_test = int(0.15 * len(index))
idx_p = np.random.choice(index.shape[0]-1, N_p, replace=False)
idx_test = np.random.choice(int(np.setdiff1d(index.shape[0]-1, idx_p)), N_p_test, replace=False)
idx_p1 = [i+1 for i in idx_p]
index_N = index[idx_p]
delta_index = index[idx_p1] - index_N
for i in range(int(len(index_N))):
    x0_train.append(x00_train[index_N[i]:index_N[i]+delta_index[i]])
    y0_train.append(y00_train[index_N[i]:index_N[i]+delta_index[i]])
    t0_train.append(t00_train[index_N[i]:index_N[i] + delta_index[i]])
    x_net.append(x_train[index_N[i]:index_N[i]+delta_index[i]])
    y_net.append(y_train[index_N[i]:index_N[i]+delta_index[i]])
    t_net.append(t_train[index_N[i]:index_N[i]+delta_index[i]])
    u_net.append(u_train[index_N[i]:index_N[i]+delta_index[i]])
    v_net.append(v_train[index_N[i]:index_N[i]+delta_index[i]])
    # p_net.append(p_train[index_N[i]:index_N[i]+delta_index[i]])

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
t0_train = list(flatten(t0_train))
x_net = list(flatten(x_net))
y_net = list(flatten(y_net))
u_net = list(flatten(u_net))
v_net = list(flatten(v_net))
# p_net = list(flatten(p_net))
t_net = list(flatten(t_net))

x0_train = np.array(x0_train).flatten()[:, None]
y0_train = np.array(y0_train).flatten()[:, None]
t0_train = np.array(t0_train).flatten()[:, None]
x_net = np.array(x_net).flatten()[:, None]
y_net = np.array(y_net).flatten()[:, None]
t_net = np.array(t_net).flatten()[:, None]
u_net = np.array(u_net).flatten()[:, None]
v_net = np.array(v_net).flatten()[:, None]





# U = 1
# L = 0.01
x_train_auto = x_auto / L
y_train_auto = y_auto / L
t_train_auto = t_auto * (U / L)
u_train_auto = u_auto / U
v_train_auto = v_auto / U
# p_train = p / (1 * (U ** 2))
extrater = []
for i in range(len(x_train_auto)):
    if t_train_auto[i]>0.04:
        extrater.append(i)

x_train_auto0 = np.delete(x_train_auto, extrater, axis=0)
y_train_auto0 = np.delete(y_train_auto, extrater, axis=0)
t_train_auto0 = np.delete(t_train_auto, extrater, axis=0)
u_train_auto0 = np.delete(u_train_auto, extrater, axis=0)
v_train_auto0 = np.delete(v_train_auto, extrater, axis=0)

extrater = []
for i in range(len(x_train_auto)):
    if t_train_auto[i]<0.04:
        extrater.append(i)

x_train_auto = np.delete(x_train_auto, extrater, axis=0)
y_train_auto = np.delete(y_train_auto, extrater, axis=0)
t_train_auto = np.delete(t_train_auto, extrater, axis=0)
u_train_auto = np.delete(u_train_auto, extrater, axis=0)
v_train_auto = np.delete(v_train_auto, extrater, axis=0)

idx_auto0 = np.random.choice(x_train_auto0.shape[0], int(0.35 * len(x_net)), replace=False)
x0_net_auto0 = x_train_auto0[idx_auto0]
y0_net_auto0 = y_train_auto0[idx_auto0]
t0_net_auto0 = t_train_auto0[idx_auto0]
x_net_auto0 = x_train_auto0[idx_auto0]
y_net_auto0 = y_train_auto0[idx_auto0]
t_net_auto0 = t_train_auto0[idx_auto0]
u_net_auto0 = u_train_auto0[idx_auto0]
v_net_auto0 = v_train_auto0[idx_auto0]



idx_auto = np.random.choice(x_train_auto.shape[0], int(1 * len(x_net)), replace=False)
x0_net_auto = x_train_auto[idx_auto]
y0_net_auto = y_train_auto[idx_auto]
t0_net_auto = t_train_auto[idx_auto]
x_net_auto = x_train_auto[idx_auto]
y_net_auto = y_train_auto[idx_auto]
t_net_auto = t_train_auto[idx_auto]
u_net_auto = u_train_auto[idx_auto]
v_net_auto = v_train_auto[idx_auto]

x0_net_auto = np.vstack((x0_net_auto0, x0_net_auto))
y0_net_auto = np.vstack((y0_net_auto0, y0_net_auto))
t0_net_auto = np.vstack((t0_net_auto0, t0_net_auto))
x_net_auto = np.vstack((x_net_auto0, x_net_auto))
y_net_auto = np.vstack((y_net_auto0, y_net_auto))
t_net_auto = np.vstack((t_net_auto0, t_net_auto))
u_net_auto = np.vstack((u_net_auto0, u_net_auto))
v_net_auto = np.vstack((v_net_auto0, v_net_auto))


x0_train = np.vstack((x0_train,x0_net_auto))
y0_train = np.vstack((y0_train,y0_net_auto))
t0_train = np.vstack((t0_train,t0_net_auto))
x_net = np.vstack((x_net,x_net_auto))
y_net = np.vstack((y_net,y_net_auto))
t_net = np.vstack((t_net,t_net_auto))
u_net = np.vstack((u_net,u_net_auto))
v_net = np.vstack((v_net,v_net_auto))


# B_gauss_1 = np.random.randn(500, 4)
# B_gauss_1 = B_gauss_1 * 1
# B_gauss_2 = np.random.randn(500, 3)
# B_gauss_2 = B_gauss_2 * 1
# np.savez('B.npz',B1 = B_gauss_1, B2 = B_gauss_2)

B_data = np.load('B.npz')
B_gauss_1 = B_data['B1']
B_gauss_2 = B_data['B2']

layers_1 = [500 * 2, 40, 40, 40, 40, 2]
layers_2 = [500 * 2, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 2]


data = np.load('high_CP.npz')
x_f = data['x']
y_f = data['y']

y_f_max = max(y_f)
y_f = y_f_max-y_f

x_f = 0.082 * 0.001 * x_f
y_f = 0.082 * 0.001 * y_f

x_f = x_f / L
y_f = y_f / L
x_f_train = np.hstack([x_f,x_f,x_f])
y_f_train = np.hstack([y_f,y_f,y_f])


# N_p = len(x0_train)
N_p_t = len(x0_train)
# multiple_in = 1
# multiple_out = 2
multiple = 1
# N_f = multiple * N_p
N_f = multiple * N_p_t
# N_f_t = multiple * N_p
# N_f_in = multiple_in * N_p
# x_in_min = x_train.min(0)
# x_in_max = x_train.max(0)
# y_in_min = y_train.min(0)
# y_in_max = 0.15 / L
t_min = t_net.min(0)
t_max = t_net.max(0)


idx_cp = np.random.choice(x_f_train.shape[0], N_f, replace=False)
x_f_train = x_f_train[idx_cp]
y_f_train = y_f_train[idx_cp]
t0_f_train = t_min + (t_max - t_min) * lhs(1, N_f)
delta_t_train = [random.uniform(0, 0.005) for _ in range(N_f)]
delta_t_train  = np.array(delta_t_train).flatten()[:,None]
t0_f_train = np.array(t0_f_train).flatten()[:,None]
x0_f_train = np.array(x_f_train).flatten()[:,None]
y0_f_train = np.array(y_f_train).flatten()[:,None]
t_f_train = t0_f_train + delta_t_train


# t_f_train = []
# t0_f_train = t_net.flatten().tolist()
#
# for i in range(2000):
#     t_f_train.append(t0_f_train[:delta_i])
# t_f_train = np.array(t_f_train).flatten()[:, None]

len_data = int(x0_train.shape[0])
x0_f_train_batch = x0_f_train[:len_data,:]
y0_f_train_batch = y0_f_train[:len_data,:]
t0_f_train_batch = t0_f_train[:len_data,:]
t_f_train_batch = t_f_train[:len_data,:]
for i in range(multiple - 1):
    j = i + 1
    x0_f_train_batch = np.hstack((x0_f_train_batch, x0_f_train[j * len_data:(j + 1) * len_data, :]))
    y0_f_train_batch = np.hstack((y0_f_train_batch, y0_f_train[j * len_data:(j + 1) * len_data, :]))
    t0_f_train_batch = np.hstack((t0_f_train_batch, t0_f_train[j * len_data:(j + 1) * len_data, :]))
    t_f_train_batch = np.hstack((t_f_train_batch, t_f_train[j * len_data:(j + 1) * len_data, :]))


xyt0_train = np.hstack([x0_train, y0_train, t_net, t0_train, x0_f_train_batch, y0_f_train_batch, t0_f_train_batch, t_f_train_batch])
xyuv_train = np.hstack([x_net, y_net, u_net, v_net])
#xyt0_star = np.hstack([x0_star, y0_star, t_star])
xyuv_test = np.hstack([xyt0_train,xyuv_train])


####################training data plot#############################


t_traindata_plot = t_net / (U / L)
t_traindata_plot = t_traindata_plot

u_traindata_plot = u_net * U
v_traindata_plot = v_net * U
extrater_traindata_plot = []
for i in range(len(t_net)):
    if t_traindata_plot[i]<0.011 and t_traindata_plot[i]>0.009:
        extrater_traindata_plot.append(i)
u_mag_train_plot = (u_traindata_plot[extrater_traindata_plot] ** 2 + v_traindata_plot[extrater_traindata_plot] ** 2) ** 0.5
fig = plt.figure(figsize=(6, 6), dpi=300)
ax = fig.add_subplot(111)


U_train = ax.scatter(x_net[extrater_traindata_plot],y_net[extrater_traindata_plot], c=u_mag_train_plot, cmap='rainbow', s=15, marker='s', alpha=0.9, edgecolors='none', vmax= 0.5, vmin=0, rasterized=True)


# plt.scatter(x_pred,y_pred,s =1, marker="o",c='#36b5fc')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
# plt.legend()
ax.set_xlabel('$x$', size=15)
ax.set_ylabel('$y$', size=15)
# ax.set_title('particle test', fontsize = 20)
plt.axis('equal')
fig.colorbar(U_train, ax=ax, fraction=0.03, shrink=0.5)
plt.xlim([0, 1])  # 设置x轴范围为0到10
plt.ylim([0, 1])
fig.savefig('3_all_train_velo_mag_0.01'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


t_traindata_plot = t_net / (U / L)
t_traindata_plot = t_traindata_plot

u_traindata_plot = u_net * U
v_traindata_plot = v_net * U
extrater_traindata_plot = []
for i in range(len(t_net)):
    if t_traindata_plot[i]<0.201 and t_traindata_plot[i]>0.199:
        extrater_traindata_plot.append(i)
u_mag_train_plot = (u_traindata_plot[extrater_traindata_plot] ** 2 + v_traindata_plot[extrater_traindata_plot] ** 2) ** 0.5
fig = plt.figure(figsize=(6, 6), dpi=300)
ax = fig.add_subplot(111)


U_train = ax.scatter(x_net[extrater_traindata_plot],y_net[extrater_traindata_plot], c=u_mag_train_plot, cmap='rainbow', s=15, marker='s', alpha=0.9, edgecolors='none', vmax= 0.5, vmin=0, rasterized=True)


# plt.scatter(x_pred,y_pred,s =1, marker="o",c='#36b5fc')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
# plt.legend()
ax.set_xlabel('$x$', size=15)
ax.set_ylabel('$y$', size=15)
# ax.set_title('particle test', fontsize = 20)
plt.axis('equal')
fig.colorbar(U_train, ax=ax, fraction=0.03, shrink=0.5)
plt.xlim([0, 1])  # 设置x轴范围为0到10
plt.ylim([0, 1])
fig.savefig('3_all_train_velo_mag_0.2'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

t_traindata_plot = t_net / (U / L)
t_traindata_plot = t_traindata_plot

u_traindata_plot = u_net * U
v_traindata_plot = v_net * U
extrater_traindata_plot = []
for i in range(len(t_net)):
    if t_traindata_plot[i]<0.251 and t_traindata_plot[i]>0.249:
        extrater_traindata_plot.append(i)
u_mag_train_plot = (u_traindata_plot[extrater_traindata_plot] ** 2 + v_traindata_plot[extrater_traindata_plot] ** 2) ** 0.5
fig = plt.figure(figsize=(6, 6), dpi=300)
ax = fig.add_subplot(111)


U_train = ax.scatter(x_net[extrater_traindata_plot],y_net[extrater_traindata_plot], c=u_mag_train_plot, cmap='rainbow', s=15, marker='s', alpha=0.9, edgecolors='none', vmax= 0.5, vmin=0, rasterized=True)


# plt.scatter(x_pred,y_pred,s =1, marker="o",c='#36b5fc')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
# plt.legend()
ax.set_xlabel('$x$', size=15)
ax.set_ylabel('$y$', size=15)
# ax.set_title('particle test', fontsize = 20)
plt.axis('equal')
fig.colorbar(U_train, ax=ax, fraction=0.03, shrink=0.5)
plt.xlim([0, 1])  # 设置x轴范围为0到10
plt.ylim([0, 1])
fig.savefig('3_all_train_velo_mag_0.25'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

t_traindata_plot = t_net / (U / L)
t_traindata_plot = t_traindata_plot

u_traindata_plot = u_net * U
v_traindata_plot = v_net * U
extrater_traindata_plot = []
for i in range(len(t_net)):
    if t_traindata_plot[i]<0.1251 and t_traindata_plot[i]>0.1249:
        extrater_traindata_plot.append(i)
u_mag_train_plot = (u_traindata_plot[extrater_traindata_plot] ** 2 + v_traindata_plot[extrater_traindata_plot] ** 2) ** 0.5
fig = plt.figure(figsize=(6, 6), dpi=300)
ax = fig.add_subplot(111)


U_train = ax.scatter(x_net[extrater_traindata_plot],y_net[extrater_traindata_plot], c=u_mag_train_plot, cmap='rainbow', s=15, marker='s', alpha=0.9, edgecolors='none', vmax= 0.5, vmin=0, rasterized=True)


# plt.scatter(x_pred,y_pred,s =1, marker="o",c='#36b5fc')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
# plt.legend()
ax.set_xlabel('$x$', size=15)
ax.set_ylabel('$y$', size=15)
# ax.set_title('particle test', fontsize = 20)
plt.axis('equal')
fig.colorbar(U_train, ax=ax, fraction=0.03, shrink=0.5)
plt.xlim([0, 1])  # 设置x轴范围为0到10
plt.ylim([0, 1])
fig.savefig('3_all_train_velo_mag_0.125'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


t_traindata_plot = t_net / (U / L)
t_traindata_plot = t_traindata_plot

u_traindata_plot = u_net * U
v_traindata_plot = v_net * U
extrater_traindata_plot = []
for i in range(len(t_net)):
    if t_traindata_plot[i]<0.051 and t_traindata_plot[i]>0.049:
        extrater_traindata_plot.append(i)
u_mag_train_plot = (u_traindata_plot[extrater_traindata_plot] ** 2 + v_traindata_plot[extrater_traindata_plot] ** 2) ** 0.5
fig = plt.figure(figsize=(6, 6), dpi=300)
ax = fig.add_subplot(111)


U_train = ax.scatter(x_net[extrater_traindata_plot],y_net[extrater_traindata_plot], c=u_mag_train_plot, cmap='rainbow', s=15, marker='s', alpha=0.9, edgecolors='none', vmax= 0.5, vmin=0, rasterized=True)


# plt.scatter(x_pred,y_pred,s =1, marker="o",c='#36b5fc')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
# plt.legend()
ax.set_xlabel('$x$', size=15)
ax.set_ylabel('$y$', size=15)
# ax.set_title('particle test', fontsize = 20)
plt.axis('equal')
fig.colorbar(U_train, ax=ax, fraction=0.03, shrink=0.5)
plt.xlim([0, 1])  # 设置x轴范围为0到10
plt.ylim([0, 1])
fig.savefig('3_all_train_velo_mag_0.05'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


t_traindata_plot = t_net / (U / L)
t_traindata_plot = t_traindata_plot

u_traindata_plot = u_net * U
v_traindata_plot = v_net * U
extrater_traindata_plot = []
for i in range(len(t_net)):
    if t_traindata_plot[i]<0.101 and t_traindata_plot[i]>0.099:
        extrater_traindata_plot.append(i)
u_mag_train_plot = (u_traindata_plot[extrater_traindata_plot] ** 2 + v_traindata_plot[extrater_traindata_plot] ** 2) ** 0.5
fig = plt.figure(figsize=(6, 6), dpi=300)
ax = fig.add_subplot(111)


U_train = ax.scatter(x_net[extrater_traindata_plot],y_net[extrater_traindata_plot], c=u_mag_train_plot, cmap='rainbow', s=15, marker='s', alpha=0.9, edgecolors='none', vmax= 0.5, vmin=0, rasterized=True)


# plt.scatter(x_pred,y_pred,s =1, marker="o",c='#36b5fc')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
# plt.legend()
ax.set_xlabel('$x$', size=15)
ax.set_ylabel('$y$', size=15)
# ax.set_title('particle test', fontsize = 20)
plt.axis('equal')
fig.colorbar(U_train, ax=ax, fraction=0.03, shrink=0.5)
plt.xlim([0, 1])  # 设置x轴范围为0到10
plt.ylim([0, 1])
fig.savefig('3_all_train_velo_mag_0.100'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

####################################################################

#jiangcaiyang
# M = []

# for i in range(int(len(x_net))):
#     if i % 100 != 0:
#         M.append(i)
#
# xyt0_train = np.delete(xyt0_train, M, axis=0)
# xyuv_train = np.delete(xyuv_train, M, axis=0)

# N_u = 50000
# idx = np.random.choice(xyt0_train.shape[0], N_u, replace=False)
# xyt0_train = xyt0_train[idx, :]
# xyuv_train = xyuv_train[idx, :]
#
fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)

plt.scatter(x0_train,y0_train,s =2, marker="o")
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.tick_params(labelsize=15)
plt.axis('equal')
plt.show()
##################################################################


xyt0_train = torch.tensor(xyt0_train).float()
xyuv_train = torch.tensor(xyuv_train).float()
B_gauss_1 = torch.tensor(B_gauss_1).float()
B_gauss_2 = torch.tensor(B_gauss_2).float()
train_data = addbatch(xyt0_train,xyuv_train, 5000000)
model = TrajectoryNSNet(B_gauss_1, B_gauss_2)
model.train( train_data)


