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

        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_1 = nn.Parameter(self.lambda_1)
        #将一个不可训练的类型Tensor转换成可以训练的类型parameter并将这个parameter绑定到这个module里面
        # (net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        # 所以经过类型转换这个self.lambda变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。

        self.net_2 = Net2(layers_2).to(device)
        self.net_2.register_parameter('lambda_1', self.lambda_1)  # 将一个参数添加到模块中

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
        self.net_2.load_state_dict(torch.load("suanli3nopretrainnet2.pth"))

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
        lambda_1 = self.lambda_1
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

        self.RE1 = 0.0008 * torch.sigmoid(lambda_1)

        f_u = u_t + u * u_x + v * u_y + p_x - self.RE1 * (u_xx + u_yy)
        f_v = v_t + u * v_x + v * v_y + p_y - self.RE1 * (v_xx + v_yy)

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
            print('It: %d, total_loss: %.5e, loss_u: %.5e, loss_f: %.5e, l1: %.6f' % (self.iter,
                    total_loss.item(),
                    loss_u.item(),
                    loss_f.item(),
                    self.lambda_1.item()
                    ))

        total_loss.backward()

        return total_loss
#训练过程先由Adam进行粗算，再使用LBFGS精算
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
        torch.save(self.net_2.state_dict(), "suanli3net2pretrain_test_FFN_forl1_batch_l1_lbfgs_0.0002_uv_sig.pth")

    def predict(self, x_star, y_star, t_star):
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

#####################自动追踪粒子#############################


data_auto = np.load('BULK_wang.npz')

# 访问保存的数组
x_auto = data_auto['x']
y_auto = data_auto['y']
t_auto = data_auto['t']
u_auto = data_auto['u']
v_auto = data_auto['v']

#################################################################


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

u_plot_max = max(u)
u_plot_min = min(u)
v_plot_max = max(v)
v_plot_min = min(v)
u_plot_auto_max = max(u_auto)
u_plot_auto_min = min(u_auto)
v_plot_auto_max = max(v_auto)
v_plot_auto_min = min(v_auto)

uuu_plot_max = float(max(u_plot_max, u_plot_auto_max))
uuu_plot_min = float(min(u_plot_min, u_plot_auto_min))
vvv_plot_max = float(max(v_plot_max, v_plot_auto_max))
vvv_plot_min = float(min(v_plot_min, v_plot_auto_min))

uuu_plot_cor = uuu_plot_max - uuu_plot_min
vvv_plot_cor = vvv_plot_max - vvv_plot_min
# U = 1
# L = 0.01
x_train = x / L
y_train = y / L
t_train = t * (U / L)
u_train = u / U
v_train = v / U
# p_train = p / (1 * (U ** 2))

##############################################

#############################################
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
# p_net =[]
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
N_p = int(0.8 * len(index))
N_p_test = int(0.15 * len(index))
idx_p = np.random.choice(index.shape[0]-1, N_p, replace=False)
idx_test = np.random.choice(int(np.setdiff1d(index.shape[0]-1, idx_p)), N_p_test, replace=False)
idx_p1 = [i+1 for i in idx_test]
index_N = index[idx_test]
delta_index = index[idx_p1] - index_N
for i in range(int(len(index_N))):
    x0_train.append(x00_train[index_N[i]:index_N[i]+delta_index[i]])
    y0_train.append(y00_train[index_N[i]:index_N[i]+delta_index[i]])
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
x_net = list(flatten(x_net))
y_net = list(flatten(y_net))
u_net = list(flatten(u_net))
v_net = list(flatten(v_net))
# p_net = list(flatten(p_net))
t_net = list(flatten(t_net))

x0_train = np.array(x0_train).flatten()[:, None]
y0_train = np.array(y0_train).flatten()[:, None]
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
x_net_auto0 = x_train_auto0[idx_auto0]
y_net_auto0 = y_train_auto0[idx_auto0]
t_net_auto0 = t_train_auto0[idx_auto0]
u_net_auto0 = u_train_auto0[idx_auto0]
v_net_auto0 = v_train_auto0[idx_auto0]

idx_auto = np.random.choice(x_train_auto.shape[0], int(1 * len(x_net)), replace=False)
x0_net_auto = x_train_auto[idx_auto]
y0_net_auto = y_train_auto[idx_auto]
x_net_auto = x_train_auto[idx_auto]
y_net_auto = y_train_auto[idx_auto]
t_net_auto = t_train_auto[idx_auto]
u_net_auto = u_train_auto[idx_auto]
v_net_auto = v_train_auto[idx_auto]

x0_net_auto = np.vstack((x0_net_auto0, x0_net_auto))
y0_net_auto = np.vstack((y0_net_auto0, y0_net_auto))
x_net_auto = np.vstack((x_net_auto0, x_net_auto))
y_net_auto = np.vstack((y_net_auto0, y_net_auto))
t_net_auto = np.vstack((t_net_auto0, t_net_auto))
u_net_auto = np.vstack((u_net_auto0, u_net_auto))
v_net_auto = np.vstack((v_net_auto0, v_net_auto))

xxx_neet = x_net
yyy_neet = y_net


# x0_train = np.vstack((x0_train,x0_net_auto))
# y0_train = np.vstack((y0_train,y0_net_auto))
# x_net = np.vstack((x_net,x_net_auto))
# y_net = np.vstack((y_net,y_net_auto))
# t_net = np.vstack((t_net,t_net_auto))
# u_net = np.vstack((u_net,u_net_auto))
# v_net = np.vstack((v_net,v_net_auto))

B_data = np.load('B.npz')
B_gauss_1 = B_data['B1']
B_gauss_2 = B_data['B2']
layers_2 = [500 * 2, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 2]
# layers_2 = [500 * 2, 60, 60, 60, 60, 60, 60, 60, 60, 3]
# p_net = np.array(p_net).flatten()[:, None]


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
multiple = 1
N_f = multiple * N_p

data = np.load('BULK_LOW_CP_before.npz')
x_f = data['x']
y_f = data['y']

y_f_max = max(y_f)
y_f = y_f_max-y_f

x_f = 0.082 * 0.001 * x_f
y_f = 0.082 * 0.001 * y_f

x_f = x_f / L
y_f = y_f / L

xxx_snap = x_f
yyy_snap = y_f
x_f = np.hstack([x_f,x_f,x_f])
y_f = np.hstack([y_f,y_f,y_f])

# N_f_in = multiple_in * N_p
# x_in_min = x_train.min(0)
# x_in_max = x_train.max(0)
# y_in_min = y_train.min(0)
# y_in_max = 0.15 / L

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
idx_cp = np.random.choice(x_f.shape[0], N_f, replace=False)
x0_f_train = x_f[idx_cp]
y0_f_train = y_f[idx_cp]

x_f_train = x_f[:,None]
y_f_train = y_f[:,None]
t_f_train = t_min + (t_max - t_min) * lhs(1, N_f)
delta_t_train = [random.uniform(0,  0.005) for _ in range(N_f)]
delta_t_train = np.array(delta_t_train).flatten()[:,None]
delta_t_train = delta_t_train * (U / L)

t0_f_train = np.array(t_f_train).flatten()[:,None]
x0_f_train = np.array(x0_f_train).flatten()[:,None]
y0_f_train = np.array(y0_f_train).flatten()[:,None]
t_f_train = t0_f_train + delta_t_train
# t_f_train = []
# t0_f_train = t_net.flatten().tolist()
#
# for i in range(2000):
#     t_f_train.append(t0_f_train[:delta_i])
# t_f_train = np.array(t_f_train).flatten()[:, None]
#
# len_data = int(x_net.shape[0])
# x0_f_train_batch = x0_f_train[:len_data,:]
# y0_f_train_batch = y0_f_train[:len_data,:]
# t0_f_train_batch = t0_f_train[:len_data,:]
# t_f_train_batch = t_f_train[:len_data,:]
# for i in range(multiple - 1):
#     j = i + 1
#     x0_f_train_batch = np.hstack((x0_f_train_batch, x0_f_train[j * len_data:(j + 1) * len_data, :]))
#     y0_f_train_batch = np.hstack((y0_f_train_batch, y0_f_train[j * len_data:(j + 1) * len_data, :]))
#     t0_f_train_batch = np.hstack((t0_f_train_batch, t0_f_train[j * len_data:(j + 1) * len_data, :]))
#     t_f_train_batch = np.hstack((t_f_train_batch, t_f_train[j * len_data:(j + 1) * len_data, :]))
# # extrater_cylinder = []
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


# xyt0_train = np.hstack([x_net, y_net, t_net, x0_f_train_batch, y0_f_train_batch, t_f_train_batch])
xyt0_train = np.hstack([x_net, y_net, t_net, x0_f_train, y0_f_train, t_f_train])
xyuv_train = np.hstack([u_net, v_net])
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

# N_u = 20000
# idx = np.random.choice(xyt0_train.shape[0], N_u, replace=False)
# xyt0_train = xyt0_train[idx, :]
# xyuv_train = xyuv_train[idx, :]
#
fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)

plt.scatter(xyt0_train[:,0:1],xyt0_train[:,1:2],s =0.1, marker="o")
# plt.scatter(xyt0_train[:,3:4],xyt0_train[:,5:6],s =0.1, marker="o",c='r')
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.tick_params(labelsize=15)
plt.axis('equal')
plt.show()
##################################################################


# xyt0_train = torch.tensor(xyt0_train, requires_grad=True).float().to(device)
# xyuv_train = torch.tensor(xyuv_train, requires_grad=True).float().to(device)

# torch.set_default_tensor_type(torch.DoubleTensor)
xyt0_train = torch.tensor(xyt0_train).float()
xyuv_train = torch.tensor(xyuv_train).float()
B_gauss_2 = torch.tensor(B_gauss_2).float()


# model = TrajectoryNSNet(xyt0_train,  xyuv_train, B_gauss_10)
train_data = addbatch(xyt0_train,xyuv_train, 30000)
model = TrajectoryNSNet(B_gauss_2)



u_pred, v_pred, p_pred = model.predict(x_train, y_train, t_train)
lambda_1 = 0.0008 * torch.sigmoid(model.lambda_1)
lambda_1_value = lambda_1.detach().cpu().numpy()

error_u = np.linalg.norm(u_train - u_pred, 2) / np.linalg.norm(u_train, 2)
error_v = np.linalg.norm(v_train - v_pred, 2) / np.linalg.norm(v_train, 2)


print('Error u: %e' % (error_u))
print('Error v: %e' % (error_v))
print('l1: %.5f' % (1 / lambda_1_value))



wangge_u_pearsonr = u_train.flatten()
wangge_v_pearsonr = v_train.flatten()
u_test_pred_pearsonr = u_pred.flatten()
v_test_pred_pearsonr = v_pred.flatten()

coefficient_u_test ,_= stats.pearsonr(wangge_u_pearsonr,u_test_pred_pearsonr)
coefficient_v_test,_ = stats.pearsonr(wangge_v_pearsonr,v_test_pred_pearsonr)
print('Coefficient u test: %f' % (coefficient_u_test))
print('Coefficient v test: %f' % (coefficient_v_test))


folder_path = '/suanli3_matlab_npz'
npz_file = '000025.mat'
with h5py.File(os.path.join(folder_path, npz_file), 'r') as data_temp:
    y_temp = np.array(data_temp['ymesh']).flatten()[:, None]
    x_temp = np.array(data_temp['xmesh']).flatten()[:, None]
    u_temp = np.array(data_temp['u']).flatten()[:, None]
    v_temp = np.array(data_temp['v']).flatten()[:, None]
data_temp = np.hstack((x_temp, y_temp, u_temp, v_temp))
data_temp = data_temp[~np.isnan(data_temp).any(axis=1)]
print(f"处理文件: {npz_file}, 数据形状: {data_temp.shape}")  # 打印数据形状

# 计算 y_f_max 和 yyy_snap
xxx_snap = data_temp[:, 0][:, None]
yyy_snap = data_temp[:, 1][:, None]
uuu_snap = data_temp[:, 2][:, None]
vvv_snap = data_temp[:, 3][:, None]

yyy_snap_max = max(yyy_snap)
yyy_snap = yyy_snap_max - yyy_snap
vvv_snap = -vvv_snap

# 缩放
xxx_snap = xxx_snap / L
yyy_snap = yyy_snap / L

# 计算 ttt_snap
file_number = int(int(re.search(r'\d+', npz_file).group()))  # 从文件名中提取数字
print(file_number)
ttt_snap = 0.005 * file_number * (U / L)
ttt_snap = np.full((len(xxx_snap), 1), ttt_snap)  # 创建相应形状的数组

# Flatten arrays
xxx_snap = np.array(xxx_snap).flatten()[:, None]
yyy_snap = np.array(yyy_snap).flatten()[:, None]

# 预测
u_pred, v_pred, p_pred = model.predict(xxx_snap, yyy_snap, ttt_snap)
u_pred = u_pred * U
v_pred = v_pred * U



fig = plt.figure(dpi=600,figsize=(5,2))
ax1= fig.add_subplot(121,aspect='equal')
ax2= fig.add_subplot(122,aspect='equal')
U_pred = ax2.scatter(xxx_snap,yyy_snap, c=u_pred, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True)
U_true = ax1.scatter(xxx_snap,yyy_snap, c=uuu_snap, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True)
# ax1.set_xlabel('$x$', size=10)
# ax2.set_xlabel('$x$', size=10)
# ax1.set_ylabel('$y$', size=10)
# ax2.set_ylabel('$y$', size=10)
ax1.set_title('u true t=25s', fontsize = 10)
ax2.set_title('u predict t=25s', fontsize = 10)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(hspace=10.0)
clim = U_true.get_clim()
# 创建包含五个刻度位置的数组
ticks_positions = np.linspace(clim[0], clim[1], 5)

# 添加颜色条，并指定其宽度、长度、刻度位置及格式
cbar = fig.colorbar(U_true, ax=[ax1, ax2], fraction=0.046, pad=0.04, ticks=ticks_positions, format='%.2f')
cbar.ax.tick_params(labelsize=8)  # 设置刻度标签字体大小

fig.savefig('3_u_25_25s' + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()



fig = plt.figure(dpi=600,figsize=(5,2))
ax1= fig.add_subplot(121,aspect='equal')
ax2= fig.add_subplot(122,aspect='equal')
V_pred = ax2.scatter(xxx_snap,yyy_snap, c=v_pred, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True)
V_true = ax1.scatter(xxx_snap,yyy_snap, c=vvv_snap, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True)

ax1.set_title('v true t=25s', fontsize = 10)
ax2.set_title('v predict t=25s', fontsize = 10)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(hspace=10.0)
clim = V_true.get_clim()
# 创建包含五个刻度位置的数组
ticks_positions = np.linspace(clim[0], clim[1], 5)

# 添加颜色条，并指定其宽度、长度、刻度位置及格式
cbar = fig.colorbar(V_true, ax=[ax1, ax2], fraction=0.046, pad=0.04, ticks=ticks_positions, format='%.2f')
cbar.ax.tick_params(labelsize=8)  # 设置刻度标签字体大小
fig.savefig('3_v_25_25s'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


######################################################################################取1 25 50
folder_path = '/home/wanjingdi/code/reproductionPINN/suanli3_matlab_npz'
npz_file = '000002.mat'
with h5py.File(os.path.join(folder_path, npz_file), 'r') as data_temp:
    y_temp = np.array(data_temp['ymesh']).flatten()[:, None]
    x_temp = np.array(data_temp['xmesh']).flatten()[:, None]
    u_temp = np.array(data_temp['u']).flatten()[:, None]
    v_temp = np.array(data_temp['v']).flatten()[:, None]
data_temp = np.hstack((x_temp, y_temp, u_temp, v_temp))
data_temp = data_temp[~np.isnan(data_temp).any(axis=1)]
print(f"处理文件: {npz_file}, 数据形状: {data_temp.shape}")  # 打印数据形状

# 计算 y_f_max 和 yyy_snap
xxx_snap = data_temp[:, 0][:, None]
yyy_snap = data_temp[:, 1][:, None]
uuu_snap = data_temp[:, 2][:, None]
vvv_snap = data_temp[:, 3][:, None]

yyy_snap_max = max(yyy_snap)
yyy_snap = yyy_snap_max - yyy_snap
vvv_snap = -vvv_snap
u_mag = (uuu_snap ** 2 + vvv_snap ** 2) ** 0.5
# 缩放
xxx_snap = xxx_snap / L
yyy_snap = yyy_snap / L

# 计算 ttt_snap
file_number = int(int(re.search(r'\d+', npz_file).group()))  # 从文件名中提取数字
print(file_number)
ttt_snap = 0.005 * file_number * (U / L)
ttt_snap = np.full((len(xxx_snap), 1), ttt_snap)  # 创建相应形状的数组

# Flatten arrays
xxx_snap = np.array(xxx_snap).flatten()[:, None]
yyy_snap = np.array(yyy_snap).flatten()[:, None]

# 预测
u_pred, v_pred, p_pred = model.predict(xxx_snap, yyy_snap, ttt_snap)
u_pred = u_pred * U
v_pred = v_pred * U
u_mag_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5
NE_u_2 = abs(u_pred - uuu_snap) / uuu_plot_cor
NE_v_2 = abs(v_pred - vvv_snap) / vvv_plot_cor
NE_mag_2 = abs(u_mag_pred - u_mag) / 0.5
np.savetxt('3NE_mag_2.csv', NE_mag_2, fmt='%.18f', delimiter=',', newline='\n')
# np.savetxt('3NE_u_2.csv', NE_u_2, fmt='%.18f', delimiter=',', newline='\n')
# np.savetxt('3NE_v_2.csv', NE_v_2, fmt='%.18f', delimiter=',', newline='\n')
fig = plt.figure(dpi=600,figsize=(5,2))
ax1= fig.add_subplot(121,aspect='equal')
ax2= fig.add_subplot(122,aspect='equal')
U_pred = ax2.scatter(xxx_snap,yyy_snap, c=u_mag_pred, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True, vmax= 0.5, vmin=0)
U_true = ax1.scatter(xxx_snap,yyy_snap, c=u_mag, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True, vmax= 0.5, vmin=0)
# ax1.set_xlabel('$x$', size=10)
# ax2.set_xlabel('$x$', size=10)
# ax1.set_ylabel('$y$', size=10)
# ax2.set_ylabel('$y$', size=10)
ax1.set_title('u true t=2s', fontsize = 10)
ax2.set_title('u predict t=2s', fontsize = 10)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(hspace=10.0)
clim = U_true.get_clim()
# 创建包含五个刻度位置的数组
ticks_positions = np.linspace(clim[0], clim[1], 5)

# 添加颜色条，并指定其宽度、长度、刻度位置及格式
cbar = fig.colorbar(U_true, ax=[ax1, ax2], fraction=0.046, pad=0.04, ticks=ticks_positions, format='%.2f')
cbar.ax.tick_params(labelsize=8)  # 设置刻度标签字体大小
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3_u_mag_2' + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

npz_file = '000025.mat'
with h5py.File(os.path.join(folder_path, npz_file), 'r') as data_temp:
    y_temp = np.array(data_temp['ymesh']).flatten()[:, None]
    x_temp = np.array(data_temp['xmesh']).flatten()[:, None]
    u_temp = np.array(data_temp['u']).flatten()[:, None]
    v_temp = np.array(data_temp['v']).flatten()[:, None]
data_temp = np.hstack((x_temp, y_temp, u_temp, v_temp))
data_temp = data_temp[~np.isnan(data_temp).any(axis=1)]
print(f"处理文件: {npz_file}, 数据形状: {data_temp.shape}")  # 打印数据形状

# 计算 y_f_max 和 yyy_snap
xxx_snap = data_temp[:, 0][:, None]
yyy_snap = data_temp[:, 1][:, None]
uuu_snap = data_temp[:, 2][:, None]
vvv_snap = data_temp[:, 3][:, None]

yyy_snap_max = max(yyy_snap)
yyy_snap = yyy_snap_max - yyy_snap
vvv_snap = -vvv_snap
u_mag = (uuu_snap ** 2 + vvv_snap ** 2) ** 0.5
# 缩放
xxx_snap = xxx_snap / L
yyy_snap = yyy_snap / L

# 计算 ttt_snap
file_number = int(int(re.search(r'\d+', npz_file).group()))  # 从文件名中提取数字
print(file_number)
ttt_snap = 0.005 * file_number * (U / L)
ttt_snap = np.full((len(xxx_snap), 1), ttt_snap)  # 创建相应形状的数组

# Flatten arrays
xxx_snap = np.array(xxx_snap).flatten()[:, None]
yyy_snap = np.array(yyy_snap).flatten()[:, None]

# 预测
u_pred, v_pred, p_pred = model.predict(xxx_snap, yyy_snap, ttt_snap)
u_pred = u_pred * U
v_pred = v_pred * U
u_mag_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5
NE_u_25 = abs(u_pred - uuu_snap) / uuu_plot_cor
NE_v_25 = abs(v_pred - vvv_snap) / vvv_plot_cor
NE_mag_25 = abs(u_mag_pred - u_mag) / 0.5
np.savetxt('3NE_mag_25.csv', NE_mag_25, fmt='%.18f', delimiter=',', newline='\n')

fig = plt.figure(dpi=600,figsize=(5,2))
ax1= fig.add_subplot(121,aspect='equal')
ax2= fig.add_subplot(122,aspect='equal')
U_pred = ax2.scatter(xxx_snap,yyy_snap, c=u_mag_pred, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True, vmax= 0.5, vmin=0)
U_true = ax1.scatter(xxx_snap,yyy_snap, c=u_mag, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True, vmax= 0.5, vmin=0)
# ax1.set_xlabel('$x$', size=10)
# ax2.set_xlabel('$x$', size=10)
# ax1.set_ylabel('$y$', size=10)
# ax2.set_ylabel('$y$', size=10)
ax1.set_title('u true t=25s', fontsize = 10)
ax2.set_title('u predict t=25s', fontsize = 10)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(hspace=10.0)
clim = U_true.get_clim()
# 创建包含五个刻度位置的数组
ticks_positions = np.linspace(clim[0], clim[1], 5)

# 添加颜色条，并指定其宽度、长度、刻度位置及格式
cbar = fig.colorbar(U_true, ax=[ax1, ax2], fraction=0.046, pad=0.04, ticks=ticks_positions, format='%.2f')
cbar.ax.tick_params(labelsize=8)  # 设置刻度标签字体大小
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3_u_mag_25' + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

npz_file = '000050.mat'
with h5py.File(os.path.join(folder_path, npz_file), 'r') as data_temp:
    y_temp = np.array(data_temp['ymesh']).flatten()[:, None]
    x_temp = np.array(data_temp['xmesh']).flatten()[:, None]
    u_temp = np.array(data_temp['u']).flatten()[:, None]
    v_temp = np.array(data_temp['v']).flatten()[:, None]
data_temp = np.hstack((x_temp, y_temp, u_temp, v_temp))
data_temp = data_temp[~np.isnan(data_temp).any(axis=1)]
print(f"处理文件: {npz_file}, 数据形状: {data_temp.shape}")  # 打印数据形状

# 计算 y_f_max 和 yyy_snap
xxx_snap = data_temp[:, 0][:, None]
yyy_snap = data_temp[:, 1][:, None]
uuu_snap = data_temp[:, 2][:, None]
vvv_snap = data_temp[:, 3][:, None]

yyy_snap_max = max(yyy_snap)
yyy_snap = yyy_snap_max - yyy_snap
vvv_snap = -vvv_snap
u_mag = (uuu_snap ** 2 + vvv_snap ** 2) ** 0.5
# 缩放
xxx_snap = xxx_snap / L
yyy_snap = yyy_snap / L

# 计算 ttt_snap
file_number = int(int(re.search(r'\d+', npz_file).group()))  # 从文件名中提取数字
print(file_number)
ttt_snap = 0.005 * file_number * (U / L)
ttt_snap = np.full((len(xxx_snap), 1), ttt_snap)  # 创建相应形状的数组

# Flatten arrays
xxx_snap = np.array(xxx_snap).flatten()[:, None]
yyy_snap = np.array(yyy_snap).flatten()[:, None]

# 预测
u_pred, v_pred, p_pred = model.predict(xxx_snap, yyy_snap, ttt_snap)
u_pred = u_pred * U
v_pred = v_pred * U
u_mag_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5
NE_u_50 = abs(u_pred - uuu_snap) / uuu_plot_cor
NE_v_50 = abs(v_pred - vvv_snap) / vvv_plot_cor
NE_mag_50 = abs(u_mag_pred - u_mag) / 0.5
np.savetxt('3NE_mag_50.csv', NE_mag_50, fmt='%.18f', delimiter=',', newline='\n')

fig = plt.figure(dpi=600,figsize=(5,2))
ax1= fig.add_subplot(121,aspect='equal')
ax2= fig.add_subplot(122,aspect='equal')
U_pred = ax2.scatter(xxx_snap,yyy_snap, c=u_mag_pred, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True, vmax= 0.5, vmin=0)
U_true = ax1.scatter(xxx_snap,yyy_snap, c=u_mag, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True, vmax= 0.5, vmin=0)
# ax1.set_xlabel('$x$', size=10)
# ax2.set_xlabel('$x$', size=10)
# ax1.set_ylabel('$y$', size=10)
# ax2.set_ylabel('$y$', size=10)
ax1.set_title('u true t=50s', fontsize = 10)
ax2.set_title('u predict t=50s', fontsize = 10)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(hspace=10.0)
clim = U_true.get_clim()
# 创建包含五个刻度位置的数组
ticks_positions = np.linspace(clim[0], clim[1], 5)

# 添加颜色条，并指定其宽度、长度、刻度位置及格式
cbar = fig.colorbar(U_true, ax=[ax1, ax2], fraction=0.046, pad=0.04, ticks=ticks_positions, format='%.2f')
cbar.ax.tick_params(labelsize=8)  # 设置刻度标签字体大小
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3_u_mag_50' + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

npz_file = '000040.mat'
with h5py.File(os.path.join(folder_path, npz_file), 'r') as data_temp:
    y_temp = np.array(data_temp['ymesh']).flatten()[:, None]
    x_temp = np.array(data_temp['xmesh']).flatten()[:, None]
    u_temp = np.array(data_temp['u']).flatten()[:, None]
    v_temp = np.array(data_temp['v']).flatten()[:, None]
data_temp = np.hstack((x_temp, y_temp, u_temp, v_temp))
data_temp = data_temp[~np.isnan(data_temp).any(axis=1)]
print(f"处理文件: {npz_file}, 数据形状: {data_temp.shape}")  # 打印数据形状

# 计算 y_f_max 和 yyy_snap
xxx_snap = data_temp[:, 0][:, None]
yyy_snap = data_temp[:, 1][:, None]
uuu_snap = data_temp[:, 2][:, None]
vvv_snap = data_temp[:, 3][:, None]

yyy_snap_max = max(yyy_snap)
yyy_snap = yyy_snap_max - yyy_snap
vvv_snap = -vvv_snap
u_mag = (uuu_snap ** 2 + vvv_snap ** 2) ** 0.5
# 缩放
xxx_snap = xxx_snap / L
yyy_snap = yyy_snap / L

# 计算 ttt_snap
file_number = int(int(re.search(r'\d+', npz_file).group()))  # 从文件名中提取数字
print(file_number)
ttt_snap = 0.005 * file_number * (U / L)
ttt_snap = np.full((len(xxx_snap), 1), ttt_snap)  # 创建相应形状的数组

# Flatten arrays
xxx_snap = np.array(xxx_snap).flatten()[:, None]
yyy_snap = np.array(yyy_snap).flatten()[:, None]

# 预测
u_pred, v_pred, p_pred = model.predict(xxx_snap, yyy_snap, ttt_snap)
u_pred = u_pred * U
v_pred = v_pred * U
u_mag_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5
NE_u_40 = abs(u_pred - uuu_snap) / uuu_plot_cor
NE_v_40 = abs(v_pred - vvv_snap) / vvv_plot_cor
NE_mag_40 = abs(u_mag_pred - u_mag) / 0.5
np.savetxt('3NE_mag_40.csv', NE_mag_40, fmt='%.18f', delimiter=',', newline='\n')

fig = plt.figure(dpi=600,figsize=(5,2))
ax1= fig.add_subplot(121,aspect='equal')
ax2= fig.add_subplot(122,aspect='equal')
U_pred = ax2.scatter(xxx_snap,yyy_snap, c=u_mag_pred, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True, vmax= 0.5, vmin=0)
U_true = ax1.scatter(xxx_snap,yyy_snap, c=u_mag, cmap='rainbow', s=1, marker=',', alpha=0.9, edgecolors='none', rasterized=True, vmax= 0.5, vmin=0)
# ax1.set_xlabel('$x$', size=10)
# ax2.set_xlabel('$x$', size=10)
# ax1.set_ylabel('$y$', size=10)
# ax2.set_ylabel('$y$', size=10)
ax1.set_title('u true t=40s', fontsize = 10)
ax2.set_title('u predict t=40s', fontsize = 10)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(False)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)
fig.subplots_adjust(hspace=10.0)
clim = U_true.get_clim()
# 创建包含五个刻度位置的数组
ticks_positions = np.linspace(clim[0], clim[1], 5)

# 添加颜色条，并指定其宽度、长度、刻度位置及格式
cbar = fig.colorbar(U_true, ax=[ax1, ax2], fraction=0.046, pad=0.04, ticks=ticks_positions, format='%.2f')
cbar.ax.tick_params(labelsize=8)  # 设置刻度标签字体大小
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3_u_mag_40' + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()
# fig = plt.figure(figsize=(10,10),dpi = 330 )
# ax = fig.add_subplot(111)
#
# sc = ax.scatter(x_net,y_net ,c = u_net,s =10, marker="o", vmin=-1, vmax=0.5)
# cbar = plt.colorbar(sc, ax=ax)
# # 为了在图中显示颜色条，我们需要创建一个ScalarMappable对象
# ax.set_aspect('equal')
# ax.set_title('u true ', fontsize = 20)
# plt.show()
#
# fig = plt.figure(figsize=(10,10),dpi = 330 )
# ax = fig.add_subplot(111)
#
# sc = ax.scatter(x_net,y_net ,c = v_net,s =10, marker="o", vmin=-0.5, vmax=0.5)
# cbar = plt.colorbar(sc, ax=ax)
# # 为了在图中显示颜色条，我们需要创建一个ScalarMappable对象
# ax.set_aspect('equal')
# ax.set_title('v true ', fontsize = 20)
# plt.show()
#
# fig = plt.figure(figsize=(10,10),dpi = 330 )
# ax = fig.add_subplot(111)
#
# sc = ax.scatter(x_net,y_net ,c = u_pred,s =10, marker="o", vmin=-1, vmax=0.5)
# cbar = plt.colorbar(sc, ax=ax)
# # 为了在图中显示颜色条，我们需要创建一个ScalarMappable对象
# ax.set_aspect('equal')
# ax.set_title('u pred ', fontsize = 20)
# plt.show()
#
# fig = plt.figure(figsize=(10,10),dpi = 330 )
# ax = fig.add_subplot(111)
#
# sc = ax.scatter(x_net,y_net ,c = v_pred,s =10, marker="o", vmin=-0.5, vmax=0.5)
# cbar = plt.colorbar(sc, ax=ax)
# # 为了在图中显示颜色条，我们需要创建一个ScalarMappable对象
# ax.set_aspect('equal')
# ax.set_title('v pred ', fontsize = 20)

# plt.show()

#
#
# data = np.load('25_CP_before.npz')
# xxx_snap = data['x']
# yyy_snap = data['y']
#
# # y_f_max = max(y_f)
# yyy_snap = y_f_max-yyy_snap
#
# xxx_snap = 0.082 * 0.001 * xxx_snap
# yyy_snap = 0.082 * 0.001 * yyy_snap
#
# xxx_snap = xxx_snap / L
# yyy_snap = yyy_snap / L
#
#
#
#
# ttt_snap = []
# for _ in range(int(len(xxx_snap))):
#     ttt_snap.append(400 * 1e-6 * 25 * (U / L))
# ttt_snap = np.array(ttt_snap).flatten()[:, None]
#
# xxx_snap = np.array(xxx_snap).flatten()[:, None]
# yyy_snap = np.array(yyy_snap).flatten()[:, None]
#
#
#
# u_pred, v_pred, p_pred = model.predict(xxx_snap, yyy_snap, ttt_snap)
#
#
# # cmap = mcolors.LinearSegmentedColormap.from_list(
# #     'custom_cmap',
# #     [(0, 'blue'), (0.5, 'white'), (1, 'red')]
# # )
# cmap = mcolors.LinearSegmentedColormap.from_list(
#     'custom_cmap',
#     [(0, 'blue'), (1, 'yellow')]
# )
# # 绘制图像
# fig = plt.figure(figsize=(10, 10), dpi=330)
# ax = fig.add_subplot(111)
#
# # 绘制散点图
# sc = ax.scatter(xxx_snap, yyy_snap, c=v_pred, s=0.005, marker="o", cmap='rainbow', vmin=-0.5, vmax=0.5)
#
# # 添加颜色条
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('v_pred values')  # 设置颜色条标签
#
# # 设置图形属性
# ax.set_aspect('equal')
# ax.set_title('v pred t=25', fontsize=20)
# plt.show()

# fig = plt.figure(figsize=(10, 10), dpi=330)
# ax = fig.add_subplot(111)
#
# # 绘制散点图
# sc = ax.scatter(xxx_snap, yyy_snap, c=u_pred, s=0.005, marker="o", cmap='rainbow', vmin=-0.5, vmax=0.5)
#
# # 添加颜色条
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('v_pred values')  # 设置颜色条标签
#
# # 设置图形属性
# ax.set_aspect('equal')
# ax.set_title('u pred t=25', fontsize=20)
# plt.show()
#
#
# fig = plt.figure(figsize=(10, 10), dpi=330)
# ax = fig.add_subplot(111)
# uuu = (u_pred ** 2 + v_pred **2) ** 0.5
# # 绘制散点图
# sc = ax.scatter(xxx_snap, yyy_snap, c=uuu, s=0.005, marker="o", cmap='rainbow', vmin=-0.5, vmax=0.5)
#
# # 添加颜色条
# cbar = plt.colorbar(sc, ax=ax)
# cbar.set_label('v_pred values')  # 设置颜色条标签
#
# # 设置图形属性
# ax.set_aspect('equal')
# ax.set_title('u pred sc t=25', fontsize=20)
# plt.show()
##############################################################################################
#
# output_video = 'output_video_u_magnitude_test.avi'  # 输出视频文件名
# fps = 1  # 视频帧率
#
# # 获取文件夹中所有 .npz 文件
# folder_path = '/home/wanjingdi/code/reproductionPINN/suanli3_matlab_npz'
# # folder_path = '/home/wanjingdi/code/reproductionPINN/suanli3test'
# # 替换为您的文件夹路径
# npz_files =  sorted(
#     [f for f in os.listdir(folder_path) if f.endswith('.mat')]  # 直接排序文件名
# )
#
# # 创建视频写入对象
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# first_frame = True
# x_min, x_max = 0, 1  # 替换为您的 x 轴范围
# y_min, y_max = 0, 1  # 替换为您的 y 轴范围
#
# for npz_file in npz_files:
#     with h5py.File(os.path.join(folder_path, npz_file), 'r') as data_temp:
#         y_temp = np.array(data_temp['ymesh']).flatten()[:, None]
#         x_temp = np.array(data_temp['xmesh']).flatten()[:, None]
#         u_temp = np.array(data_temp['u']).flatten()[:, None]
#         v_temp = np.array(data_temp['v']).flatten()[:, None]
#     data_temp = np.hstack((x_temp, y_temp, u_temp, v_temp))
#     data_temp = data_temp[~np.isnan(data_temp).any(axis=1)]
#     print(f"处理文件: {npz_file}, 数据形状: {data_temp.shape}")  # 打印数据形状
#
#     if data_temp.shape[0] == 0:  # 如果没有有效数据，跳过
#         print(f"文件 {npz_file} 没有有效数据，跳过。")
#         continue
#     # 计算 y_f_max 和 yyy_snap
#     xxx_snap = data_temp[:, 0][:, None]
#     yyy_snap = data_temp[:, 1][:, None]
#     uuu_snap = data_temp[:, 2][:, None]
#     vvv_snap = data_temp[:, 3][:, None]
#
#     yyy_snap_max = max(yyy_snap)
#     yyy_snap = yyy_snap_max - yyy_snap
#     vvv_snap = -vvv_snap
#
#     # 缩放
#     xxx_snap = xxx_snap / L
#     yyy_snap = yyy_snap / L
#
#     # 计算 ttt_snap
#     file_number = int(int(re.search(r'\d+', npz_file).group()))  # 从文件名中提取数字
#     print(file_number)
#     ttt_snap =  0.005 * file_number * (U / L)
#     ttt_snap = np.full((len(xxx_snap), 1), ttt_snap)  # 创建相应形状的数组
#
#     # Flatten arrays
#     xxx_snap = np.array(xxx_snap).flatten()[:, None]
#     yyy_snap = np.array(yyy_snap).flatten()[:, None]
#
#     # 预测
#     u_pred, v_pred, p_pred = model.predict(xxx_snap, yyy_snap, ttt_snap)
#     u_pred = u_pred * U
#     v_pred = v_pred * U
#     u_scalar =  (u_pred ** 2 + v_pred ** 2) ** 0.5
#     u_true_mag = (uuu_snap ** 2 + vvv_snap ** 2) ** 0.5
#     # 绘制图像
#     fig = plt.figure(dpi=600, figsize=(5, 2))
#     ax1 = fig.add_subplot(121, aspect='equal')
#     ax2 = fig.add_subplot(122, aspect='equal')
#     U_pred = ax2.scatter(xxx_snap, yyy_snap, c=u_scalar, cmap='rainbow', s=1, marker=',', alpha=0.9,
#                          edgecolors='none', rasterized=True, vmax=0.5, vmin=0)
#     U_true = ax1.scatter(xxx_snap, yyy_snap, c=u_true_mag, cmap='rainbow', s=1, marker=',', alpha=0.9,
#                          edgecolors='none',
#                          rasterized=True, vmax=0.5, vmin=0)
#
#     # 添加颜色条
#     ax1.set_xticks([])
#     ax1.set_yticks([])
#     for spine in ax1.spines.values():
#         spine.set_visible(False)
#     ax2.set_xticks([])
#     ax2.set_yticks([])
#     for spine in ax2.spines.values():
#         spine.set_visible(False)
#     fig.subplots_adjust(hspace=10.0)
#     clim = U_true.get_clim()
#     # 创建包含五个刻度位置的数组
#     ticks_positions = np.linspace(clim[0], clim[1], 5)
#
#     # 添加颜色条，并指定其宽度、长度、刻度位置及格式
#     cbar = fig.colorbar(U_true, ax=[ax1, ax2], fraction=0.046, pad=0.04, ticks=ticks_positions, format='%.2f')
#     cbar.ax.tick_params(labelsize=8)  # 设置刻度标签字体大小
#     # plt.rcParams['svg.fonttype'] = 'none'
#     # 设置标题
#     ax1.set_title(f'u truth t={file_number}', fontsize=8)
#     ax2.set_title(f'u prediction t={file_number}', fontsize=8)
#     # ax.set_aspect('equal')
#     # 保存图像
#     plt.savefig('temp_image.png', bbox_inches='tight', pad_inches=0)
#     plt.close(fig)
#
#     # 读取图像并写入视频
#     img = cv2.imread('temp_image.png')
#     if first_frame:
#         height, width, layers = img.shape
#         video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
#         first_frame = False
#     video_writer.write(img)
#
# # 释放视频写入对象
# video_writer.release()
#
# # 删除临时图像
# os.remove('temp_image.png')
#
# print("视频生成完成！")