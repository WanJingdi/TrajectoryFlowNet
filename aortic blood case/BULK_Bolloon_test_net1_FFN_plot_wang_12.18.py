import random

import onnx
###########u_t 不添加到 totalloss 损失函数形式与原始PINNs相同##############
###########不收敛，每次随机结果不一#############################
import os
import torch
import torch.nn as nn
import numpy as np
from numpy import genfromtxt
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
from collections import OrderedDict
from pyDOE import lhs
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from itertools import product, combinations
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import netron
from onnx import shape_inference
from more_itertools import flatten
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.animation as animation
import scipy.stats as stats
from matplotlib.patches import Polygon

os.environ['CUDA_VISIBLE_DEVICES']='5'
np.random.seed(1234)
torch.cuda.manual_seed_all(1234)
torch.manual_seed(1234)
device = torch.device('cuda')


# 自定义第一个神经网络
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
            # layers_list.append(
            #     ('layer_norm%d' % i, nn.LayerNorm(layers_2[i + 1]))
            # )
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



class TrajectoryNSNet():
    def __init__(self, B1):
        self.layers_1 = layers_1
        self.layers_2 = layers_2
        self.B1 = B1.to(device)
        self.net_1 = Net1(layers_1).to(device)

        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        self.lambda_1 = nn.Parameter(self.lambda_1)

        self.net_1.load_state_dict(torch.load("suanli3nopretrainnet1.pth"))

        self.optimizer = torch.optim.LBFGS(
            self.net_1.parameters(),
            lr=1.0,
            max_iter=500000,
            max_eval=500000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )
        self.optimizer_Adam = torch.optim.Adam(self.net_1.parameters(), lr=1e-3)
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_Adam, gamma=0.99999996)
        self.iter = 0


    def net_x(self, x0, y0, t1, t0, B1):
        x, y, dt = self.net_1(x0, y0, t1, t0, B1)

        return x, y, dt



    def closure(self):
        self.optimizer.zero_grad()
        self.x_pred, self.y_pred, _ = self.net_x(self.x0, self.y0, self.t1, self.t0, self.B1)
        loss_x = torch.mean((self.x - self.x_pred) ** 2) + \
                 torch.mean((self.y - self.y_pred) ** 2)
        # loss_u = torch.mean((self.u - self.u_pred) ** 2) + \
        #          torch.mean((self.v - self.v_pred) ** 2)
        # loss_p = torch.mean((self.p - self.p_pred) ** 2)
        # loss_f = torch.mean((self.f_u_pred) ** 2) + \
        #          torch.mean((self.f_v_pred) ** 2)

        total_loss = loss_x
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'It: %d, total_loss: %.5e, loss_x: %.5e' %
                (
                    self.iter,
                    total_loss.item(),
                    loss_x.item(),

                )
            )


        total_loss.backward()
        return total_loss

    def train(self, traindata):
        self.net_1.train()
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
        # torch.save(self.net_1.state_dict(), "boxnet1phydim.pth")

    def predict_x(self, x_star, y_star, t_star, t0_star, B1):
        self.x = torch.tensor(x_star, requires_grad=True).float().to(device)
        self.y = torch.tensor(y_star, requires_grad=True).float().to(device)
        self.t1 = torch.tensor(t_star, requires_grad=True).float().to(device)  # 将t分离出来
        self.t0 = torch.tensor(t0_star, requires_grad=True).float().to(device)
        self.B1 = torch.tensor(B1, requires_grad=True).float().to(device)
        self.net_1.eval()  # 设置predict模式
        x_star, y_star,_= self.net_x(self.x, self.y, self.t1, self.t0, self.B1)



        x_star = x_star.detach().cpu().numpy()
        y_star = y_star.detach().cpu().numpy()

        return x_star, y_star



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


# layers_2 = [3, 60, 60, 60, 60, 60, 60, 2]
# Load Data

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
idx_p1 = [i+1 for i in idx_test]
index_N = index[idx_test]
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

B_data = np.load('B.npz')
B_gauss_1 = B_data['B1']
B_gauss_2 = B_data['B2']
# B_gauss_1 = np.random.randn(500, 4)
# B_gauss_1 = B_gauss_1 * 1
# B_gauss_2 = np.random.randn(500, 3)
# B_gauss_2 = B_gauss_2 * 1

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
multiple = 2
# N_f = multiple * N_p
N_f = multiple * N_p_t
N_f_t = multiple * N_p
# N_f_in = multiple_in * N_p
# x_in_min = x_train.min(0)
# x_in_max = x_train.max(0)
# y_in_min = y_train.min(0)
# y_in_max = 0.15 / L
t_min = t_net.min(0)
t_max = t_net.max(0)


idx_cp = np.random.choice(x_f_train.shape[0], N_f_t, replace=False)
x_f_train = x_f_train[idx_cp]
y_f_train = y_f_train[idx_cp]
t0_f_train = t_min + (t_max - t_min) * lhs(1, N_f_t)




t_f_train = []
for _ in range(len(t0_f_train)):
    t0_value = t0_f_train[_]
    t_random = np.random.uniform(t0_value, t_max , size=50)
    t_f_train.append(t_random)

t_f_train = np.array(t_f_train).flatten()[:,None]
t0_f_train_extend = []
for element in t0_f_train:
    t0_f_train_extend.extend([element] * 50)

x_f_train_extend = []
for element in x_f_train:
    x_f_train_extend.extend([element] * 50)

y_f_train_extend = []
for element in y_f_train:
    y_f_train_extend.extend([element] * 50)

t0_f_train = np.array(t0_f_train_extend).flatten()[:,None]
x0_f_train = np.array(x_f_train_extend).flatten()[:,None]
y0_f_train = np.array(y_f_train_extend).flatten()[:,None]
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

##################################################################


xyt0_train = torch.tensor(xyt0_train).float()
xyuv_train = torch.tensor(xyuv_train).float()
B1 = B_gauss_1
B_gauss_1 = torch.tensor(B_gauss_1).float()
# B_gauss_2 = torch.tensor(B_gauss_2).float()
# train_data = addbatch(xyt0_train,xyuv_train, 500000)

model = TrajectoryNSNet(B_gauss_1)
#############################################
x0_test = x0_train
y0_test = y0_train
t_test = t_net
t0_test = t0_train
x_test = x_net
y_test = y_net



x_pred, y_pred = model.predict_x(x0_test, y0_test, t_test, t0_test, B1)

error_x = np.linalg.norm(x_test - x_pred, 2) / np.linalg.norm(x_test, 2)
error_y = np.linalg.norm(y_test - y_pred, 2) / np.linalg.norm(y_test, 2)
# error_u = np.linalg.norm(u_net - u_pred, 2) / np.linalg.norm(u_net, 2)
# error_v = np.linalg.norm(v_net - v_pred, 2) / np.linalg.norm(v_net, 2)
# error_p = np.linalg.norm(p_net - p_pred, 2) / np.linalg.norm(p_net, 2)

print('Error x: %e' % (error_x))
print('Error y: %e' % (error_y))
# print('Error u: %e' % (error_u))
# print('Error v: %e' % (error_v))
# print('Error p: %e' % (error_p))
x_test_pearsonr = x_test.flatten()
y_test_pearsonr = y_test.flatten()
x_pred_pearsonr = x_pred.flatten()
y_pred_pearsonr = y_pred.flatten()

coefficient_x_test ,_= stats.pearsonr(x_test_pearsonr,x_pred_pearsonr)
coefficient_y_test,_ = stats.pearsonr(y_test_pearsonr,y_pred_pearsonr)

print('coefficient x: %e' % (coefficient_x_test))
print('coefficient y: %e' % (coefficient_y_test))
############训练集############
fig = plt.figure(figsize=(12, 12), dpi=600)
ax = fig.add_subplot(111)

plt.scatter(x_test,y_test,s =1, marker="^",c='r')
plt.scatter(x_pred,y_pred,s =1, marker="^",c='b')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle test', fontsize = 20)
plt.axis('equal')
plt.show()


index_plot = np.cumsum(delta_index)
index_plot = np.append(0, index_plot)
index_plot = index_plot[:-1]



fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
# plt.tick_params(direction='in')
# vertices = [(0, 0), (0, 0.1/L), (0.5/L, 0.1/L), (0.5/L, 0.08/L), (0.34619/L, 0.08/L), (0.3/L, 0)]
# poly = Polygon(vertices, facecolor='white', edgecolor='black')
# ax.add_patch(poly)
# circle = plt.Circle((0.05/L, 0.05/L), 0.01/L, color='black', fill=False)
# plt.gca().add_patch(circle)

def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plot_test_index = np.random.choice(69, 69, replace=False)
# plot_test_index = plot_test_index[plot_test_index != 160]
# plot_test_index = plot_test_index[plot_test_index != 90]
# plot_test_index = plot_test_index[plot_test_index != 120]
# plot_test_index = plot_test_index[plot_test_index != 100]
for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_pred[index_plot[j]:index_plot[j]+delta_index[j]], y_pred[index_plot[j]:index_plot[j]+delta_index[j]],  color='#FAAF78', alpha=1.0,label='Predict',linewidth=1)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='#666666', alpha=0.7, label='Truth', linestyle='--',linewidth=1)
    plt.scatter(x0_test[index_plot[j]:index_plot[j]+delta_index[j]], y0_test[index_plot[j]:index_plot[j]+delta_index[j]], s=2, marker="o", c='red', label='Initial position',zorder=10)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),fontsize='small')


#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小
# ax.set_title('particle test', fontsize = 10)
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 0.2)
ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3_all_particals'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()
############################################################
# NE_x_line = abs(x_pred - x_test)
# NE_y_line = abs(y_pred - y_test)
# NE_x = np.hstack((t_test,NE_x_line))
# NE_y = np.hstack((t_test,NE_y_line))
NE_x_line_pred = x_pred
NE_y_line_pred = y_pred
NE_x_pred = np.hstack((t_test,NE_x_line_pred))
NE_y_pred = np.hstack((t_test,NE_y_line_pred))

NE_x_line_test = x_test
NE_y_line_test = y_test
NE_x_test = np.hstack((t_test,NE_x_line_test))
NE_y_test = np.hstack((t_test,NE_y_line_test))


def split_to_columns(data):
    """
    将 (n, 2) 的数据分割为多列，每列对应一个递增子序列（根据第一列判断），
    输出为 (max_row, num_cols+1) 的二维数组，第一列为排序后的唯一值，其余列用 NaN 填充空缺。
    """
    if data.shape[1] != 2:
        raise ValueError("Input data must be a 2D array with 2 columns.")

    # 分割数据到各个子序列，每个子序列保存两列的数据
    columns = []
    current_col = [[data[0, 0], data[0, 1]]]  # 初始化为第一个元素的两列

    for i in range(1, len(data)):
        if data[i, 0] > data[i - 1, 0]:
            current_col.append([data[i, 0], data[i, 1]])
        else:
            columns.append(np.array(current_col))
            current_col = [[data[i, 0], data[i, 1]]]

    # 添加最后一个子序列
    if current_col:
        columns.append(np.array(current_col))

    # 收集所有第一列的唯一值并排序
    sorted_unique = np.unique(data[:, 0])

    # 创建结果数组
    rows = len(sorted_unique)
    cols = len(columns) + 1  # 第一列 + 子序列数目的列
    result = np.full((rows, cols), np.nan)
    result[:, 0] = sorted_unique  # 第一列填充排序后的唯一值

    # 将每个子序列的数据填充到对应的列中
    for j, subseq in enumerate(columns):
        for x, y in subseq:
            # 找到x在sorted_unique中的行索引
            row_idx = np.where(sorted_unique == x)[0][0]
            result[row_idx, j + 1] = y  # j+1对应第j个子序列的列

    return result

def nan_norm(x, axis=None):
    return np.sqrt(np.nansum(x**2, axis=axis))



NE_x = split_to_columns(NE_x_pred)
NE_y = split_to_columns(NE_y_pred)
NT_x = split_to_columns(NE_x_test)
NT_y = split_to_columns(NE_y_test)

diff_x = NE_x - NT_x
norm_diff_x = nan_norm(diff_x,axis=1)
norm_test_x = nan_norm(NT_x, axis=1)
relative_errors_x = norm_diff_x / norm_test_x
relative_errors_x = relative_errors_x[:,None]

diff_y = NE_y - NT_y
norm_diff_y = nan_norm(diff_y,axis=1)
norm_test_y = nan_norm(NT_y, axis=1)
relative_errors_y = norm_diff_y / norm_test_x
relative_errors_y = relative_errors_y[:,None]

relative_errors_xy = np.hstack([relative_errors_x,relative_errors_y])
np.savetxt('3l2trajectory.csv', relative_errors_xy, fmt='%.18f', delimiter=',', newline='\n')

# NE_x = NE_x_line[index_plot[0]:index_plot[0]+delta_index[0]]
# NE_y = NE_y_line[index_plot[0]:index_plot[0]+delta_index[0]]
# for i in range(199):
#     i = i + 1
#     NE_x_i = NE_x_line[index_plot[i]:index_plot[i]+delta_index[i]]
#     NE_y_i = NE_y_line[index_plot[i]:index_plot[i] + delta_index[i]]
#     NE_x = np.hstack((NE_x, NE_x_i))
#     NE_y = np.hstack((NE_y, NE_y_i))
# np.savetxt('2NE_x_line.csv', NE_x, fmt='%.18f', delimiter=',', newline='\n')
# np.savetxt('2NE_y_line.csv', NE_y, fmt='%.18f', delimiter=',', newline='\n')
############################################################















x_plot_true = []
y_plot_true = []
x_plot_pred = []
y_plot_pred = []
for i in range(len(t_test)):
    if t_test[i] == 0.005 * 2 * (U / L):
        x_plot_true.append(x_test[i])
        y_plot_true.append(y_test[i])
        x_plot_pred.append(x_pred[i])
        y_plot_pred.append(y_pred[i])
NE_x_2 = abs(np.array(x_plot_pred) - np.array(x_plot_true)) / 1
NE_y_2 = abs(np.array(y_plot_pred) - np.array(y_plot_true)) / 1
np.savetxt('3NE_x_2.csv', NE_x_2, fmt='%.18f', delimiter=',', newline='\n')
np.savetxt('3NE_y_2.csv', NE_y_2, fmt='%.18f', delimiter=',', newline='\n')
colors = np.random.rand(len(x_plot_true),3)
# fig = plt.figure(figsize=(9, 6), dpi=300)
# ax = fig.add_subplot(111)
# plt.scatter(x_plot_true,y_plot_true,s =15, marker="o",c=colors, rasterized=True)
# ax.set_xlabel('$x$', size=20)
# ax.set_ylabel('$y$', size=20)
# ax.set_title('particle test t=2 true', fontsize = 20)
# # plt.legend()
# ax.tick_params(labelsize=15)
# plt.axis('equal')
# fig.savefig('particle_test_t=2_true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()
# fig = plt.figure(figsize=(9, 6), dpi=300)
# ax = fig.add_subplot(111)
# plt.scatter(x_plot_pred,y_plot_pred,s =15, marker="o",c=colors, rasterized=True)
# ax.set_xlabel('$x$', size=20)
# ax.set_ylabel('$y$', size=20)
# ax.set_title('particle test t=2 pred', fontsize = 20)
# # plt.legend()
# ax.tick_params(labelsize=15)
# plt.axis('equal')
# fig.savefig('particle_test_t=2_pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()



fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plot_test_index = np.random.choice(69, 69, replace=False)

for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)
    plt.scatter(x_plot_true, y_plot_true, s=5, marker="o", c=colors, rasterized=True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小
ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test t=2 true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plot_test_index = np.random.choice(69, 69, replace=False)

for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)
    plt.scatter(x_plot_pred, y_plot_pred, s=5, marker="o", c=colors, rasterized=True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小
ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test t=2 pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


x_plot_true = []
y_plot_true = []
x_plot_pred = []
y_plot_pred = []
for i in range(len(t_test)):
    if t_test[i] == 0.005 * 25 * (U / L):
        x_plot_true.append(x_test[i])
        y_plot_true.append(y_test[i])
        x_plot_pred.append(x_pred[i])
        y_plot_pred.append(y_pred[i])
NE_x_25 = abs(np.array(x_plot_pred) - np.array(x_plot_true)) / 1
NE_y_25 = abs(np.array(y_plot_pred) - np.array(y_plot_true)) / 1
np.savetxt('3NE_x_25.csv', NE_x_25, fmt='%.18f', delimiter=',', newline='\n')
np.savetxt('3NE_y_25.csv', NE_y_25, fmt='%.18f', delimiter=',', newline='\n')
colors = np.random.rand(len(x_plot_true),3)
# fig = plt.figure(figsize=(9, 6), dpi=300)
# ax = fig.add_subplot(111)
# plt.scatter(x_plot_true,y_plot_true,s =15, marker="o",c=colors, rasterized=True)
# ax.set_xlabel('$x$', size=20)
# ax.set_ylabel('$y$', size=20)
# ax.set_title('particle test t=2 true', fontsize = 20)
# # plt.legend()
# ax.tick_params(labelsize=15)
# plt.axis('equal')
# fig.savefig('particle_test_t=2_true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()
# fig = plt.figure(figsize=(9, 6), dpi=300)
# ax = fig.add_subplot(111)
# plt.scatter(x_plot_pred,y_plot_pred,s =15, marker="o",c=colors, rasterized=True)
# ax.set_xlabel('$x$', size=20)
# ax.set_ylabel('$y$', size=20)
# ax.set_title('particle test t=2 pred', fontsize = 20)
# # plt.legend()
# ax.tick_params(labelsize=15)
# plt.axis('equal')
# fig.savefig('particle_test_t=2_pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()



fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plot_test_index = np.random.choice(69, 69, replace=False)

for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)
    plt.scatter(x_plot_true, y_plot_true, s=5, marker="o", c=colors, rasterized=True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小
ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test t=25 true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plot_test_index = np.random.choice(69, 69, replace=False)

for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)
    plt.scatter(x_plot_pred, y_plot_pred, s=5, marker="o", c=colors, rasterized=True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小
ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test t=25 pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

x_plot_true = []
y_plot_true = []
x_plot_pred = []
y_plot_pred = []
for i in range(len(t_test)):
    if t_test[i] == 0.005 * 50 * (U / L):
        x_plot_true.append(x_test[i])
        y_plot_true.append(y_test[i])
        x_plot_pred.append(x_pred[i])
        y_plot_pred.append(y_pred[i])
NE_x_50 = abs(np.array(x_plot_pred) - np.array(x_plot_true)) / 1
NE_y_50 = abs(np.array(y_plot_pred) - np.array(y_plot_true)) / 1
np.savetxt('3NE_x_50.csv', NE_x_50, fmt='%.18f', delimiter=',', newline='\n')
np.savetxt('3NE_y_50.csv', NE_y_50, fmt='%.18f', delimiter=',', newline='\n')
colors = np.random.rand(len(x_plot_true),3)
# fig = plt.figure(figsize=(9, 6), dpi=300)
# ax = fig.add_subplot(111)
# plt.scatter(x_plot_true,y_plot_true,s =15, marker="o",c=colors, rasterized=True)
# ax.set_xlabel('$x$', size=20)
# ax.set_ylabel('$y$', size=20)
# ax.set_title('particle test t=2 true', fontsize = 20)
# # plt.legend()
# ax.tick_params(labelsize=15)
# plt.axis('equal')
# fig.savefig('particle_test_t=2_true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()
# fig = plt.figure(figsize=(9, 6), dpi=300)
# ax = fig.add_subplot(111)
# plt.scatter(x_plot_pred,y_plot_pred,s =15, marker="o",c=colors, rasterized=True)
# ax.set_xlabel('$x$', size=20)
# ax.set_ylabel('$y$', size=20)
# ax.set_title('particle test t=2 pred', fontsize = 20)
# # plt.legend()
# ax.tick_params(labelsize=15)
# plt.axis('equal')
# fig.savefig('particle_test_t=2_pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()



fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plot_test_index = np.random.choice(69, 69, replace=False)

for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)
    plt.scatter(x_plot_true, y_plot_true, s=5, marker="o", c=colors, rasterized=True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小
ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test t=50 true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plot_test_index = np.random.choice(69, 69, replace=False)

for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)
    plt.scatter(x_plot_pred, y_plot_pred, s=5, marker="o", c=colors, rasterized=True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小
ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test t=50 pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

x_plot_true = []
y_plot_true = []
x_plot_pred = []
y_plot_pred = []
for i in range(len(t_test)):
    if t_test[i] == 0.005 * 40 * (U / L):
        x_plot_true.append(x_test[i])
        y_plot_true.append(y_test[i])
        x_plot_pred.append(x_pred[i])
        y_plot_pred.append(y_pred[i])
NE_x_40 = abs(np.array(x_plot_pred) - np.array(x_plot_true)) / 1
NE_y_40 = abs(np.array(y_plot_pred) - np.array(y_plot_true)) / 1
np.savetxt('3NE_x_40.csv', NE_x_40, fmt='%.18f', delimiter=',', newline='\n')
np.savetxt('3NE_y_40.csv', NE_y_40, fmt='%.18f', delimiter=',', newline='\n')
colors = np.random.rand(len(x_plot_true),3)
# fig = plt.figure(figsize=(9, 6), dpi=300)
# ax = fig.add_subplot(111)
# plt.scatter(x_plot_true,y_plot_true,s =15, marker="o",c=colors, rasterized=True)
# ax.set_xlabel('$x$', size=20)
# ax.set_ylabel('$y$', size=20)
# ax.set_title('particle test t=2 true', fontsize = 20)
# # plt.legend()
# ax.tick_params(labelsize=15)
# plt.axis('equal')
# fig.savefig('particle_test_t=2_true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()
# fig = plt.figure(figsize=(9, 6), dpi=300)
# ax = fig.add_subplot(111)
# plt.scatter(x_plot_pred,y_plot_pred,s =15, marker="o",c=colors, rasterized=True)
# ax.set_xlabel('$x$', size=20)
# ax.set_ylabel('$y$', size=20)
# ax.set_title('particle test t=2 pred', fontsize = 20)
# # plt.legend()
# ax.tick_params(labelsize=15)
# plt.axis('equal')
# fig.savefig('particle_test_t=2_pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()



fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plot_test_index = np.random.choice(69, 69, replace=False)

for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)
    plt.scatter(x_plot_true, y_plot_true, s=5, marker="o", c=colors, rasterized=True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小
ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test t=40 true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plot_test_index = np.random.choice(69, 69, replace=False)

for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)
    plt.scatter(x_plot_pred, y_plot_pred, s=5, marker="o", c=colors, rasterized=True)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小
ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test t=40 pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()
##############################################################################################
delta_index = delta_index.tolist()
i = 7
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
x_plot_true = x_test[plot_number:plot_number + delta_index[i]]
y_plot_true = y_test[plot_number:plot_number + delta_index[i]]
x_plot_pred = x_pred[plot_number:plot_number + delta_index[i]]
y_plot_pred = y_pred[plot_number:plot_number + delta_index[i]]

NE_x_7 = abs(np.array(x_plot_pred) - np.array(x_plot_true)) / 1
NE_y_7 = abs(np.array(y_plot_pred) - np.array(y_plot_true)) / 1
np.savetxt('3NE_x_7.csv', NE_x_7, fmt='%.18f', delimiter=',', newline='\n')
np.savetxt('3NE_y_7.csv', NE_y_7, fmt='%.18f', delimiter=',', newline='\n')

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plt.scatter(x_plot_true, y_plot_true,s=2, marker="x", c='r', label='True')
plot_test_index = np.random.choice(69, 69, replace=False)
for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小

ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test 7 true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
plt.scatter(x_plot_pred,y_plot_pred,s =2, marker="^",c='b', label='predict')
plot_test_index = np.random.choice(69, 69, replace=False)
for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小

ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test 7 pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


i = 35
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
x_plot_true = x_test[plot_number:plot_number + delta_index[i]]
y_plot_true = y_test[plot_number:plot_number + delta_index[i]]
x_plot_pred = x_pred[plot_number:plot_number + delta_index[i]]
y_plot_pred = y_pred[plot_number:plot_number + delta_index[i]]

NE_x_35 = abs(np.array(x_plot_pred) - np.array(x_plot_true)) / 1
NE_y_35 = abs(np.array(y_plot_pred) - np.array(y_plot_true)) / 1
np.savetxt('3NE_x_35.csv', NE_x_35, fmt='%.18f', delimiter=',', newline='\n')
np.savetxt('3NE_y_35.csv', NE_y_35, fmt='%.18f', delimiter=',', newline='\n')

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plt.scatter(x_plot_true, y_plot_true,s=2, marker="x", c='r', label='True')
plot_test_index = np.random.choice(69, 69, replace=False)
for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小

ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test 35 true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
plt.scatter(x_plot_pred,y_plot_pred,s =2, marker="^",c='b', label='predict')
plot_test_index = np.random.choice(69, 69, replace=False)
for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小

ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test 35 pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()



i = 19
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
x_plot_true = x_test[plot_number:plot_number + delta_index[i]]
y_plot_true = y_test[plot_number:plot_number + delta_index[i]]
x_plot_pred = x_pred[plot_number:plot_number + delta_index[i]]
y_plot_pred = y_pred[plot_number:plot_number + delta_index[i]]

NE_x_19 = abs(np.array(x_plot_pred) - np.array(x_plot_true)) / 1
NE_y_19 = abs(np.array(y_plot_pred) - np.array(y_plot_true)) / 1
np.savetxt('3NE_x_19.csv', NE_x_19, fmt='%.18f', delimiter=',', newline='\n')
np.savetxt('3NE_y_19.csv', NE_y_19, fmt='%.18f', delimiter=',', newline='\n')

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plt.scatter(x_plot_true, y_plot_true,s=2, marker="x", c='r', label='True')
plot_test_index = np.random.choice(69, 69, replace=False)
for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小

ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test 19 true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
plt.scatter(x_plot_pred,y_plot_pred,s =2, marker="^",c='b', label='predict')
plot_test_index = np.random.choice(69, 69, replace=False)
for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小

ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test 19 pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


i = 63
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
x_plot_true = x_test[plot_number:plot_number + delta_index[i]]
y_plot_true = y_test[plot_number:plot_number + delta_index[i]]
x_plot_pred = x_pred[plot_number:plot_number + delta_index[i]]
y_plot_pred = y_pred[plot_number:plot_number + delta_index[i]]

NE_x_63 = abs(np.array(x_plot_pred) - np.array(x_plot_true)) / 1
NE_y_63 = abs(np.array(y_plot_pred) - np.array(y_plot_true)) / 1
np.savetxt('3NE_x_63.csv', NE_x_63, fmt='%.18f', delimiter=',', newline='\n')
np.savetxt('3NE_y_63.csv', NE_y_63, fmt='%.18f', delimiter=',', newline='\n')

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
def annotate_point(label, x, y, xytext_offset):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=xytext_offset,
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle="arc3,rad=0",  # 确保有延长线
            shrinkA=0,                # 调整起点收缩距离
            shrinkB=0                 # 调整终点收缩距离
        ),
        fontsize=8,
    )
plt.scatter(x_plot_true, y_plot_true,s=2, marker="x", c='r', label='True')
plot_test_index = np.random.choice(69, 69, replace=False)
for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小

ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test 63 true'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
plt.scatter(x_plot_pred,y_plot_pred,s =2, marker="^",c='b', label='predict')
plot_test_index = np.random.choice(69, 69, replace=False)
for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='gray', alpha=0.7, label='Truth', linestyle='--',linewidth=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(),fontsize='small')
#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=5)
ax.set_ylabel('$y$', size=5)
plt.tick_params(axis='x', labelsize=5)  # 调整x轴刻度线字体大小
plt.tick_params(axis='y', labelsize=5)  # 调整y轴刻度线字体大小

ax.set_aspect('equal')
plt.rcParams['svg.fonttype'] = 'none'
fig.savefig('3particle test 63 pred'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


###############################################################################################
delta_index = delta_index.tolist()
fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 674
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
plt.scatter(x_test,y_test,s =1, marker="^",c='y')
plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =10, marker="x",c='r',label='True')
plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =10, marker="^",c='b', label='predict')
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle test 674', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
#fig.savefig('single_partical'  + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 420
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
plt.scatter(x_test,y_test,s =1, marker="^",c='y')
plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =10, marker="x",c='r',label='True')
plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =10, marker="^",c='b', label='predict')
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle test 420', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
#fig.savefig('single_partical'  + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 552
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
plt.scatter(x_test,y_test,s =1, marker="^",c='y')
plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =10, marker="x",c='r',label='True')
plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =10, marker="^",c='b', label='predict')
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle test 552', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
#fig.savefig('single_partical'  + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()
fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 552
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =10, marker="x",c='r',label='True')
plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =10, marker="^",c='b', label='predict')
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle test 552', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
#fig.savefig('single_partical'  + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 1044
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
plt.scatter(x_test,y_test,s =1, marker="^",c='y')
plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =10, marker="x",c='r',label='True')
plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =10, marker="^",c='b', label='predict')
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle test 1044', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
#fig.savefig('single_partical'  + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 61
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
plt.scatter(x_test,y_test,s =1, marker="^",c='y')
plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =10, marker="x",c='r',label='True')
plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =10, marker="^",c='b', label='predict')
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle test 61', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
#fig.savefig('single_partical'  + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 28
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
plt.scatter(x_test,y_test,s =1, marker="^",c='y')
plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =10, marker="x",c='r',label='True')
plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =10, marker="^",c='b', label='predict')
ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle test 38', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
#fig.savefig('single_partical'  + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()


fig = plt.figure(figsize=(10,10),dpi = 330 )
ax = fig.add_subplot(111, projection='3d')

sc1 = ax.scatter(x_test[plot_number:plot_number + delta_index[i]],t_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]] ,s =2, marker="o",c='r')
# 为了在图中显示颜色条，我们需要创建一个ScalarMappable对象
sc2 = ax.scatter(x_pred[plot_number:plot_number + delta_index[i]],t_test[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]] ,s =2, marker="o",c='b')

# 将归一化后的刻度位置转换为原始的z值，并设置颜色条的刻度标签

plt.show()
# ########################################################
# fig = plt.figure()
# ax1= fig.add_subplot(121)
# ax2= fig.add_subplot(122)
#
# filename_test = "F:/code/pycharm/reproductionPINN/wangge.csv"
#
# #train_data = np.genfromtxt(filename, delimiter=',', skip_header=True)
# wangge = pd.read_csv(filename_test, header=0)
# wangge = np.array(wangge)
#
# #print('从fluent导出的训练数据为：')
# wangge = np.delete(wangge, 0, axis=1)
# # train_data = np.delete(train_data, 0, axis=0)
# wangge = wangge.astype(float)
# np.set_printoptions(suppress=True)
#
#
# # wangge = np.around(wangge,3)
# wangge_u = np.array(wangge[:,4:5]).flatten()[:, None]
# wangge_x = np.array(wangge[:,0:1]).flatten()[:, None]
# wangge_y = np.array(wangge[:,1:2]).flatten()[:, None]
# wangge_p = np.array(wangge[:,2:3]).flatten()[:, None]
# wangge_v = np.array(wangge[:,5:6]).flatten()[:, None]
# X_true_star = np.hstack((wangge_x, wangge_y))
# wangge_x = np.around(wangge_x,2)
# wangge_y = np.around(wangge_y,2)
# # print(wangge_x)
# np.set_printoptions(precision=7)
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
# # print(X_star)
# t_plot = []
# for _ in range(10201):
#     t_plot.append(t_net[10])
# t_plot = np.array(t_plot).flatten()[:, None]
#
# x_test_pred, y_test_pred, u_test_pred, v_test_pred, p_test_pred = model.predict(X.flatten()[:, None], Y.flatten()[:, None], t_plot)
#
# U_true = griddata(X_true_star , wangge_u, (X, Y), method='cubic')
# U = griddata(X_star, u_test_pred.flatten(), (X, Y), method='cubic')
# h_U = ax2.imshow(U, interpolation='nearest', cmap='rainbow',
#               extent=[wangge_x.min(), wangge_x.max(), wangge_y.min(), wangge_y.max()],
#               origin='lower', aspect='equal')
# h_U_true = ax1.imshow(U_true, interpolation='nearest', cmap='rainbow',
#               extent=[wangge_x.min(), wangge_x.max(), wangge_y.min(), wangge_y.max()],
#               origin='lower', aspect='equal')
#
#
# ax1.set_xlabel('$x$', size=10)
# ax2.set_xlabel('$x$', size=10)
# ax1.set_ylabel('$y$', size=10)
# ax2.set_ylabel('$y$', size=10)
# ax1.set_title('u true t=50', fontsize = 10)
# ax2.set_title('u predict t=50', fontsize = 10)
# ax.tick_params(labelsize=15)
# fig.subplots_adjust(wspace=0.5)
# fig.colorbar(h_U_true, ax=[ax1,ax2], fraction=0.03, shrink=0.5)
# #fig.savefig('single_partical'  + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()
# fig = plt.figure()
# ax1= fig.add_subplot(121)
# ax2= fig.add_subplot(122)
#
# V_true = griddata(X_true_star , wangge_v,(X,Y),method='cubic')
# V = griddata(X_star, v_test_pred.flatten(), (X, Y), method='cubic')
# h_V = ax2.imshow(V, interpolation='nearest', cmap='rainbow',
#               extent=[wangge_x.min(), wangge_x.max(), wangge_y.min(), wangge_y.max()],
#               origin='lower', aspect='equal')
# h_V_true = ax1.imshow(V_true, interpolation='nearest', cmap='rainbow',
#               extent=[wangge_x.min(), wangge_x.max(), wangge_y.min(), wangge_y.max()],
#               origin='lower', aspect='equal')
#
#
# ax1.set_xlabel('$x$', size=10)
# ax2.set_xlabel('$x$', size=10)
# ax1.set_ylabel('$y$', size=10)
# ax2.set_ylabel('$y$', size=10)
# ax1.set_title('v true t=50', fontsize = 10)
# ax2.set_title('v predict t=50', fontsize = 10)
# ax.tick_params(labelsize=15)
# fig.subplots_adjust(wspace=0.5)
# fig.colorbar(h_V_true, ax=[ax1,ax2], fraction=0.03, shrink=0.5)
# #fig.savefig('single_partical'  + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
# plt.show()
#
# fig = plt.figure()
# ax1= fig.add_subplot(121)
# ax2= fig.add_subplot(122)
#
#
# P = griddata(X_star,p_test_pred.flatten(), (X, Y), method='cubic')
# P_true = griddata(X_true_star ,wangge_p, (X, Y), method='cubic')
# h_P = ax2.imshow(P, interpolation='nearest', cmap='rainbow',
#               extent=[wangge_x.min(), wangge_x.max(), wangge_y.min(), wangge_y.max()],
#               origin='lower', aspect='equal')
# h_P_true = ax1.imshow(P_true, interpolation='nearest', cmap='rainbow',
#               extent=[wangge_x.min(), wangge_x.max(), wangge_y.min(), wangge_y.max()],
#               origin='lower', aspect='equal')
#
#
# ax1.set_xlabel('$x$', size=10)
# ax2.set_xlabel('$x$', size=10)
# ax1.set_ylabel('$y$', size=10)
# ax2.set_ylabel('$y$', size=10)
# ax1.set_title('p true t=50', fontsize = 10)
# ax2.set_title('p predict t=50', fontsize = 10)
# ax.tick_params(labelsize=15)
# fig.subplots_adjust(wspace=0.5)
# #fig.savefig('single_partical'  + ".png", bbox_inches='tight', dpi=600, pad_inches=0.1)
# fig.colorbar(h_P_true, ax=[ax1,ax2], fraction=0.03, shrink=0.5)
# plt.show()