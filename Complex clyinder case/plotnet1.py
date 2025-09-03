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
import scipy.stats as stats
from onnx import shape_inference
from more_itertools import flatten
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.animation as animation
from matplotlib.patches import Polygon
os.environ['CUDA_VISIBLE_DEVICES']='3'
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


        self.net_1.load_state_dict(torch.load("suanli2nopretrainnet1.pth"))

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


B_data = np.load('B.npz')
B1_data = np.load('suanli2B1.npz')
B_gauss_1 = B1_data['suanli2B1']
B_gauss_2 = B_data['B2']
layers_1 = [5000 * 2, 40, 40, 40, 40,40, 40, 40, 40,40, 40, 2]
layers_2 = [500 * 2, 60, 60, 60, 60, 60, 60, 2]
# layers_2 = [3, 200, 200, 200, 200, 200, 200, 200, 200, 2]
# Load Data
filename = "shiyanparticle0.20611.csv"

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
p = train_data[:, 4:5]
u = train_data[:, 4:5]
v = train_data[:, 5:6]



t_start = 24.5
t_end = 25.5
t_step = (t_end-t_start)/0.02
extrater = []
for i in range(len(x)):
    if t[i]<t_start or t[i]>t_end:
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
# for i in index:
#     print(id[i])
# x = [i-0.08 for i in x]
# y = [i+0.02 for i in y]
# u_min = min(abs(u))
# v_min = min(abs(v))
# uv_min = float(min(u_min, v_min))


t = [i-24.5 for i in t]
t = np.round(t,2)
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

# U = 1
# L = 0.01
x_train = x / L
y_train = y / L
t_train = t * (U / L)
u_train = u / U
v_train = v / U
p_train = p / (998.2 * (U ** 2))


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
p_net =[]
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

# k = 0
# for i in index:
index = np.array(index)
N_p = 2000
N_p_test = 200
idx_p = np.random.choice(index.shape[0]-1, N_p, replace=False)
idx_test = np.random.choice(int(np.setdiff1d(index.shape[0]-1, idx_p)), N_p_test, replace=False)
idx_p1 = [i+1 for i in idx_test]
index_N = index[idx_test]
delta_index = index[idx_p1] - index_N
for i in range(int(len(index_N))):
    x0_train.append(x00_train[index_N[i]:index_N[i] + delta_index[i]])
    y0_train.append(y00_train[index_N[i]:index_N[i] + delta_index[i]])
    t0_train.append(t00_train[index_N[i]:index_N[i] + delta_index[i]])
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
x0_test = list(flatten(x0_train))
y0_test = list(flatten(y0_train))
t0_test = list(flatten(t0_train))
x_test = list(flatten(x_net))
y_test = list(flatten(y_net))
u_test = list(flatten(u_net))
v_test = list(flatten(v_net))
p_test = list(flatten(p_net))
t_test = list(flatten(t_net))

x0_test = np.array(x0_test).flatten()[:, None]
y0_test = np.array(y0_test).flatten()[:, None]
t0_test = np.array(t0_test).flatten()[:, None]
x_test = np.array(x_test).flatten()[:, None]
y_test = np.array(y_test).flatten()[:, None]
t_test = np.array(t_test).flatten()[:, None]
u_test = np.array(u_test).flatten()[:, None]
v_test = np.array(v_test).flatten()[:, None]
p_test = np.array(p_test).flatten()[:, None]

multiple=3
# xyt0_train = torch.tensor(xyt0_train).float()
# xyuv_train = torch.tensor(xyuv_train).float()
B1 = B_gauss_1
B_gauss_1 = torch.tensor(B_gauss_1).float()
# B_gauss_2 = torch.tensor(B_gauss_2).float()
# train_data = addbatch(xyt0_train,xyuv_train, 500000)

model = TrajectoryNSNet(B_gauss_1)
#############################################

xyt0_train = np.hstack([x_test, y_test, t_test, t0_test])
xyuv_train = np.hstack([x_test, y_test, u_test, v_test, p_test])
# xyuv_test = np.hstack([x0_test,y0_test,t_test])
xyt0_train = torch.tensor(xyt0_train).float()
xyuv_train = torch.tensor(xyuv_train).float()
# xyuv_test = torch.tensor(xyuv_test).float()
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
index_plot = np.cumsum(delta_index)
index_plot = np.append(0, index_plot)
index_plot = index_plot[:-1]
fig = plt.figure(figsize=(6, 3), dpi=300)
ax = fig.add_subplot(111)
plt.tick_params(direction='in')
vertices = [(0, 0), (0, 0.1/L), (0.5/L, 0.1/L), (0.5/L, 0.08/L), (0.34619/L, 0.08/L), (0.3/L, 0)]
poly = Polygon(vertices, facecolor='white', edgecolor='black')
ax.add_patch(poly)
circle = plt.Circle((0.05/L, 0.05/L), 0.01/L, color='black', fill=False)
plt.gca().add_patch(circle)

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
plot_test_index = np.random.choice(200, 44, replace=False)
plot_test_index = plot_test_index[plot_test_index != 160]
plot_test_index = plot_test_index[plot_test_index != 90]
plot_test_index = plot_test_index[plot_test_index != 120]
plot_test_index = plot_test_index[plot_test_index != 100]
for i in range(int(len(plot_test_index))):

    j=plot_test_index[i]
    print(j)
    plt.plot(x_pred[index_plot[j]:index_plot[j]+delta_index[j]], y_pred[index_plot[j]:index_plot[j]+delta_index[j]],  color='#FAAF78', alpha=1.0,label='Predict')
    plt.plot(x_test[index_plot[j]:index_plot[j]+delta_index[j]], y_test[index_plot[j]:index_plot[j]+delta_index[j]],
             color='#666666', alpha=0.7, label='Truth', linestyle='--')
    plt.scatter(x0_test[index_plot[j]:index_plot[j]+delta_index[j]], y0_test[index_plot[j]:index_plot[j]+delta_index[j]], s=20, marker="o", c='red', label='Initial position',zorder=10)
    # plt.annotate(j,xy=(x0_test[index_plot[j]],y0_test[index_plot[j]]),xytext=(x0_test[index_plot[j]]+0.1,y0_test[index_plot[j]]+0.1),arrowprops=dict(arrowstyle='->'))
# plt.scatter(x0_test, y0_test, s=20, marker="o", c='red', label='Initial position',zorder=10)
# plt.annotate(155,xy=(x_test[index_plot[155]],y_test[index_plot[155]]),xytext=(x_test[index_plot[155]]-0.011,y_test[index_plot[155]]+0.021),arrowprops=dict(arrowstyle='->'),fontsize=8)
# plt.annotate(12,xy=(x_test[index_plot[12]],y_test[index_plot[12]]),xytext=(x_test[index_plot[12]]+0.019,y_test[index_plot[12]]+0.03),arrowprops=dict(arrowstyle='->'),fontsize=8)
# plt.annotate(50,xy=(x_test[index_plot[50]],y_test[index_plot[50]]),xytext=(x_test[index_plot[50]]-0.03,y_test[index_plot[50]]+0.035),arrowprops=dict(arrowstyle='->'),fontsize=8)
annotate_point(50, x_test[index_plot[50]], y_test[index_plot[50]],
               (x_test[index_plot[50]]-0.039, y_test[index_plot[50]]+0.03))
annotate_point(12, x_test[index_plot[12]], y_test[index_plot[12]],
               (x_test[index_plot[12]]+0.02, y_test[index_plot[12]]+0.03))
annotate_point(155, x_test[index_plot[155]], y_test[index_plot[155]],
               (x_test[index_plot[155]]-0.048, y_test[index_plot[155]]-0.08))
annotate_point(71, x_test[index_plot[71]], y_test[index_plot[71]],
               (x_test[index_plot[71]]-0.058, y_test[index_plot[71]]-0.04))
# annotate_point(50, x_test[index_plot[50]], y_test[index_plot[50]],
#                (x_test[index_plot[50]]-0.04, y_test[index_plot[40]]+0.039))
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),fontsize='small')


#plt.scatter(x0_f_train,y0_f_train,s =0.2, marker="x",c='k')
ax.set_xlabel('$x$', size=10)
ax.set_ylabel('$y$', size=10)
# ax.set_title('particle test', fontsize = 10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.2)
ax.set_aspect('equal')
fig.savefig('2_all_particals'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()
#####################################################################################
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
np.savetxt('2l2trajectory.csv', relative_errors_xy, fmt='%.18f', delimiter=',', newline='\n')

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

#####################155 12 50#######################################################
step = 3
fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 155
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]
plt.scatter(x0_test[plot_number:plot_number + delta_index[i]], y0_test[plot_number:plot_number + delta_index[i]], s=20, marker="o", c='red', label='Initial position',zorder=10)
plt.plot(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]], color='#FAAF78',linewidth=2, alpha=1.0,label='Predict')
plt.plot(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],color='#666666',linewidth=2, alpha=0.7, label='Truth', linestyle='--')
# plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =0.1, marker="x",c='#666666')

# plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =0.1, marker="^",color='#FAAF78')


sparse_indices = slice(None, None, step)  # 创建一个切片对象用于稀疏化

plt.scatter(x_test[plot_number:plot_number + delta_index[i]][sparse_indices],
            y_test[plot_number:plot_number + delta_index[i]][sparse_indices],
            s=20, marker="^", c='#666666')

plt.scatter(x_pred[plot_number:plot_number + delta_index[i]][sparse_indices],
            y_pred[plot_number:plot_number + delta_index[i]][sparse_indices],
            s=20, marker="o", color='#FAAF78')



ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle 155', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
fig.savefig('2_partical_155'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 12
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]

plt.scatter(x0_test[plot_number:plot_number + delta_index[i]], y0_test[plot_number:plot_number + delta_index[i]], s=20, marker="o", c='red', label='Initial position',zorder=10)
plt.plot(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]], color='#FAAF78',linewidth=2, alpha=1.0,label='Predict')
plt.plot(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],color='#666666',linewidth=2, alpha=0.7, label='Truth', linestyle='--')
# plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =0.1, marker="x",c='#666666')

# plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =0.1, marker="^",color='#FAAF78')


plt.scatter(x_test[plot_number:plot_number + delta_index[i]][sparse_indices],
            y_test[plot_number:plot_number + delta_index[i]][sparse_indices],
            s=20, marker="^", c='#666666')

plt.scatter(x_pred[plot_number:plot_number + delta_index[i]][sparse_indices],
            y_pred[plot_number:plot_number + delta_index[i]][sparse_indices],
            s=20, marker="o", color='#FAAF78')

ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle 12', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
fig.savefig('2_partical_12'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()



fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 50
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]

plt.scatter(x0_test[plot_number:plot_number + delta_index[i]], y0_test[plot_number:plot_number + delta_index[i]], s=20, marker="o", c='red', label='Initial position',zorder=10)
plt.plot(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]], color='#FAAF78',linewidth=2, alpha=1.0,label='Predict')
plt.plot(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],color='#666666',linewidth=2, alpha=0.7, label='Truth', linestyle='--')
# plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =0.1, marker="x",c='#666666')

# plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =0.1, marker="^",color='#FAAF78')


plt.scatter(x_test[plot_number:plot_number + delta_index[i]][sparse_indices],
            y_test[plot_number:plot_number + delta_index[i]][sparse_indices],
            s=20, marker="^", c='#666666')

plt.scatter(x_pred[plot_number:plot_number + delta_index[i]][sparse_indices],
            y_pred[plot_number:plot_number + delta_index[i]][sparse_indices],
            s=20, marker="o", color='#FAAF78')

ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle 50', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
fig.savefig('2_partical_50'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

fig = plt.figure(figsize=(9, 6), dpi=300)
ax = fig.add_subplot(111)
i = 71
plot_number = 0
for j in range(int(len(delta_index))):
    if j < i:
        plot_number = plot_number + delta_index[j]

plt.scatter(x0_test[plot_number:plot_number + delta_index[i]], y0_test[plot_number:plot_number + delta_index[i]], s=20, marker="o", c='red', label='Initial position',zorder=10)
plt.plot(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]], color='#FAAF78',linewidth=2, alpha=1.0,label='Predict')
plt.plot(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],color='#666666',linewidth=2, alpha=0.7, label='Truth', linestyle='--')
# plt.scatter(x_test[plot_number:plot_number + delta_index[i]],y_test[plot_number:plot_number + delta_index[i]],s =0.1, marker="x",c='#666666')

# plt.scatter(x_pred[plot_number:plot_number + delta_index[i]],y_pred[plot_number:plot_number + delta_index[i]],s =0.1, marker="^",color='#FAAF78')


plt.scatter(x_test[plot_number:plot_number + delta_index[i]][sparse_indices],
            y_test[plot_number:plot_number + delta_index[i]][sparse_indices],
            s=20, marker="^", c='#666666')

plt.scatter(x_pred[plot_number:plot_number + delta_index[i]][sparse_indices],
            y_pred[plot_number:plot_number + delta_index[i]][sparse_indices],
            s=20, marker="o", color='#FAAF78')

ax.set_xlabel('$x$', size=20)
ax.set_ylabel('$y$', size=20)
ax.set_title('particle 71', fontsize = 20)
plt.legend()
ax.tick_params(labelsize=15)
plt.axis('equal')
fig.savefig('2_partical_71'  + ".svg", bbox_inches='tight', dpi=600, pad_inches=0.1)
plt.show()

