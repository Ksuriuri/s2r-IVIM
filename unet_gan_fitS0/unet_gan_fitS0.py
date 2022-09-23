import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
import torch.optim as optim
from os.path import exists, join
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from ops import *
from glob import glob
import copy

img_w = 192  # 168 160 32
img_h = 256  # 210 192 32
num_epochs = 80  # 80, 160
batch_size = 4  # 128
patience = num_epochs  # 100
repeat_num = 1  # 10
learning_rate_g = 1e-4  # 1e-5
learning_rate_d = 1e-4
train_num = 95  # 73
step = 4
slice = 0 * step
slice_num = 8 * step

data_path = r'/home/public/Documents/hhy/data/IVIM6-1/real_new3'  # real_new2, real_noise50
syn_path = r'/home/public/Documents/hhy/data/IVIM6-1/syn_data_n3.npz'
save_path = r'/home/public/Documents/hhy/ivim_new/s2r_result_new_plus/unet_gan_80_4_0_9_100_2_fitS0'
if not exists(save_path):
    os.makedirs(save_path)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# define b values
b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000]).astype(np.float32)
b_fit = np.expand_dims(b_values, -1)  # .repeat(batch_size, axis=0)
b_fit = Variable(torch.from_numpy(b_fit).to(device))
print(b_values.shape)

b_files = glob(join(data_path, '*'))
b_files.sort(key=lambda x: (int(x.split('/')[-1].split('.')[0])))
# print(len(b_files))
np.random.seed(2022)
np.random.shuffle(b_files)

dp_min, dp_max = 0.005, 0.3
dt_min, dt_max = 0.0005, 0.005
fp_min, fp_max = -0.00, 0.7
f0_min, f0_max = 0, 3.0  # 0.7, 1.3


class TestSet(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x = x
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        return input_data

    def __len__(self):
        return len(self.idx)


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, x, syn_x, syn_ivim):
        self.x = x
        self.syn_x = syn_x
        self.syn_ivim = syn_ivim
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        x = self.idx[index]
        syn_x = self.syn_x[index]
        syn_ivim = self.syn_ivim[index]
        return x, syn_x, syn_ivim

    def __len__(self):
        return len(self.idx)


def loaddata():
    # train_num = 300
    train_data = []
    for idx, bf in enumerate(b_files[:train_num]):
        img = np.load(bf)['x'][::4]
        train_data.append(img)  # [slice:slice+slice_num:step]
    train_data = np.concatenate(train_data, 0).transpose([0, 3, 1, 2]).astype(np.float32)

    test_data = []
    for idx, bf in enumerate(b_files[train_num:]):
        img = np.load(bf)['x']
        test_data.append(img)  # [slice:slice+slice_num:step]
    test_data = np.concatenate(test_data, 0).transpose([0, 3, 1, 2]).astype(np.float32)
    print(train_data.shape, ' ', test_data.shape)

    syn_b = np.load(syn_path)['x'].transpose([0, 3, 1, 2]).astype(np.float32)
    syn_ivim = np.load(syn_path)['ivim'].transpose([0, 3, 1, 2]).astype(np.float32)

    trainset = TrainSet(train_data, syn_b, syn_ivim)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=2, drop_last=True)

    testset = TestSet(test_data)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False,  # slice_num // step
                                             num_workers=2, drop_last=True)
    return trainloader, testloader, train_data.shape[0] // batch_size


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ch = 16
        self.conv_in = nn.Sequential(
            nn.Conv2d(9, ch, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(ch), nn.LeakyReLU(),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(ch * 2), nn.LeakyReLU(),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(ch * 4), nn.LeakyReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(ch * 4, ch * 8, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(ch * 8), nn.LeakyReLU(),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(ch * 8, ch * 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(ch * 16), nn.LeakyReLU(),
        )
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(ch * 16 * (img_w // 32) * (img_h // 32), 1), nn.Sigmoid())

    def forward(self, inputs):
        # max_mat = torch.max(inputs.reshape((inputs.size(0), -1)), dim=1)[0].reshape((inputs.size(0), 1, 1, 1))
        # inputs_norm = torch.clamp(inputs / max_mat, 0, 1)
        # inputs_norm = torch.clamp(inputs / (inputs[:, :1] + 1e-8), 0, 1)
        x = self.conv_in(inputs)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        # x = self.avg_pool(x)
        out = self.fc(x.reshape((inputs.size(0), -1))).squeeze()
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.d1 = DownsampleLayer(9, 64)  # 9-64
        self.d2 = DownsampleLayer(64, 128)  # 64-128
        self.d3 = DownsampleLayer(128, 256)  # 128-256
        self.d4 = DownsampleLayer(256, 512)  # 256-512

        self.u1 = UpSampleLayer(512, 512)  # 512-1024-512
        self.u2 = UpSampleLayer(1024, 256)  # 1024-512-256
        self.u3 = UpSampleLayer(512, 128)  # 512-256-128
        self.u4 = UpSampleLayer(256, 64)  # 256-128-64

        self.o = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # max_mat = torch.max(inputs.reshape((inputs.size(0), -1)), dim=1)[0].reshape((inputs.size(0), 1, 1, 1))
        # inputs_norm = torch.clamp(inputs / max_mat, 0, 1)
        # inputs_norm = torch.clamp(inputs / (inputs[:, :1] + 1e-8), 0, 1)
        d_1, d1 = self.d1(inputs)
        d_2, d2 = self.d2(d1)
        d_3, d3 = self.d3(d2)
        d_4, d4 = self.d4(d3)

        u1 = self.u1(d4, d_4)
        u2 = self.u2(u1, d_3)
        u3 = self.u3(u2, d_2)
        u4 = self.u4(u3, d_1)

        # mask = (inputs[:, :1] > 0).float()
        # params = torch.clamp(torch.abs(self.o(u4)), 0, 1)  # * mask
        out_params = self.sigmoid(self.o(u4)[:, :4])  #
        dp = out_params[:, 0:1] * (dp_max - dp_min) + dp_min
        dt = out_params[:, 1:2] * (dt_max - dt_min) + dt_min
        fp = out_params[:, 2:3] * (fp_max - fp_min) + fp_min
        f0 = out_params[:, 3:4] * (f0_max - f0_min) + f0_min
        params = torch.cat((dp, dt, fp, f0), dim=1)

        out_rec = self.ivim_matmul(params) * inputs[:, :1]  # self.o(u4)[:, 3:]
        params_ture = torch.cat((dp, dt, fp / (fp + f0)), dim=1)

        return out_rec, params_ture, f0

    def ivim_matmul(self, params):
        flat = params.view(params.size(0), 4, params.size(2) * params.size(3))
        dp = flat[:, 0].unsqueeze(1)
        dt = flat[:, 1].unsqueeze(1)
        fp = flat[:, 2].unsqueeze(1)
        f0 = flat[:, 3].unsqueeze(1)
        b_fit_ = b_fit.unsqueeze(0).repeat(params.size(0), 1, 1)
        outputs = fp * torch.exp(-torch.bmm(b_fit_, dp)) + f0 * torch.exp(-torch.bmm(b_fit_, dt))
        outputs = outputs.view(params.size(0), b_values.shape[0], params.size(2), params.size(3))
        # print(outputs.shape)
        return outputs


def order(ivim):
    Dp_, Dt_, Fp_ = ivim[..., 0:1], ivim[..., 1:2], ivim[..., 2:]
    if np.mean(Dp_) < np.mean(Dt_):
        print('swap')
        temp = copy.deepcopy(Dp_)
        Dp_ = copy.deepcopy(Dt_)
        Dt_ = temp
        Fp_ = 1-Fp_
    return np.concatenate((Dp_, Dt_, Fp_), -1)


def train():
    trainloader, testloader, num_batch = loaddata()

    real_label = Variable(torch.ones(batch_size)).to(device)  # 定义真实的图片label为1
    fake_label = Variable(torch.zeros(batch_size)).to(device)

    g = Generator()
    g = g.to(device)

    d = Discriminator()
    d = d.to(device)

    total = sum([param.nelement() for param in g.parameters()]) + sum([param.nelement() for param in d.parameters()])
    print('params ', total, ' flops ', 0)
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())

    optimizer_g = optim.Adam(g.parameters(), lr=learning_rate_g, betas=(0.9, 0.999), weight_decay=1e-4)
    optimizer_d = optim.Adam(d.parameters(), lr=learning_rate_d, betas=(0.9, 0.999), weight_decay=1e-4)

    l1_loss_p = nn.SmoothL1Loss(reduction='mean')  # .to(device)  # reduce=True, size_average=True
    l1_loss_rec = nn.SmoothL1Loss(reduction='mean')
    # l1_loss = nn.L1Loss(reduction='mean')  # .to(device)  # reduce=True, size_average=True
    criterion = nn.BCELoss(reduction='mean')  # .to(device)

    # lc_, acc_ = [], []
    # sen_, spe_, auc_ = [], [], []
    for epoch in range(num_epochs):
        running_loss_rec = 0.0
        for i, data in enumerate(trainloader, 0):
            d.train()
            g.train()
            x, syn_x, syn_ivim = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
            x, syn_x, syn_ivim = Variable(x.to(device)), Variable(syn_x.to(device)), Variable(syn_ivim.to(device))

            outputs_rec_syn, ivim_pre_syn, _ = g(syn_x)
            loss_p = l1_loss_p(ivim_pre_syn, syn_ivim)

            real_d = d(x)
            loss_d_real = criterion(real_d, real_label)
            fake_d = d(outputs_rec_syn.detach())
            loss_d_fake = criterion(fake_d, fake_label)
            loss_d = loss_d_real + loss_d_fake

            optimizer_d.zero_grad()
            loss_d.backward()  # retain_graph=True
            optimizer_d.step()

            fake_d = d(outputs_rec_syn)
            loss_g_fake = criterion(fake_d, real_label)

            outputs_rec, ivim_pre, f0 = g(x)
            rec_d = d(outputs_rec)
            loss_rec = l1_loss_rec(outputs_rec, x)
            # print(torch.max(outputs_rec), torch.mean(outputs_rec), torch.min(outputs_rec))
            loss_g_rec = criterion(rec_d, real_label)

            loss_g = loss_g_rec + loss_g_fake

            # outputs_rec_syn, ivim_pre_syn = g(syn_x)
            loss_G = loss_rec + 1 * loss_p + 1e-4 * loss_g

            optimizer_g.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            loss_G.backward()  # retain_graph=True
            optimizer_g.step()
            running_loss_rec += loss_rec.item()
            if i == num_batch - 1:
                print('[%3d, %3d] lrec %.4f dp %.4f dt %.4f fp %.4f f0 %.4f lg %.4f ld %.4f' %
                      (epoch + 1, i + 1, running_loss_rec / num_batch, torch.mean(ivim_pre[:, 0]).item(),
                       torch.mean(ivim_pre[:, 1]).item(), torch.mean(ivim_pre[:, 2]).item(), torch.mean(f0).item(),
                       loss_G.item(), loss_d.item()))
                running_loss_rec = 0.0

                # err_ = []
                # pre = nn.Softmax(dim=1)(outputs_t)
                # eq = torch.eq(predicted_t, test_lb)
                # for ii in range(test_lb.shape[0]):
                #     err_t = 1 - pre[ii][test_lb[ii]].item()
                #     if eq[ii].item() is False and err_t < 0.9:
                #         err_.append(err_t)
                # print(err_)
    print('Finished Training')

    d.eval()
    g.eval()
    with torch.no_grad():
        ivim_pre_all, x_fit_pre_all = [], []
        for i, data in enumerate(testloader):
            num = b_files[i+train_num].split('/')[-1].split('.')[0]
            test_img = data
            test_img = Variable(test_img.to(device))
            outputs_rec, ivim_p, _ = g(test_img)
            ivim_pre = ivim_p.cpu().detach().numpy().transpose([0, 2, 3, 1]).astype(np.float32)
            x_fit_pre = outputs_rec.cpu().detach().numpy().transpose([0, 2, 3, 1]).astype(np.float32)
            ivim_pre = order(ivim_pre)
            ivim_pre_all.append(ivim_pre), x_fit_pre_all.append(x_fit_pre)

            np.savez(join(save_path, str(num) + '.npz'),
                     ivim=ivim_pre, x=x_fit_pre)

        np.savez(save_path, ivim=ivim_pre_all, x=x_fit_pre_all)

    def loadtest(dp):
        test_data = np.load(dp)['train_data'].transpose([0, 3, 1, 2]).astype(np.float32)
        print(test_data.shape)
        testset = TestSet(test_data)
        tl = torch.utils.data.DataLoader(testset, batch_size=test_data.shape[0], shuffle=False,
                                         num_workers=2, drop_last=True)
        return tl

    data_path_ = r'/home/public/Documents/hhy/ivim_new/data/mvi4.npz'
    save_path_ = r'/home/public/Documents/hhy/ivim_new/s2r_result/unet_gan_roi_fitS0_2'
    testloader = loadtest(data_path_)

    g.eval()
    with torch.no_grad():
        ivim_pre_all, x_fit_pre_all = [], []
        for i, data in enumerate(testloader):
            # num = b_files[i + train_num].split('/')[-1].split('.')[0]
            test_img = data
            test_img = Variable(test_img.to(device))
            outputs_rec, ivim_p, _ = g(test_img)
            ivim_pre = ivim_p.cpu().detach().numpy().transpose([0, 2, 3, 1]).astype(np.float32)
            x_fit_pre = outputs_rec.cpu().detach().numpy().transpose([0, 2, 3, 1]).astype(np.float32)
            ivim_pre_all.append(ivim_pre), x_fit_pre_all.append(x_fit_pre)

        ivim_pre_all = np.concatenate(ivim_pre_all, 0)
        x_fit_pre_all = np.concatenate(x_fit_pre_all, 0)
        np.savez(save_path_, ivim=ivim_pre_all, x=x_fit_pre_all)

# print(torch.__version__)
# print(torch.cuda.device_count())
# print(torch.cuda.is_available())

for _ in range(repeat_num):
    train()
