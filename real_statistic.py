import math
import os
import numpy as np
from os.path import join
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import pandas as pd


def processing_outliers(inputs, low=1, high=99):
    for ii in range(inputs.shape[-1]):
        # ql = np.percentile(inputs[..., ii], low)
        qh = np.percentile(inputs[..., ii], high)
        inputs[..., ii] = np.clip(inputs[..., ii], 0, qh)
        # inputs[inputs == qh] = 0
    return inputs


b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000])
width = 0.15

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18}

path_list = [r'/home/public/Documents/hhy/data/IVIM6-1/real_threshold2',
             r'/home/public/Documents/hhy/IVIM/UDA_real/leastsq',
             r'/home/public/Documents/hhy/IVIM/UDA_real/ANN',
             r'/home/public/Documents/hhy/IVIM/UDA_real/DNN',
             r'/home/public/Documents/hhy/IVIM/UDA_real/CNN_ul1',
             r'/home/public/Documents/hhy/IVIM/UDA_real/CNN_GAN2_3'  # 3_4 CNN_GAN2_2, CNN_GAN2_3
             ]  #

name_list = ['Ground Truth',
             'Least-Square',
             'ANN',
             'DNN',
             'self-Unet',
             'UNet+GAN2'
             ]

# for nfl_idx, nfl in enumerate(nofit_list):
files = os.listdir(path_list[-1])
files.sort(key=lambda x: (int(x.split('.')[0])))

b_list = []
# b0 = []
for idx, pl in enumerate(path_list[:1]):
    b_ = []
    for idxt, fls in enumerate(files):
        if idx == 0:
            b_.append(np.load(join(pl, fls))['x'][17][..., 1:])
        else:
            b_.append(np.load(join(pl, fls))['x_fit_pre'])
    print(np.array(b_).shape)
    b_list.append(b_)
b_list = np.array(b_list)
print(b_list.shape)

mask = np.ones_like(b_list[0][..., 0])
print(mask.shape)
for i in range(b_list.shape[-1]):
    indice = b_list[0][..., i] == 0.0
    mask[indice] = 0.0

ivim_all = []
for idx, fls in enumerate(files):
    ivim_list = []
    for pl in path_list[1:]:
        ivim_list.append(processing_outliers(np.load(join(pl, fls))['ivim'], 0, 100))
    ivim_list = np.array(ivim_list)
    ivim_all.append(ivim_list)
ivim_all = np.array(ivim_all)
print(ivim_all.shape)
ivim_all = np.transpose(ivim_all, [1, 0, 2, 3, 4])  # .reshape((5, 83, -1, 3))
print(ivim_all.shape)

param_name_list = ['Dp', 'Dt', 'Fp']
param_clip_l = [-0.05, -0.0005, -0.05]
param_clip_h = [0.3, 0.005, 1.0]  # 0.6

y_lim = [70000, 30000, 20000]  # [50000, 17000, 10000]
# param_clip_l = [0.02, 0.0009, 0.01]
# param_clip_h = [0.28, 0.0049, 0.59]

patient = 1
fig, ax = plt.subplots(ivim_all.shape[0], 3, figsize=(9, 10))  #
hist_all = []
for i in range(ivim_all.shape[0]):
    outlier_list = []
    for j in range(3):





        plt.subplot(ivim_all.shape[0], 3, j+i*3+1)
        plt.xlim((param_clip_l[j], param_clip_h[j]))
        plt.ylim((0, y_lim[j]))
        if i < ivim_all.shape[0] - 1:
            plt.xticks([])
        ivim = np.clip(ivim_all[i][..., j] * mask, param_clip_l[j], param_clip_h[j]).reshape((-1))
        ivim = [ivim_ for ivim_ in ivim if ivim_ != 0.0]
        d_list_pd = pd.DataFrame({'ivim': ivim})
        d_list_pd.ivim.plot(kind='hist', bins=500, color='skyblue', edgecolor='black', ylabel=None)  # , density=True, stacked=True

        hist = np.zeros(501)
        interval = np.max(ivim) / 500.0
        for ivim_ in ivim:
            hist[int(ivim_ / interval)] += 1
        # hist_all.append(hist)
        hist = hist[:-1]
        outlier_sum = 0
        outlier = np.percentile(hist, 99.8)
        print(outlier)
        for hist_ in hist:
            if hist_ >= outlier:
                outlier_sum += hist_
        outlier_list.append(outlier_sum / np.sum(hist))
    print('dp %.4f\tdt %.4f\tfp %.4f' % (outlier_list[0], outlier_list[1], outlier_list[2]))
plt.tight_layout()
plt.show()

for i in range(3):
    print(param_name_list[i])
    for j in range(ivim_all.shape[0]):
        # ivim = np.clip(ivim_all[j][..., i], param_clip_l[i], param_clip_h[i])
        ivim = ivim_all[j][..., i]  # * mask
        mean_list = np.mean(ivim, (1, 2))
        ql = np.percentile(mean_list, 10)
        qh = np.percentile(mean_list, 90)
        # mean_list = np.clip(mean_list, ql, qh)
        mean_list = [ml for ml in mean_list if ml > ql and ml < qh]
        print('%15s ' % name_list[j+1], '%2.4f' % (np.std(mean_list, ddof=1) / np.mean(mean_list) * 100))
    print()

