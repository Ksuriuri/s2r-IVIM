import math
import os
import numpy as np
from os.path import join
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf


def moving_avg(input_array, beta1=0.8, beta2=0.3):
    input_array = input_array[::-1]
    # reverse
    output_array = []
    output_array.append(input_array[0])
    for ii in range(1, len(input_array) - 1):
        output_array.append(output_array[ii-1] * beta1 + (1 - beta1) * input_array[ii])
    output_array.append(input_array[-1])
    # forward
    forward_arr = np.array(output_array)[::-1]
    output_array = []
    output_array.append(forward_arr[0])
    for ii in range(1, len(forward_arr)):
        output_array.append(output_array[ii - 1] * beta2 + (1 - beta2) * forward_arr[ii])
    return np.array(output_array)


def rmse(hr, sr):
    return np.sqrt(np.mean(np.square(hr - sr)))


def psnr(imgs1, imgs2):
    output = []
    PIXEL_MAX = 1.0
    for ii in range(imgs1.shape[0]):
        mse = np.mean((imgs1[ii] - imgs2[ii]) ** 2)
        if mse == 0:
            output.append(100.0)
        else:
            output.append(20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
    return np.mean(output)


def clim(x, min_v, max_v):
    return np.clip(x, min_v, max_v)


def processing_outliers(inputs, low=1, high=99):
    for ii in range(inputs.shape[-1]):
        # ql = np.percentile(inputs[..., ii], low)
        qh = np.percentile(inputs[..., ii], high)
        inputs[..., ii] = np.clip(inputs[..., ii], 0, qh)
        # inputs[inputs == qh] = 0
    return inputs


def curve_plot(input_arr, name_l, x_labels, title='title', loc='lower left', ylim=None):
    fg, ax = plt.subplots(3, 1, sharex='all', figsize=(4.5, 6))
    if ylim is not None:
        plt.ylim(ylim)
    x_ticks = np.arange(input_arr.shape[1])

    ax[0].plot(x_ticks, input_arr[0, :, 0], label=name_l[0])
    ax[0].plot(x_ticks, input_arr[1, :, 0], label=name_l[1])
    ax[0].plot(x_ticks, input_arr[2, :, 0], label=name_l[2])
    ax[0].legend(loc=loc)
    # ax[0].set(ylabel=r'Dp')
    ax[0].set_ylabel(r'Dp', fontdict=font1)
    ax[0].set_xticks(x_ticks)
    ax[0].set_xticklabels(x_labels)
    ax[0].set_title(title)

    ax[1].plot(x_ticks, input_arr[0, :, 1], label=name_l[0])
    ax[1].plot(x_ticks, input_arr[1, :, 1], label=name_l[1])
    ax[1].plot(x_ticks, input_arr[2, :, 1], label=name_l[2])
    ax[1].legend(loc=loc)
    # ax[1].set(ylabel=r'Dt')
    ax[1].set_ylabel(r'Dt', fontdict=font1)
    # ax[1].set_yticks(np.arange(0.9925, 1.015, 3))
    ax[1].set_xticks(x_ticks)
    ax[1].set_xticklabels(x_labels)

    LS = ax[2].plot(x_ticks, input_arr[0, :, 2], label=name_l[0])
    DNN = ax[2].plot(x_ticks, input_arr[1, :, 2], label=name_l[1])
    CNN = ax[2].plot(x_ticks, input_arr[2, :, 2], label=name_l[2])
    ax[2].legend(loc=loc)
    # ax[2].set(ylabel=r'Fp')
    ax[2].set_ylabel(r'Fp', fontdict=font1)
    ax[2].set(xlabel='Amount of training data')
    ax[2].set_xticks(x_ticks)
    ax[2].set_xticklabels(x_labels)

    fg.tight_layout()
    plt.show()


b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000])
width = 0.15

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 13}

data_num = [83, 83, 65, 50, 35, 15, 5]  #
one_patient_num = 18
data_num_all = np.array(data_num) * one_patient_num
print(data_num_all)

path_list = [# r'/home/public/Documents/hhy/data/IVIM6-1/real_threshold2',
             # r'/home/public/Documents/hhy/IVIM/UDA_real/leastsq',
             # r'/home/public/Documents/hhy/IVIM/UDA_real/ANN',
             r'/home/public/Documents/hhy/IVIM/UDA_real/DNN',
             r'/home/public/Documents/hhy/IVIM/UDA_real/CNN_ul3_1',
             # r'/home/public/Documents/hhy/IVIM/UDA_real/CNN1',
             r'/home/public/Documents/hhy/IVIM/UDA_real/CNN_GAN3_6'  # CNN_GAN2_2, CNN_GAN2_3
             ]  #

name_list = [# 'Ground Truth',
             # 'Nonlinear Least Square',
             # 'ANN',
             'DNN',
             'Self U-net',
             # 'UNet sd',
             # 'UNet+GAN1',
             'Proposed'
             ]

param_clip_l = [-0.05, -0.0005, -0.05]
param_clip_h = [0.3, 0.005, 0.6]

# for nfl_idx, nfl in enumerate(nofit_list):
files = os.listdir(path_list[-1] + '_' + str(data_num[0]))
files.sort(key=lambda x: (int(x.split('.')[0])))
np.random.seed(2021)
np.random.shuffle(files)
files = np.array(files)  # [65:]

# b_list = []
# # b0 = []
# for idx, pl in enumerate(path_list):
#     b_ = []
#     for idxt, fls in enumerate(files):
#         if idx == 0:
#             b_.append(np.load(join(pl, fls))['x'][17][..., 1:])
#             # b0.append(np.load(join(pl, fls))['x'][17][..., :1])
#         else:
#             b_.append(np.load(join(pl, fls))['x_fit_pre'])
#     print(np.array(b_).shape)
#     b_list.append(b_)
# b_list = np.array(b_list)

# b0 = np.array(b0)
# idx0 = b0 <= 0
# b0[idx0] = 1.0
#
# mask = np.ones_like(b0)
# mask[idx0] = 0
# b_list = b_list / b0 * mask
#
# for ii in range(b_list.shape[0]):
#     for jj in range(b_list.shape[1]):src_dis_
#         for kk in range(b_list.shape[-1]):
#             b_list[ii, jj][..., kk] = processing_outliers(b_list[ii, jj][..., kk], 5, 95)

# outliers_idx = b_list[0] > 1.0
# for idx in range(b_list.shape[0]):
#     b_list[idx][outliers_idx] = 1.0
#
# outliers_idx = b_list[0] <= 0.0
# for idx in range(b_list.shape[0]):
#     b_list[idx][outliers_idx] = 0.0

# b_list[0] = clim(b_list[0], 0, 1)
# outliers_idx = b_list[0] == 1.0
# for idx in range(1, b_list.shape[0]):
#     b_list[idx][outliers_idx] = 1.0
#
# outliers_idx = b_list[0] == 0.0
# for idx in range(1, b_list.shape[0]):
#     b_list[idx][outliers_idx] = 0.0

# *************************************************************************
# ****************************** visualization ****************************
# *************************************************************************
ivim_all = []
for idx, fls in enumerate(files):
    ivim_list = []
    for pl in path_list:
        ivim_list_ = []
        for dn in data_num:
            ivim_ = np.load(join(pl + '_%s' % str(dn), fls))['ivim']
            for ivim_n in range(3):
                ivim_[..., ivim_n] = np.clip(ivim_[..., ivim_n], param_clip_l[ivim_n], param_clip_h[ivim_n])
            ivim_list_.append(ivim_)  # 95
        ivim_list.append(ivim_list_)
    ivim_list = np.array(ivim_list)
    ivim_all.append(ivim_list)
ivim_all = np.array(ivim_all)
print(ivim_all.shape)

# for ii in range(1, len(data_num)):
#     ivim_all[:, :, ii] = np.abs(ivim_all[:, :, ii] - ivim_all[:, :, 0])

# *************************************************************************
# ****************************** fit b ssim *******************************
# *************************************************************************
# graph = tf.Graph()
# with graph.as_default():
x_ = tf.placeholder(tf.float32, shape=(None, ivim_all.shape[3], ivim_all.shape[4], None))
y_ = tf.placeholder(tf.float32, shape=(None, ivim_all.shape[3], ivim_all.shape[4], None))
ssim = tf.reduce_mean(tf.image.ssim(x_, y_, max_val=1.0))


def calc_ssim(gt_, pre_):
    ssim_ = []
    with tf.Session() as sess:  # graph=graph
        tf.global_variables_initializer().run()
        for ii in range(ivim_all.shape[-1]):
            ssim_.append(sess.run(ssim, feed_dict={x_: gt_[..., ii:ii+1], y_: pre_[..., ii:ii+1]}))
    return np.array(ssim_)

ssim_list = []
for mn in range(ivim_all.shape[1]):
    ssim_list_ = []
    for sp in range(1, ivim_all.shape[2]):
        ssim_list_.append(calc_ssim(ivim_all[:, mn, 0], ivim_all[:, mn, sp]))
    ssim_list_ = moving_avg(ssim_list_)
    ssim_list.append(ssim_list_)
ssim_list = np.array(ssim_list)
print(ssim_list.shape)
# (3, 5, 3)

curve_plot(ssim_list, name_list, data_num_all[1:], 'SSIM of test set')

# for idx, sl in enumerate(ssim_list):
#     print('%s ssim: %.3f' % (name_list[idx+1], sl.mean()))

# *************************************************************************
# ****************************** fit b psnr *******************************
# *************************************************************************
psnr_list = []
for mn in range(ivim_all.shape[1]):
    psnr_list_ = []
    for sp in range(1, ivim_all.shape[2]):
        psnr_ = []
        for ii in range(ivim_all.shape[-1]):
            psnr_.append(psnr(ivim_all[:, mn, 0, :, :, ii: ii+1], ivim_all[:, mn, sp, :, :, ii: ii+1]))
        psnr_list_.append(psnr_)
    psnr_list_ = moving_avg(np.array(psnr_list_))
    psnr_list.append(psnr_list_)
psnr_list = np.array(psnr_list)
print(psnr_list.shape)
# (3, 5, 3)

curve_plot(psnr_list, name_list, data_num_all[1:], 'PSNR of test set', loc='upper right')  # , ylim=(0, 75)

# for idx, pl in enumerate(psnr_list):
#     print('%s psnr: %.3f' % (name_list[idx+1], pl.mean()))

# # *************************************************************************
# # ****************************** fit b rmse *******************************
# # *************************************************************************
rmse_list = []
for mn in range(ivim_all.shape[1]):
    rmse_list_ = []
    for sp in range(1, ivim_all.shape[2]):
        rmse_ = []
        for ii in range(ivim_all.shape[-1]):
            rmse_.append(rmse(ivim_all[:, mn, 0, :, :, ii: ii+1], ivim_all[:, mn, sp, :, :, ii: ii+1]))
        rmse_list_.append(rmse_)
    rmse_list_ = moving_avg(np.array(rmse_list_))
    rmse_list.append(rmse_list_)
rmse_list = np.array(rmse_list)
print(rmse_list.shape)
# (3, 5, 3)

curve_plot(rmse_list, name_list, data_num_all[1:], 'RMSE of test set', loc='upper left')

# for idx, pl in enumerate(psnr_list):
#     print('%s psnr: %.3f' % (name_list[idx+1], pl.mean()))
#
# # *************************************************************************
# # ****************************** all plot *********************************
# # *************************************************************************
# fig, ax = plt.subplots(3, 1, figsize=(9, 2.5*3))  # 3.3
# x = np.arange(ssim_list.shape[-1])
#
# index_list = [ssim_list,
#               psnr_list,
#               rmse_list]
#
# ylabel_list = ['Structural Similarity',
#                'Peak-Signal-to-Noise-Ratio',
#                'Root-Mean-Squard-Error']
#
# ylim_list = [(0.70, 1.08),
#              (17, 29),
#              (0.04, 0.20)]
#
# for i, il in enumerate(index_list):
#     for idx, sl in enumerate(il):
#         label = name_list[idx + 1] + ' (%.3f±%.2f)' % (sl.mean(), np.std(sl, ddof=1))
#         ax[i].bar(x + idx * width, sl, width, alpha=0.9, label=label)
#     ax[i].legend(loc=2)
#     ax[i].set_xticks(x + width * il.shape[0] / 2 - width / 2)
#     if i == 2:
#         ax[i].set_xticklabels(['b=' + str(bv) for bv in b_values[1:]])
#         ax[i].set(xlabel='S(b)/S(0)')
#     else:
#         ax[i].set_xticklabels(['' for bv in b_values[1:]])
#     ax[i].set(ylabel=ylabel_list[i])
#     ax[i].set_ylim(ylim_list[i])
# plt.tight_layout()
# plt.show()
#
#
# label_list = [r'D_p',
#               r'D_t',
#               r'F_p']
#
# for idx, ivim_list in enumerate(ivim_all[65:]):
#     for ivim_num in range(3):
#         print(label_list[ivim_num])
#         fig = plt.figure(figsize=(14, 6))  #
#         grid = ImageGrid(fig, 111,
#                          nrows_ncols=(ivim_list.shape[0], ivim_list.shape[1]),
#                          direction='row',
#                          axes_pad=0.01,
#                          cbar_location='right',
#                          cbar_mode='edge',
#                          cbar_size='5%',
#                          cbar_pad=0.15)
#
#         for ii in range(ivim_list.shape[0]):
#             print('%.4f\t%.4f\t%.4f\t%.4f' % (np.mean(ivim_list[ii][1][..., ivim_num]),
#                                                 np.mean(ivim_list[ii][2][..., ivim_num]),
#                                                 np.mean(ivim_list[ii][3][..., ivim_num]),
#                                                 np.mean(ivim_list[ii][4][..., ivim_num])))
#             for jj in range(ivim_list.shape[1]):
#                 cp = grid[ivim_list.shape[1]*ii+jj].imshow(ivim_list[ii][jj][..., ivim_num], cmap='gray')
#                 if ii == ivim_list.shape[0] - 1:
#                     grid[ii*ivim_list.shape[1]+jj].set_xlabel(str(data_num[jj]), font1)
#             # grid[ivim_list.shape[1]*(ii+1)-1].cax.colorbar(cp)
#             grid[ii*ivim_list.shape[1]].set_ylabel(name_list[ii], font1)
#
#         for ii, axis in enumerate(grid):
#             axis.set_xticks([])
#             axis.set_yticks([])
#             # if (i == 1) or (i == 2):
#             #     axis.set_axis_off()
#
#         # plt.tight_layout()
#         plt.show()
