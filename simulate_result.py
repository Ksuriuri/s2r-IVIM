import math
import random
import numpy as np
import cv2
import os
import tensorflow as tf
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
from pylab import *
from mpl_toolkits.axes_grid1 import ImageGrid


def rmse(hr, sr):
    return np.sqrt(np.mean(np.square(hr - sr)))


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def processing_outliers(inputs):
    for ii in range(inputs.shape[-1]):
        ql = np.percentile(inputs[..., ii], 1)
        qh = np.percentile(inputs[..., ii], 99)
        inputs[..., ii] = np.clip(inputs[..., ii], ql, qh)
    return inputs


path_list = [r'/home/public/Documents/hhy/data/IVIM6-1/simulate_data/test1.npz',
             r'/home/public/Documents/hhy/IVIM/UDA_fake/leastsq/test.npz',
             r'/home/public/Documents/hhy/IVIM/UDA_fake/DNN/test.npz',
             r'/home/public/Documents/hhy/IVIM/UDA_fake/ANN/test.npz',
             r'/home/public/Documents/hhy/IVIM/UDA_fake/self_Unet/test.npz',
             r'/home/public/Documents/hhy/IVIM/UDA_fake/Unet/test.npz'
             ]

method_names = ['Least-Sqaure',
                'DNN',
                'ANN',
                'self-Unet',
                'Unet'
                ]

ivim_list = []
for pl in path_list:
    ivim_list.append(np.load(pl)['ivim'])

ivim_list = np.array(ivim_list)

# *************************************************************************
# ****************************** ivim ssim ******************************
# *************************************************************************
graph = tf.Graph()
with graph.as_default():
    x_ = tf.placeholder(tf.float32, shape=ivim_list[0][..., :1].shape)
    y_ = tf.placeholder(tf.float32, shape=ivim_list[0][..., :1].shape)
    ssim = tf.reduce_mean(tf.image.ssim(x_, y_, max_val=1.0))


def calc_ssim(gt_, pre_):
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        ssim_ = []
        for jj in range(3):
            ssim_.append(sess.run(ssim, feed_dict={x_: gt_[..., jj: jj+1], y_: pre_[..., jj: jj+1]}))
        ssim_ = np.array(ssim_)
    return np.array(ssim_)


ssim_list = []
for i, ivim_ in enumerate(ivim_list[1:]):
    ssim_list.append(calc_ssim(ivim_list[0], ivim_))
ssim_list = np.array(ssim_list)

print('ivim ssim')
print('%20s\t  dp  \t  dt  \t  fp  ' % '')
for i, sl in enumerate(ssim_list):
    print('%20s\t%.3f\t%.3f\t%.3f' % (method_names[i], sl[0], sl[1], sl[2]))
print()

# *************************************************************************
# ****************************** ivim psnr ********************************
# *************************************************************************
print('ivim psnr')
print('%20s\t  dp  \t  dt  \t  fp  ' % '')
for i, ivim_ in enumerate(ivim_list[1:]):
    psnr_list = []
    for j in range(3):
        psnr_list.append(psnr(ivim_list[0][..., j], ivim_[..., j]))
    print('%20s\t%.3f\t%.3f\t%.3f' % (method_names[i], psnr_list[0], psnr_list[1], psnr_list[2]))
print()

# *************************************************************************
# ****************************** ivim rmse ********************************
# *************************************************************************
print('ivim rmse')
print('%20s\t  dp  \t  dt  \t  fp  ' % '')
for i, ivim_ in enumerate(ivim_list[1:]):
    rmse_list = []
    for j in range(3):
        rmse_list.append(rmse(ivim_list[0].reshape((-1, 3))[..., j], ivim_.reshape((-1, 3))[..., j]))
    print('%20s\t%.4f\t%.6f\t%.4f' % (method_names[i], rmse_list[0], rmse_list[1], rmse_list[2]))
print()

# *************************************************************************
# ****************************** boxplot **********************************
# *************************************************************************
error_list = []
for ivim_ in ivim_list[1:]:
    error_list.append(ivim_list[0].reshape((-1, 3)) - ivim_.reshape((-1, 3)))
error_list = np.array(error_list)
print(error_list.shape)

fg, ax = plt.subplots(3, 1, sharex='all', sharey='row', figsize=(8, 10))

ivim_names = [r'($D_p$ fit)-($D_p$ true) [mm$^2$/sec]',
              r'($D_t$ fit)-($D_t$ true) [mm$^2$/sec]',
              r'($F_p$ fit)-($F_p$ true) [%]']

ax[0].set(title='Boxplot')
for i in range(3):
    ax[i].boxplot(list(error_list[..., i]), showfliers=False, labels=method_names)
    ax[i].set(ylabel=ivim_names[i])
    ax[i].axhline(linestyle='--')
    # ax[i].set(xlabel='SNR')
    # ax[2, 0].set_xticklabels(snr_vals)

fg.tight_layout()
plt.show()
