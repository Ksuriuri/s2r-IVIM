import math
import os
import numpy as np
from os.path import join
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf


def rmse(hr, sr):
    return np.sqrt(np.mean(np.square(hr - sr)))


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def clim(x, min_v, max_v):
    return np.clip(x, min_v, max_v)


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
             r'/home/public/Documents/hhy/IVIM/UDA_real/CNN_GAN3_5_83'  # CNN_GAN2_2, CNN_GAN2_3
             ]  #

name_list = ['Ground Truth',
             'Nonlinear Least Square',
             'ANN',
             'DNN',
             'Self U-net',
             'Proposed'
             ]

# for nfl_idx, nfl in enumerate(nofit_list):
files = os.listdir(path_list[-1])
files.sort(key=lambda x: (int(x.split('.')[0])))

b_list = []
# b0 = []
for idx, pl in enumerate(path_list):
    b_ = []
    for idxt, fls in enumerate(files):
        if idx == 0:
            b_.append(np.load(join(pl, fls))['x'][17][..., 1:])
            # b0.append(np.load(join(pl, fls))['x'][17][..., :1])
        else:
            b_.append(np.load(join(pl, fls))['x_fit_pre'])
    print(np.array(b_).shape)
    b_list.append(b_)
b_list = np.array(b_list)


# *************************************************************************
# ****************************** fit b ssim *******************************
# *************************************************************************
# graph = tf.Graph()
# with graph.as_default():
x_ = tf.placeholder(tf.float32, shape=(None, b_list.shape[2], b_list.shape[3], None))
y_ = tf.placeholder(tf.float32, shape=(None, b_list.shape[2], b_list.shape[3], None))
ssim = tf.reduce_mean(tf.image.ssim(x_, y_, max_val=1.0))


def calc_ssim(gt_, pre_):
    ssim_ = []
    with tf.Session() as sess:  # graph=graph
        tf.global_variables_initializer().run()
        for ii in range(b_list.shape[-1]):
            ssim_.append(sess.run(ssim, feed_dict={x_: gt_[..., ii:ii+1], y_: pre_[..., ii:ii+1]}))
    return np.array(ssim_)

ssim_list = []
for bl in b_list[1:]:
    ssim_list.append(calc_ssim(b_list[0], bl))
ssim_list = np.array(ssim_list)

for idx, sl in enumerate(ssim_list):
    print('%s ssim: %.3f' % (name_list[idx+1], sl.mean()))


# *************************************************************************
# ****************************** fit b psnr *******************************
# *************************************************************************
def calc_psnr(gt_, pre_):
    psnr_ = []
    for ii in range(b_list.shape[-1]):
        psnr_.append(psnr(gt_[..., ii:ii+1], pre_[..., ii:ii+1]))
    return np.array(psnr_)

psnr_list = []
for bl in b_list[1:]:
    psnr_list.append(calc_psnr(b_list[0], bl))
psnr_list = np.array(psnr_list)

for idx, pl in enumerate(psnr_list):
    print('%s psnr: %.3f' % (name_list[idx+1], pl.mean()))

# *************************************************************************
# ****************************** fit b rmse *******************************
# *************************************************************************
def calc_rmse(gt_, pre_):
    rmse_ = []
    for ii in range(b_list.shape[-1]):
        rmse_.append(rmse(gt_[..., ii:ii+1], pre_[..., ii:ii+1]))
    return np.array(rmse_)

rmse_list = []
for bl in b_list[1:]:
    rmse_list.append(calc_rmse(b_list[0], bl))
rmse_list = np.array(rmse_list)

for idx, rl in enumerate(rmse_list):
    print('%s rmse: %.3f' % (name_list[idx+1], rl.mean()))


# *************************************************************************
# ****************************** all plot *********************************
# *************************************************************************
fig, ax = plt.subplots(3, 1, figsize=(9, 2.5*3))  # 3.3
x = np.arange(ssim_list.shape[-1])

# font2 = {'family': 'Times New Roman',
#          'weight': 'normal',
#          'size': 14}

index_list = [ssim_list,
              psnr_list,
              rmse_list]

ylabel_list = ['Structural Similarity',
               'Peak-Signal-to-Noise-Ratio',
               'Root-Mean-Squard-Error']

ylim_list = [(0.70, 1.08),
             (17, 29),
             (0.04, 0.20)]

for i, il in enumerate(index_list):
    for idx, sl in enumerate(il):
        label = name_list[idx + 1] + ' (%.3f±%.2f)' % (sl.mean(), np.std(sl, ddof=1))
        ax[i].bar(x + idx * width, sl, width, alpha=0.9, label=label)
    ax[i].legend(loc=2)
    ax[i].set_xticks(x + width * il.shape[0] / 2 - width / 2)
    if i == 2:
        ax[i].set_xticklabels(['b=' + str(bv) for bv in b_values[1:]])
        ax[i].set(xlabel='S(b)/S(0)')
    else:
        ax[i].set_xticklabels(['' for bv in b_values[1:]])
    ax[i].set(ylabel=ylabel_list[i])
    ax[i].set_ylim(ylim_list[i])
plt.tight_layout()
plt.show()

# *************************************************************************
# ****************************** visualization ****************************
# *************************************************************************
ivim_all = []
for idx, fls in enumerate(files):
    ivim_list = []
    for pl in path_list[1:]:
        ivim_list.append(processing_outliers(np.load(join(pl, fls))['ivim'], 0, 95))
    ivim_list = np.array(ivim_list)
    ivim_all.append(ivim_list)
ivim_all = np.array(ivim_all)
for i in range(len(path_list) - 1):
    print('dp ', ivim_all[:, i, :, :, 0].max(), ivim_all[:, i, :, :, 0].mean(), ivim_all[:, i, :, :, 0].min())
    print('dt ', ivim_all[:, i, :, :, 1].max(), ivim_all[:, i, :, :, 1].mean(), ivim_all[:, i, :, :, 1].min())
    print('fp ', ivim_all[:, i, :, :, 2].max(), ivim_all[:, i, :, :, 2].mean(), ivim_all[:, i, :, :, 2].min())
    print()
# print(ivim_all.shape)

for idx, fls in enumerate(files):
    ivim_list = []
    for pl in path_list[1:]:
        ivim_list.append(processing_outliers(np.load(join(pl, fls))['ivim'], 0, 95))
    ivim_list = np.array(ivim_list)

    fig = plt.figure(figsize=(14, 6))  #
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(3, ivim_list.shape[0]),
                     direction='row',
                     axes_pad=0.2,
                     cbar_location='right',
                     cbar_mode='edge',
                     cbar_size='5%',
                     cbar_pad=0.15)

    label_list = [r'$D_p$ [mm$^2$/sec]',
                  r'$D_t$ [mm$^2$/sec]',
                  r'$F_p$ [%]']
    for ii in range(3):
        for jj in range(ivim_list.shape[0]):
            cp = grid[ivim_list.shape[0]*ii+jj].imshow(ivim_list[jj][..., ii], cmap='gray')
            if ii == 2:
                grid[ii*ivim_list.shape[0]+jj].set_xlabel(name_list[jj+1], font1)
        grid[ivim_list.shape[0]*(ii+1)-1].cax.colorbar(cp)
        grid[ii*ivim_list.shape[0]].set_ylabel(label_list[ii], font1)

    for ii, axis in enumerate(grid):
        axis.set_xticks([])
        axis.set_yticks([])
        # if (i == 1) or (i == 2):
        #     axis.set_axis_off()

    plt.tight_layout()
    plt.show()

    # # *************************************************************************
    # # ****************************** b value **********************************
    # # *************************************************************************
    # # plot gt
    # fig = plt.figure(figsize=(15, 8))  #
    # plt.title('S(b)/S(0)', y=-0.1)
    # plt.axis('off')
    # grid = ImageGrid(fig, 111,
    #                  nrows_ncols=(b_list.shape[0], b_list.shape[-1]),
    #                  direction='row',
    #                  axes_pad=0.2,
    #                  cbar_location='right',
    #                  cbar_mode='edge',
    #                  cbar_size='5%',
    #                  cbar_pad=0.15)
    #
    # for ii, bl in enumerate(b_list):
    #     for jj in range(b_list.shape[-1]):
    #         cp = grid[b_list.shape[-1]*ii+jj].imshow(b_list[ii][idx][..., jj], cmap='gray', clim=(0, 1))  #
    #         if jj == 0:
    #             grid[b_list.shape[-1]*ii].set_ylabel(name_list[ii])
    #         if ii == b_list.shape[0] - 1:
    #             grid[b_list.shape[-1]*ii+jj].set_xlabel('b=' + str(b_values[jj+1]))  # 'S(' + str(b_values[jj+1]) + ')/S(0)'
    #     grid[b_list.shape[-1]*(ii+1)-1].cax.colorbar(cp)
    #
    # for ii, axis in enumerate(grid):
    #     axis.set_xticks([])
    #     axis.set_yticks([])
    #     # if (i == 1) or (i == 2):
    #     #     axis.set_axis_off()
    #
    # plt.tight_layout()
    # plt.show()
