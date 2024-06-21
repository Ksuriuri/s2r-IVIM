import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from mpl_toolkits.axes_grid1 import ImageGrid
from glob import glob
from os.path import join
import cv2


# define ivim function
def ivim(b, dp, dt, fp):
    return fp * np.exp(np.matmul(-dp, b)) + (1-fp) * np.exp(np.matmul(-dt, b))


def ivim_p2p(b, dp, dt, fp):
    return fp * np.exp(-b * dp) + (1-fp) * np.exp(-b * dt)


def snr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    # PIXEL_MAX = 1.0
    return 20 * math.log10(math.sqrt(np.mean(img1 ** 2)) / math.sqrt(mse))


def patch(inputs, w, h):
    patchs = []
    for i in range(0, inputs.shape[0], w):
        for j in range(0, inputs.shape[1], h):
            patchs.append(inputs[i: i+w, j: j+h])
    return np.array(patchs)


def unpatch(inputs, w, h):
    wp, hp = inputs.shape[1], inputs.shape[2]
    output = np.zeros((w, h, inputs.shape[-1]))
    for i in range(0, w, wp):
        for j in range(0, h, hp):
            output[i: i+wp, j: j+hp] = inputs[i//wp*h//hp+j//hp]
    return np.array(output)


img_w = 160
img_h = 192

img_wp, img_hp = 160, 192  # 32, 32

# define b values
b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000])
b_gt = b_values
s0 = 1500  # 1500

# batch_size = 64  # 64
train_num = 1494  # 14940  # batch_size * 300  # 200
test_num = 50

min_n, max_n = 0, 250  # 180  0, 165

dp_min, dp_max = 0.0, 0.2  # 0.01, 0.1
dt_min, dt_max = 0.0005, 0.0025  # 0.0005, 0.003
fp_min, fp_max = 0.0, 0.9  # 0.0, 0.4

# 1: no d, 2: d(0-1)
train_save_path = r'/home/public/Documents/hhy/data/IVIM6-1/simulate_data/train3.npz'
train_gt_save_path = r'/home/public/Documents/hhy/data/IVIM6-1/simulate_data/train_gt3.npz'
test_save_path = r'/home/public/Documents/hhy/data/IVIM6-1/simulate_data/test3.npz'

Dp_train = np.random.uniform(dp_min, dp_max, (train_num * img_wp * img_hp, 1))  # + 0.4
Dt_train = np.random.uniform(dt_min, dt_max, (train_num * img_wp * img_hp, 1))  # + 0.001
Fp_train = np.random.uniform(fp_min, fp_max, (train_num * img_wp * img_hp, 1))  # + 0.1
b_values_ = np.expand_dims(b_values, 0)
X_train = ivim(b_values_, Dp_train, Dt_train, Fp_train)  # range 0 - 1

Dp_train = np.reshape(Dp_train, (train_num, img_wp, img_hp, 1))
Dt_train = np.reshape(Dt_train, (train_num, img_wp, img_hp, 1))
Fp_train = np.reshape(Fp_train, (train_num, img_wp, img_hp, 1))
X_train = np.reshape(X_train, (train_num, img_wp, img_hp, len(b_values)))
ivim_train = np.concatenate([Dp_train, Dt_train, Fp_train], axis=-1)  # clear map
print(X_train.shape)

np.savez(train_gt_save_path, x=X_train, ivim=ivim_train)

# add noise
X_train_ = []
# p1 = np.float32([[0, 0], [192, 0], [0, 160], [160, 192]])
for idx, nosie_val in enumerate(np.linspace(min_n, max_n, train_num)):  # 0, 0.11
    dwi = X_train[idx] * s0  # [s0_idx]
    X_train_real = dwi + np.random.normal(0, nosie_val, X_train[idx].shape)
    X_train_imag = np.random.normal(0, nosie_val, X_train[idx].shape)
    dwi_nosiy = np.sqrt(X_train_real**2 + X_train_imag**2)  # / s0
    X_train_.append(dwi_nosiy / dwi_nosiy[..., :1])
X_train_ = np.array(X_train_)
print(X_train_.shape)
np.savez(train_save_path, x=X_train_, ivim=ivim_train)

# ****************************** simulate test ******************************
# area_num = 20
# remain = np.sqrt(img_w * img_w + img_h * img_h) / area_num
# # radius_list = np.linspace(0, remain, (82 * 82 + 98 * 98) )
# dp_list = np.linspace(dp_min, dp_max, area_num)
# dt_list = np.linspace(dt_min, dt_max, area_num)
# fp_list = np.linspace(fp_min, fp_max, area_num)
#
# sx, sy, sb = img_w, img_h, len(b_values)
# # create image
# X_test = np.zeros((1, sx, sy, sb))
# Dp_truth = np.zeros((1, sx, sy, 1))
# Dt_truth = np.zeros((1, sx, sy, 1))
# Fp_truth = np.zeros((1, sx, sy, 1))
#
# # area_len_x = sx // area_num  # (area_num * 2)
# # area_len_y = sy // area_num  # (area_num * 2)
# for i in range(sx):
#     for j in range(sy):
#         # xi, yi = i - 80, j - 96
#         # radius_idx = (xi * xi + yi * yi) // remain
#         radius_idx = int(np.sqrt(i * i + j * j) / remain)
#         if radius_idx >= area_num:
#             radius_idx = area_num - 1
#
#         X_test[0, i, j, :] = ivim_p2p(b_values, dp_list[radius_idx], dt_list[radius_idx], fp_list[radius_idx])
#         Dp_truth[0, i, j], Dt_truth[0, i, j], Fp_truth[0, i, j] = dp_list[radius_idx], dt_list[radius_idx], fp_list[radius_idx]
#
# ivim_test = np.concatenate([Dp_truth, Dt_truth, Fp_truth], axis=-1)
# print(ivim_test.shape)
#
# # plot ivim
# fig = plt.figure(figsize=(10, 6))
# grid = ImageGrid(fig, 111,
#                  nrows_ncols=(3, 1),
#                  direction='row',
#                  axes_pad=0.2,
#                  cbar_location='right',
#                  cbar_mode='edge',
#                  cbar_size='2%',
#                  cbar_pad=0.15)
#
# cp_Dp = grid[0].imshow(ivim_test[0][..., 0])  # , clim=(0, 0.1), cmap='gray'
# cp_Dt = grid[1].imshow(ivim_test[0][..., 1])  # , clim=(0, 0.002), cmap='gray'
# cp_Fp = grid[2].imshow(ivim_test[0][..., 2])  # , clim=(0, 0.4), cmap='gray'
# grid[0].cax.colorbar(cp_Dp)
# grid[1].cax.colorbar(cp_Dt)
# grid[2].cax.colorbar(cp_Fp)
#
# for i, axis in enumerate(grid):
#     axis.set_xticks([])
#     axis.set_yticks([])
#     if (i == 1) or (i == 2):
#         axis.set_axis_off()
#
# plt.tight_layout()
# plt.show()
#
# # plot gt
# # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
# fig, ax = plt.subplots(2, 4)  # , figsize=(20, 20)
# b_id = 1
# for i in range(2):
#     for j in range(4):
#         print(X_test[0, :, :, b_id].max(), X_test[0, :, :, b_id].mean(), X_test[0, :, :, b_id].min())
#         print(len(set(X_test[0][..., b_id].reshape((-1)))), set(X_test[0][..., b_id].reshape((-1))))
#         ax_ = ax[i, j].imshow(X_test[0, :, :, b_id], clim=(0, 1), cmap='gray')  #  cmap='gray'
#         ax[i, j].set_title('b = ' + str(b_values[b_id]))
#         ax[i, j].set_xticks([])
#         ax[i, j].set_yticks([])
#         b_id += 1
# fig.colorbar(ax_, ax=ax.ravel().tolist())
# plt.show()

# np.savez(test_save_path + '.npz', x=X_test, ivim=ivim_test)

# X_test_ = []
# rg = np.random.RandomState(456)
# for ii in range(len(nosie_vals)):
#     # # add some noise
#     # dwi = X_test * s0
#     # nosiy = np.random.normal(0, nosie_vals[ii], X_test[0].shape)
#     # X_test_real = dwi + nosiy  # + np.random.normal(scale=nosie_vals[ii], size=X_test.shape)
#     # X_test_imag = nosiy
#     # X_test_ = np.sqrt(X_test_real**2 + X_test_imag**2)  # / s0
#     # X_test_ = X_test_ / X_test_[..., :1]
#
#     nosiy = np.random.normal(scale=nosie_vals[ii], size=X_test.shape)
#     # X_test_.append((X_test + nosiy)[0])
#     X_test_ = X_test + nosiy
#
#     print('snr ', snr(X_test, X_test_))
#
#     # # plot noised
#     # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
#     fig, ax = plt.subplots(2, 4)  # , figsize=(20, 20)
#     b_id = 1
#     for i in range(2):
#         for j in range(4):
#             print(X_test_[0][:, :, b_id].max(), X_test_[0][:, :, b_id].mean(), X_test_[0][:, :, b_id].min())
#             # print(len(set(X_test_[0][..., b_id].reshape((-1)))), set(X_test_[0][..., b_id].reshape((-1))))
#             ax_ = ax[i, j].imshow(X_test_[0][:, :, b_id], clim=(0, 1), cmap='gray')  # cmap='gray'
#             ax[i, j].set_title('b = ' + str(b_values[b_id]))
#             ax[i, j].set_xticks([])
#             ax[i, j].set_yticks([])
#             b_id += 1
#     fig.colorbar(ax_, ax=ax.ravel().tolist())
#     plt.show()

#     np.savez(test_save_path + str(ii) + '.npz', x_gt=X_test, x=x_test_, ivim=ivim_test)

Dp_train = np.random.uniform(dp_min, dp_max, (test_num * img_w * img_h, 1))  # + 0.4
Dt_train = np.random.uniform(dt_min, dt_max, (test_num * img_w * img_h, 1))  # + 0.001
Fp_train = np.random.uniform(fp_min, fp_max, (test_num * img_w * img_h, 1))  # + 0.1
b_values_ = np.expand_dims(b_values, 0)
X_test = ivim(b_values_, Dp_train, Dt_train, Fp_train)  # range 0 - 1

Dp_train = np.reshape(Dp_train, (test_num, img_w, img_h, 1))
Dt_train = np.reshape(Dt_train, (test_num, img_w, img_h, 1))
Fp_train = np.reshape(Fp_train, (test_num, img_w, img_h, 1))
X_test = np.reshape(X_test, (test_num, img_w, img_h, len(b_values)))
ivim_test = np.concatenate([Dp_train, Dt_train, Fp_train], axis=-1)  # clear map
print(X_test.shape)

# add noise
x_test_ = []
p1 = np.float32([[0, 0], [192, 0], [0, 160], [160, 192]])
for idx, nosie_val in enumerate(np.linspace(min_n, max_n, test_num)):  # 0, 0.11
    # for ii in range(1, X_test.shape[-1]):
    #     deformation = np.random.uniform(-df_list_test[idx], df_list_test[idx], p1.shape).astype(np.float32)
    #     p2 = p1 + deformation
    #     M = cv2.getPerspectiveTransform(p1, p2)
    #     X_test[idx][..., ii] = cv2.warpPerspective(X_test[idx][..., ii], M, (img_h, img_w), borderMode=1)
    dwi = X_test[idx] * s0
    # nosiy = np.random.normal(0, nosie_val, X_test[idx].shape)
    X_test_real = dwi + np.random.normal(0, nosie_val, X_test[idx].shape)
    X_test_imag = np.random.normal(0, nosie_val, X_test[idx].shape)
    dwi_nosiy = np.sqrt(X_test_real**2 + X_test_imag**2)  # / s0
    x_test_.append(dwi_nosiy / dwi_nosiy[..., :1])
# for idx, noise_val in enumerate(np.linspace(0.0, 0.11, train_num)):
#     nosiy = np.random.normal(scale=noise_val, size=X_train[idx].shape)
#     # X_train_ = X_train + nosiy
#     X_train_.append(X_train[idx] + nosiy)
x_test_ = np.array(x_test_)
print(x_test_.shape)

np.savez(test_save_path, x_gt=X_test, x=x_test_, ivim=ivim_test)
