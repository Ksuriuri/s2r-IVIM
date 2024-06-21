import os
import matplotlib.pyplot as plt
from glob import glob
from os.path import join, exists
import numpy as np
import SimpleITK
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2

save_path = r'/home/public/Documents/hhy/data/IVIM6-1/real_threshold2'  # real
if not exists(save_path):
    os.makedirs(save_path)
dir_path = r'/home/public/Documents/hhy/data/IVIM6-1/HCC'
dir_list = glob(join(dir_path, '*'))  # all patients path
dir_list.sort(key=lambda x: (int(x.split('/')[-1].split(' ')[0])))
handle_num = len(dir_list)
w, h = 160, 192  # 168, 210
# 36, 38, 40, 42, 45, 46, 48
slice = 36
b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000])
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 18}

for dl in dir_list[:handle_num]:
    idx_dl = dl.split('/')[-1].split(' ')[0]
    # if exists(join(save_path, idx_dl) + '.npz'):
    #     continue
    file_list = os.listdir(dl)
    b_file_list = []
    for fl in file_list:
        if fl.split('.')[-1] == 'mha' and (('DWI' in fl) or ('DW' in fl) or ('dw' in fl)):
            b_file_list.append(fl)
    if len(b_file_list) == 10:
        continue
    b_file_list.sort(key=lambda x: (int(x.split('.')[0][-1])), reverse=True)
    b_img, b0 = [], [
    iscontinue = False
    for idx, bfl in enumerate(b_file_list):
        img = SimpleITK.ReadImage(join(dl, bfl))
        img_data = SimpleITK.GetArrayFromImage(img)
        edge_w, edge_h = (img_data.shape[1] - w) // 2, (img_data.shape[2] - h) // 2
        img = np.expand_dims(img_data[:, edge_w: edge_w + w, edge_h: edge_h + h], -1)
        if img[0].shape != (w, h, 1) or img.shape[0] < slice:
            iscontinue = True
            break
        img = img[:36]
        indice = img <= np.percentile(img, 5)
        img[indice] = 0.0
        indice = img < 1.0
        img[indice] = 0.0
        # mask[indice] = 0
        # print('%.4f %.4f %.4f' % (img.max(), img.mean(), img.min()))
        if idx == 0:
            # ql = np.percentile(img, 5)
            qh = np.percentile(img, 95)
            img = np.clip(img, 0.0, qh)
            b_img.append(img)
            # b0 = img_data.max()
            img[img < 1.0] = 1.0
            b0 = img
            print(idx_dl, img.shape)
            # break
        else:
            img = img / b0
            img = np.clip(img, 0.0, 1.0)
            b_img.append(img)
        print('%.4f %.4f %.4f' % (img.max(), img.mean(), img.min()))

    if iscontinue:
        continue
    # b_img = b_img * mask
    print()
    if int(idx_dl) < 10:
        fig = plt.figure(figsize=(14, 2))  #
        plt.axis('off')
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, b_values.shape[-1]),
                         direction='row',
                         axes_pad=0.2,
                         cbar_location='right',
                         cbar_mode='edge',
                         cbar_size='5%',
                         cbar_pad=0.15)

        for jj in range(b_values.shape[-1]):
            cp = grid[jj].imshow(b_img[jj][17], cmap='gray', clim=(0, 1))
            # if jj == 0:
            #     grid[0].set_ylabel(name_list[ii])
            grid[jj].set_xlabel('b=' + str(b_values[jj]), font1)
        grid[b_values.shape[-1]-1].cax.colorbar(cp)

        for ii, axis in enumerate(grid):
            axis.set_xticks([])
            axis.set_yticks([])
            # if (i == 1) or (i == 2):
            #     axis.set_axis_off()

        plt.tight_layout()
        plt.show()

    np.savez(join(save_path, idx_dl), x=np.concatenate(b_img, -1))  # (36, 168, 210, 9)
