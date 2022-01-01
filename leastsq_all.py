import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
from os.path import join, exists
from glob import glob
import multiprocessing as mp
import copy

if __name__ == "__main__":
    def ivim_matmul(params, b_v):
        shape = params.shape
        flat = np.reshape(params, (shape[0] * shape[1] * shape[2], 3))  #
        dp = np.expand_dims(flat[..., 0], -1)
        dt = np.expand_dims(flat[..., 1], -1)
        fp = np.expand_dims(flat[..., 2], -1)
        b_v = np.expand_dims(b_v, 0)
        outputs = fp * np.exp(-np.matmul(dp, b_v)) + (1 - fp) * np.exp(-np.matmul(dt, b_v))
        outputs = np.reshape(outputs, (shape[0], shape[1], shape[2], b_v.shape[1]))
        return outputs


    def ivim(b, Dp, Dt, Fp):
        return Fp*np.exp(-b*Dp) + (1-Fp)*np.exp(-b*Dt)


    def order(Dp_, Dt_, Fp_):
        if Dp_ < Dt_:
            temp = copy.deepcopy(Dp_)
            Dp_ = copy.deepcopy(Dt_)
            Dt_ = temp
            Fp_ = 1-Fp_
        return Dp_, Dt_, Fp_


    def fit_segmented(b, x_dw):
      try:
        high_b = b[b>=250]
        high_x_dw = x_dw[b>=250]
        bounds = (0, 1)
        # bounds = ([0, 0.4], [0.005, 1])
        params, _ = curve_fit(lambda high_b, Dt, int : int*np.exp(-high_b*Dt), high_b, high_x_dw, p0=(0.001, 0.9), bounds=bounds)
        Dt, Fp = params[0], 1-params[1]
        x_dw_remaining = x_dw - (1-Fp)*np.exp(-b*Dt)
        bounds = (0, 1)
        # bounds = (0.01, 0.3)
        params, _ = curve_fit(lambda b, Dp : Fp*np.exp(-b*Dp), b, x_dw_remaining, p0=(0.01), bounds=bounds)
        Dp = params[0]
        return order(Dp, Dt, Fp)
      except:
        return 0., 0., 0.


    def fit_least_squares(b, x_dw):
      try:
        # bounds = (0, 1)
        bounds = ([0.01, 0, 0], [0.3, 0.005, 0.6])
        params, _ = curve_fit(ivim, b, x_dw, p0=[0.01, 0.001, 0.1], bounds=bounds)
        Dp_, Dt_, Fp_ = params[0], params[1], params[2]
        return order(Dp_, Dt_, Fp_)
      except:
        return fit_segmented(b, x_dw)


    def handle_img(low, high, ivim_p):
        # print(low // pix_num, ' init')
        ivim_ = np.reshape(ivim_p, (-1, 3))
        if high > x_fit.shape[0]:
            high = x_fit.shape[0]
        # print(low // pix_num, ' start')
        for ii in range(low, high):  # x_fit.shape[0]
            # if x_fit[ii, 0] == 0:
            #     continue
            try:
                ivim_[ii-low] = fit_least_squares(b_fit, x_fit[ii, :])
                # ivim_[ii - low] = np.ones_like(ivim_[ii-low])
            except:
                continue
        ivim_ = np.reshape(ivim_, (-1))
        for ii in range(ivim_.shape[0]):
            ivim_p[ii] = ivim_[ii]

    process_num = 45  # must small than cpu num
    # define b values
    b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000])
    # b_fit, b_no_fit = get_fit_and_no_fit(b_values, no_fit_idx)
    b_fit = b_values[1:]
    # print(b_fit, b_no_fit)

    save_path = r'/home/public/Documents/hhy/data/IVIM6-1/simulate_data/leastsq'
    if not exists(save_path):
        os.makedirs(save_path)
    b_path = r'/home/public/Documents/hhy/data/IVIM6-1/real_threshold2'
    b_files = glob(join(b_path, '*'))
    b_files.sort(key=lambda x: (int(x.split('/')[-1].split('.')[0])))
    print(len(b_files))

    for idx, bf in enumerate(b_files):
        print('processing ', idx)
        save_name = bf.split('/')[-1]
        # if exists(join(save_path, save_name)):
        #     continue
        test_data = np.load(bf)
        X_test = test_data['x'][::2]
        x_fit = X_test[..., 1:]
        print(x_fit.shape)

        pixel_num = x_fit.shape[0] * x_fit.shape[1] * x_fit.shape[2]
        x_fit = np.reshape(x_fit, (pixel_num, -1))
        ivim_pre_ = np.zeros((pixel_num, 3))
        pix_num = int(np.ceil(pixel_num * 1.0 / process_num))
        ivim_pre = [mp.Array('f', np.reshape(ivim_pre_[pn: pn+pix_num], (-1))) for pn in range(0, pixel_num, pix_num)]
        step_time = time.time()
        process = [mp.Process(target=handle_img,
                              args=(pn, pn+pix_num, ivim_pre[pn//pix_num],)) for pn in range(0, ivim_pre_.shape[0], pix_num)]
        [p.start() for p in process]
        [p.join() for p in process]

        time_per_iter = time.time() - step_time
        time_per_pixel = time_per_iter / (X_test.shape[0] * X_test.shape[1] * X_test.shape[2])
        print('time per pixel ', time_per_pixel)
        x_fit = np.reshape(x_fit, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 8))
        ivim_pre = np.reshape(ivim_pre, (X_test.shape[0], x_fit.shape[1], x_fit.shape[2], 3))
        # for i in range(3):
        #     # ql = np.percentile(ivim_pre[..., i], 1)
        #     qh = np.percentile(ivim_pre[..., i], 95)
        #     ivim_pre[..., i] = np.clip(ivim_pre[..., i], 0.0, qh)

        print(ivim_pre[17][..., 0].max(), ' ', ivim_pre[17][..., 0].mean(), ' ', ivim_pre[17][..., 0].min())
        print(ivim_pre[17][..., 1].max(), ' ', ivim_pre[17][..., 1].mean(), ' ', ivim_pre[17][..., 1].min())
        print(ivim_pre[17][..., 2].max(), ' ', ivim_pre[17][..., 2].mean(), ' ', ivim_pre[17][..., 2].min())
        print()

        # x_no_fit_pre = ivim_matmul(ivim_pre, b_no_fit)
        x_fit_pre = np.array(ivim_matmul(ivim_pre, b_fit))
        # # mask = np.ones((x_fit.shape[0], x_fit.shape[1], 1))
        # # indice = x_fit[..., 0] == 0
        # mask = np.ones((x_fit.shape[0], x_fit.shape[1], x_fit.shape[2], 1))
        # indice = x_fit[..., 0] == 0
        # mask[indice] = 0
        # # x_no_fit_pre = x_no_fit_pre * mask
        # x_fit_pre = x_fit_pre * mask
        print(x_fit_pre.shape)

        # plt.subplot(131)
        # plt.imshow(ivim_pre[17][..., 0], cmap='gray')  # , clim=(0, 1)
        # plt.colorbar()
        # plt.axis('off')
        # plt.subplot(132)
        # plt.imshow(ivim_pre[17][..., 1], cmap='gray')  # , clim=(0, 0.002)
        # plt.colorbar()
        # plt.axis('off')
        # plt.subplot(133)
        # plt.imshow(ivim_pre[17][..., 2], cmap='gray')  # , clim=(0, 1)
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()
        #
        # plt.subplot(131)
        # plt.imshow(x_fit_pre[17][..., 0], cmap='gray')  # , clim=(0, 1)
        # plt.colorbar()
        # plt.axis('off')
        # plt.subplot(132)
        # plt.imshow(x_fit_pre[17][..., 1], cmap='gray')  # , clim=(0, 0.002)
        # plt.colorbar()
        # plt.axis('off')
        # plt.subplot(133)
        # plt.imshow(x_fit_pre[17][..., 2], cmap='gray')  # , clim=(0, 1)
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()

        np.savez(join(join(save_path, save_name)), ivim=ivim_pre, x=x_fit_pre)
