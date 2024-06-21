import os

import numpy as np


import tensorflow as tf
from ops import *
import matplotlib.pyplot as plt
import time
import copy
from os.path import exists

# no_fit_num = 1  # 1 - 8

img_w = 32  # 160
img_h = 32  # 192

# define b values
b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000])
# b_values_no0 = b_values[1:]
# b_fit, b_no_fit = get_fit_and_no_fit(b_values, no_fit_num)
b_fit = b_values[1:]

num_epochs = 100  # 50
batch_size = 8  # 8
patience = 10  # 5

learning_rate = 1e-4  # 1e-4

train_data = np.load(r'/home/public/Documents/hhy/data/IVIM6-1/simulate_data/train1.npz')
X_train = train_data['x']
ivim_train = train_data['ivim']
# ivim_train = train_data['ivim']
print(X_train.shape)
# x_fit, x_no_fit = get_fit_and_no_fit(X_train, no_fit_num)
x_fit = X_train[..., 1:]

test_save_path = r'/home/public/Documents/hhy/IVIM/UDA_fake/ANN'
if not exists(test_save_path):
    os.makedirs(test_save_path)
X_test = np.load(r'/home/public/Documents/hhy/data/IVIM6-1/simulate_data/test1.npz')['x']
print(X_test.shape)

# # train_num = 50
save_path_real = r'/home/public/Documents/hhy/IVIM/UDA_real/ANN'
if not exists(save_path_real):
    os.makedirs(save_path_real)
b_path = r'/home/public/Documents/hhy/data/IVIM6-1/real_threshold2'
b_files = glob(join(b_path, '*'))
b_files = [bf for bf in b_files if 'size' not in bf]
b_files.sort(key=lambda x: (int(x.split('/')[-1].split('.')[0])))
# x_fit = np.concatenate([np.load(bf)['x'][::6][..., 1:] for bf in b_files], 0)
# x_fit = np.concatenate([patch(xf, 32, 32) for xf in x_fit], 0)
train_data = [[np.load(bf)['x'][17]] for bf in b_files]
X_test_real = np.concatenate(train_data, 0)
print(x_fit.shape)

num_samples = x_fit.shape[0]
num_batches = int(num_samples / batch_size)
# num_batches = 188  # 200


def model(inputs, b_v, reuse=False, mask_=None):
    w_init = tf.random_normal_initializer(stddev=0.059)  # 0.1
    b_init = tf.constant_initializer(value=0.0)
    # b_init = tf.random_uniform_initializer(-0.005, 0.005)
    # g_init = tf.random_normal_initializer(1., 0.02)

    # filters = 64

    # for iter_ in range(iter):
    with tf.variable_scope("fit", reuse=reuse):
        inputs = tf.reshape(inputs, (-1, inputs.shape[-1]))
        net = dense(inputs, 10, w_init=w_init, b_init=b_init, name='d1', act=tf.nn.elu)
        # net = dense(net, 8, w_init=w_init, b_init=b_init, name='d2', act=tf.nn.elu)
        # net = dense(net, 8, w_init=w_init, b_init=b_init, name='d3', act=tf.nn.elu)
        params = dense(net, 3, w_init=tf.random_normal_initializer(stddev=0.177), b_init=b_init, name='d4')  # 289
        params = tf.abs(params)

        outputs = ivim_matmul(params, b_v)
        params = tf.reshape(params, (-1, img_w, img_h, 3))
        # if mask_ is not None:
        #     outputs = tf.multiply(outputs, mask_)
        #     params = tf.multiply(params, mask_)

    return outputs, params  # , params_outputs


def ivim_matmul(params, b_v):
    flat = tf.reshape(params, (-1, img_w * img_h, 3))  #
    dp = tf.expand_dims(flat[..., 0], -1)
    dt = tf.expand_dims(flat[..., 1], -1)
    fp = tf.expand_dims(flat[..., 2], -1)
    b_v = tf.expand_dims(b_v, 0)
    outputs = fp * tf.exp(-tf.matmul(dp, b_v)) + (1 - fp) * tf.exp(-tf.matmul(dt, b_v))
    outputs = tf.reshape(outputs, (-1, img_w, img_h, b_v.shape[1]))
    return outputs


tf.reset_default_graph()
x_fit_ = tf.placeholder(dtype=tf.float32, shape=(None, img_w, img_h, 8))
# x_no_fit_ = tf.placeholder(dtype=tf.float32, shape=(None, img_w, img_h, 1))
ivim_ = tf.placeholder(dtype=tf.float32, shape=(None, img_w, img_h, 3))
# x_gt = tf.placeholder(dtype=tf.float32, shape=(None, img_w, img_h, 5))
# mask = tf.placeholder(dtype=tf.float32, shape=(None, img_w, img_h, 1))
# b_values_ = tf.placeholder(dtype=tf.float32, shape=None)
b_values_fit = tf.constant(b_fit, dtype=tf.float32)
# b_values_nofit = tf.constant(b_no_fit, dtype=tf.float32)

x_pre_fit, ivim_pre = model(x_fit_, b_values_fit)  # , mask_=mask
# _, _, ivim_pre = model(x_low, b_values_low, 2, reuse=True)
# x_pre_nofit = ivim_matmul(ivim_pre, b_values_nofit)

# loss_l2 = tf.losses.mean_squared_error(x_pre_fit, x_fit_)
# loss_l2 = tf.reduce_mean(tf.abs(x_pre_fit - x_fit_))
loss_l2 = tf.reduce_mean(tf.abs(ivim_pre - ivim_))

saver = tf.train.Saver()

trainable_vars = tf.trainable_variables()
var_f = [v for v in trainable_vars if 'fit' in v.name]

global_step_f = tf.Variable(0, trainable=False, name='global_step_f')
# decayed_learning_rate_f = tf.train.exponential_decay(
#     learning_rate=learning_rate,
#     global_step=global_step_f,
#     decay_steps=num_batches,  # max(num_epochs * num_batches / 2, 1), 200
#     decay_rate=.99,  # .95
#     staircase=True
# )

optimizer_f = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999).minimize(
    loss_l2, var_list=var_f, global_step=global_step_f)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

iter_time = []
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()

    idx = np.arange(num_samples)
    current_eta = None
    best = 1e16
    num_bad_epochs = 0
    for epoch in range(num_epochs):
        np.random.shuffle(idx)
        running_loss = 0.
        # x_fit, ivim_train = simulate_data(img_w, img_h, batch_size * num_batches, b_fit)
        for n_batch in range(num_batches):
            step_time = time.time()
            sub_idx = idx[n_batch * batch_size:n_batch * batch_size + batch_size]
            batch_fit = x_fit[sub_idx]
            # batch_nofit = x_no_fit[sub_idx]
            batch_ivim = ivim_train[sub_idx]
            # batch_gt = x_low_b_gt[sub_idx]
            # batch_mask = np.ones_like(batch_fit[..., :1])
            # batch_mask[batch_fit[..., :1] == 0] = 0

            # batch_fit, batch_ivim = simulate_data(img_w, img_h, batch_size, b_fit)
            # strat_idx = n_batch * batch_size
            # batch_fit, batch_ivim = x_fit[strat_idx: strat_idx+batch_size], ivim_train[strat_idx: strat_idx+batch_size]

            _, l2, ivim_p = sess.run(
                fetches=[optimizer_f, loss_l2, ivim_pre],
                feed_dict={x_fit_: batch_fit, ivim_: batch_ivim}  # , x_gt: batch_gt, mask: batch_mask
            )

            time_per_iter = time.time() - step_time
            n_iter_remain = (num_epochs - epoch - 1) * num_batches + num_batches - n_batch
            eta_str, eta_ = eta(time_per_iter, n_iter_remain, current_eta)
            current_eta = eta_
            running_loss += l2

            if (n_batch + 1) % int(num_batches) == 0:
                print('Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\tloss %.6f\n'
                      '\ttrain:\tmse = %.8f\tdp = %.5f\tdt = %.4f\tfp = %.4f\tbad epochs = %d\n' %
                      (epoch + 1, num_epochs, n_batch + 1, num_batches, eta_str, learning_rate,
                       running_loss, ivim_p[..., 0].mean(), ivim_p[..., 1].mean(), ivim_p[..., 2].mean(), num_bad_epochs))

        # early stopping
        if running_loss < best:
            saver.save(sess, "Model/model.ckpt")
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best))
                break

    saver.restore(sess, "./Model/model.ckpt")

    x_fit_pre_all, ivim_pre_all = [], []
    for idxt, x_test in enumerate(X_test):
        # x_test_fit, x_test_nofit = get_fit_and_no_fit(x_test, no_fit_num)
        x_test_fit = x_test[..., 1:]

        # x_test_fit, x_test_nofit = patch(x_test_fit, img_w, img_h), patch(x_test_nofit, img_w, img_h)
        x_test_fit = patch(x_test_fit, img_w, img_h)
        ivim_pre_, x_fit_pre = sess.run(
            fetches=[ivim_pre, x_pre_fit],
            feed_dict={x_fit_: x_test_fit}  # , mask: np.ones_like(x_test_fit[..., :1])
        )

        # make sure Dp is the larger value between Dp and Dt
        if ivim_pre_[..., 0].mean() < ivim_pre_[..., 1].mean():
            print('sawp')
            temp = copy.deepcopy(ivim_pre_[..., 0])
            ivim_pre_[..., 0] = copy.deepcopy(ivim_pre_[..., 1])
            ivim_pre_[..., 1] = temp
            ivim_pre_[..., 2] = 1 - ivim_pre_[..., 2]

        x_fit_pre_all.append(unpatch(x_fit_pre, 160, 192))
        ivim_pre_all.append(unpatch(ivim_pre_, 160, 192))

    np.savez(join(test_save_path, 'test.npz'), ivim=ivim_pre_all, x_fit_pre=x_fit_pre_all)

    for idxt, x_test in enumerate(X_test_real):
        # x_test_fit, x_test_nofit = get_fit_and_no_fit(x_test, no_fit_num)
        x_test_fit = x_test[..., 1:]

        # x_test_fit, x_test_nofit = patch(x_test_fit, img_w, img_h), patch(x_test_nofit, img_w, img_h)
        x_test_fit = patch(x_test_fit, img_w, img_h)

        # real_mask = np.ones_like(x_test_fit[..., :1])
        # real_mask[x_test_fit[..., :1] == 0] = 0
        ivim_pre_, x_fit_pre = sess.run(
            fetches=[ivim_pre, x_pre_fit],
            feed_dict={x_fit_: x_test_fit}
        )

        # make sure Dp is the larger value between Dp and Dt
        if ivim_pre_[..., 0].mean() < ivim_pre_[..., 1].mean():
            print('sawp')
            temp = copy.deepcopy(ivim_pre_[..., 0])
            ivim_pre_[..., 0] = copy.deepcopy(ivim_pre_[..., 1])
            ivim_pre_[..., 1] = temp
            ivim_pre_[..., 2] = 1 - ivim_pre_[..., 2]

        # x_fit_pre = x_fit_pre * real_mask
        # ivim_pre_ = ivim_pre_ * real_mask

        x_fit_pre = unpatch(x_fit_pre, 160, 192)
        ivim_pre_ = unpatch(ivim_pre_, 160, 192)

        np.savez(join(save_path_real, b_files[idxt].split('/')[-1]), ivim=ivim_pre_, x_fit_pre=x_fit_pre)

    sess.close()
