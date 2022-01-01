import os
import numpy as np
import tensorflow as tf
from ops import *
import matplotlib.pyplot as plt
import time
import copy
from os.path import exists
# from vgg19 import *
from mpl_toolkits.axes_grid1 import ImageGrid

# no_fit_num = 1  # 1 - 8

img_w = 160  # 160 32
img_h = 192  # 192 32

# define b values
b_values = np.array([0, 10, 20, 40, 80, 200, 400, 600, 1000])
b_fit = b_values[1:]
# b_values = np.array([10, 20, 40, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 2400, 2800])
# b_fit = b_values

num_epochs = 100  # 120
batch_size = 2  # 2
num_pre_epochs = 20
pre_batch_size = 64  # 2
patience = 100  # 5

# learning_rate = 1e-3  # 1e-4
learning_rate_d = 3e-4
# learning_rate_g = 1e-4
# learning_rate_f = 1e-4
learning_rate_e = 3e-4  # 1e-4

save_model_path = 'single'

sl_num = '3'

train_data = np.load(r'/home/public/Documents/hhy/data/IVIM6-1/simulate_data/train%s.npz' % sl_num)
X_train = train_data['x']
ivim_train = train_data['ivim']
# ivim_train = np.array([unpatch(ivim_train[ii: ii+30], img_w, img_h) for ii in range(0, ivim_train.shape[0], 30)])
ivim_train[..., [0, 1]] = ivim_train[..., [1, 0]]
ivim_train[..., 2] = 1 - ivim_train[..., 2]
print(X_train.shape)
x_fit_fake = X_train[..., 1:]
# x_fit_fake = np.array([unpatch(x_fit_fake[ii: ii+30], img_w, img_h) for ii in range(0, x_fit_fake.shape[0], 30)])
print(x_fit_fake.shape)

# train_data = np.load(r'/home/public/Documents/hhy/data/IVIM6-1/simulate_data/train_gt%s.npz' % sl_num)
# X_train = train_data['x']
# print(X_train.shape)
# x_fit_gt = X_train[..., 1:]
# x_fit_gt = np.array([unpatch(x_fit_gt[ii: ii+30], img_w, img_h) for ii in range(0, x_fit_gt.shape[0], 30)])

# test_save_path = r'/home/public/Documents/hhy/IVIM/UDA_fake/CNN_UDA'
# if not exists(test_save_path):
#     os.makedirs(test_save_path)
# X_test = np.load(r'/home/public/Documents/hhy/data/IVIM6-1/simulate_data/test%s.npz' % sl_num)['x']
# print(X_test.shape)

train_num = 73
save_path_real = r'/home/public/Documents/hhy/IVIM/UDA_real/CNN_GAN%s_3' % sl_num  # _UDA_noGAN
if not exists(save_path_real):
    os.makedirs(save_path_real)
b_path = r'/home/public/Documents/hhy/data/IVIM6-1/real_threshold2'
b_files = glob(join(b_path, '*'))
b_files = [bf for bf in b_files if 'size' not in bf]
b_files.sort(key=lambda x: (int(x.split('/')[-1].split('.')[0])))
np.random.seed(2021)
np.random.shuffle(b_files)
x_fit_real = np.concatenate([np.load(bf)['x'][::2][..., 1:] for bf in b_files], 0)  # [:train_num]
print(x_fit_real.shape)
X_test_real = np.concatenate([[np.load(bf)['x'][17][..., 1:]] for bf in b_files], 0)  # [train_num:]
print(X_test_real.shape)

num_samples = x_fit_fake.shape[0]
num_batches = int(num_samples / batch_size)
num_pre_batches = int(num_samples / pre_batch_size)
# num_batches = 188  # 200


def feature_extractor(inputs, reuse=False):
    filters = 16  # 32

    with tf.variable_scope("feature_extractor", reuse=reuse):
        endpoints = {}
        conv = convolutional(inputs, [3, 3, inputs.shape[-1], filters], name='conv1')  # , activate=False
        # conv = tf.nn.relu(batch_norm(conv, name='bn1', _ops=[]))
        conv = convolutional(conv, [3, 3, filters, filters], name='conv2')  # , activate=False
        # conv = tf.nn.relu(batch_norm(conv, name='bn2', _ops=[]))
        print(conv.shape)
        endpoints['C1'] = conv
        # downsample 1
        conv = convolutional(conv, [3, 3, filters, filters], name='conv3', downsample=True)
        conv = convolutional(conv, [3, 3, filters, filters * 2], name='conv4')  # , activate=False
        # conv = tf.nn.relu(batch_norm(conv, name='bn4', _ops=[]))
        conv = convolutional(conv, [3, 3, filters * 2, filters * 2], name='conv5')  # , activate=False
        # conv = tf.nn.relu(batch_norm(conv, name='bn5', _ops=[]))
        print(conv.shape)
        endpoints['C2'] = conv
        # downsample 2
        conv = convolutional(conv, [3, 3, filters * 2, filters * 2], name='conv6', downsample=True)
        conv = convolutional(conv, [3, 3, filters * 2, filters * 4], name='conv7')  # , activate=False
        # conv = tf.nn.relu(batch_norm(conv, name='bn7', _ops=[]))
        conv = convolutional(conv, [3, 3, filters * 4, filters * 4], name='conv8')  # , activate=False
        # conv = tf.nn.relu(batch_norm(conv, name='bn8', _ops=[]))
        print(conv.shape)
        # endpoints['C3'] = conv
        # # downsample 3
        # conv = convolutional(conv, [3, 3, filters * 4, filters * 4], name='conv9', downsample=True)
        # conv = convolutional(conv, [3, 3, filters * 4, filters * 8], name='conv10')
        # conv = convolutional(conv, [3, 3, filters * 8, filters * 8], name='conv11')
        # print(conv.shape)
        # endpoints['C4'] = conv
        # # downsample 3
        # conv = convolutional(conv, [3, 3, filters * 8, filters * 8], name='conv12', downsample=True)
        # conv = convolutional(conv, [3, 3, filters * 8, filters * 16], name='conv13')
        # conv = convolutional(conv, [3, 3, filters * 16, filters * 16], name='conv14')
        # print(conv.shape)
    return conv, endpoints  # , params_outputs


def estimator(conv, endpoints, b_v, reuse=False):
    with tf.variable_scope("estimator", reuse=reuse):
        for i in range(2, 0, -1):
            with tf.variable_scope('Ronghe%d' % i):
                uplayer = upsample(conv, endpoints['C%d' % i].shape[1:3], 'deconv%d' % (3 - i), method="deconv")
                print(uplayer.shape)
                concat = tf.concat([endpoints['C%d' % i], uplayer], axis=-1)
                dim = concat.get_shape()[-1].value
                conv = convolutional(concat, [3, 3, dim, dim // 2], name='conv1')  # , activate=False
                # conv = tf.nn.relu(batch_norm(conv, name='bn1', _ops=[]))
                conv = convolutional(conv, [3, 3, dim // 2, dim // 2], name='conv2')  # , activate=False
                # conv = tf.nn.relu(batch_norm(conv, name='bn2', _ops=[]))
        params = convolutional(conv, [3, 3, dim // 2, 3], name='params', activate=False)
        params = tf.abs(params)

        # if mask_ is not None:
        #     params = tf.multiply(params, mask_)
        outputs = ivim_matmul(params, b_v)
        # outputs = reconstruction(params, reuse)
        # if mask_ is not None:
        #     outputs = tf.multiply(outputs, mask_)
    return outputs, params


# def generator(conv, endpoints, reuse=False):
#     with tf.variable_scope("generator", reuse=reuse):
#         for i in range(3, 0, -1):
#             with tf.variable_scope('Ronghe%d' % i):
#                 uplayer = upsample(conv, endpoints['C%d' % i].shape[1:3], 'deconv%d' % (3 - i), method="deconv")
#                 print(uplayer.shape)
#                 concat = tf.concat([endpoints['C%d' % i], uplayer], axis=-1)
#                 dim = concat.get_shape()[-1].value
#                 conv = convolutional(concat, [3, 3, dim, dim // 2], name='conv1')
#                 conv = convolutional(conv, [3, 3, dim // 2, dim // 2], name='conv2')
#         outputs = convolutional(conv, [3, 3, dim // 2, 8], name='params', activate=False)
#     return outputs
#
#
def discriminator(inputs, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)  # 0.1
    b_init = tf.constant_initializer(value=0.0)
    with tf.variable_scope('discriminator', reuse=reuse):
        # # net = conv2d(inputs, 512, filter_size=(3, 3), strides=(2, 2), w_init=w_init,
        # #              b_init=b_init, padding='SAME', name='conv1', act=tf.nn.leaky_relu)
        # net = conv2d(inputs, 16, filter_size=(4, 4), strides=(2, 2), w_init=w_init,
        #              b_init=b_init, padding='SAME', name='conv2')  # , act=tf.nn.leaky_relu
        # # net = batch_norm(net, name='bn2', _ops=[])
        # net = tf.nn.leaky_relu(net, name='lrelu2')
        #
        # net = conv2d(net, 32, filter_size=(4, 4), strides=(2, 2), w_init=w_init,
        #              b_init=b_init, padding='SAME', name='conv3')  # , act=tf.nn.leaky_relu
        # # net = batch_norm(net, name='bn3', _ops=[])
        # net = tf.nn.leaky_relu(net, name='lrelu3')
        #
        # net = conv2d(net, 64, filter_size=(4, 4), strides=(2, 2), w_init=w_init,
        #              b_init=b_init, padding='SAME', name='conv4')  # , act=tf.nn.leaky_relu
        # # net = batch_norm(net, name='bn4', _ops=[])
        # net = tf.nn.leaky_relu(net, name='lrelu4')
        #
        # net = conv2d(net, 128, filter_size=(4, 4), strides=(2, 2), w_init=w_init,
        #              b_init=b_init, padding='SAME', name='conv5')  # , act=tf.nn.leaky_relu
        # # net = batch_norm(net, name='bn5', _ops=[])
        # net = tf.nn.leaky_relu(net, name='lrelu5')
        #
        # outputs = conv2d(net, 1, filter_size=(1, 1), strides=(1, 1), w_init=w_init,
        #                  b_init=b_init, padding='SAME', name='outputs')  # , act=tf.nn.sigmoid
        # # outputs = tf.reduce_mean(outputs, [1, 2])

        # # flat = flatten(net, name='flat')
        # # # dense1 = dense(flat, n_units=128, act=tf.nn.relu, name='dense1')
        # # outputs = dense(flat, n_units=2, name='dense2')  # , act=tf.nn.sigmoid

        net = conv2d(inputs, 16, filter_size=(1, 1), strides=(1, 1), w_init=w_init,
                     b_init=b_init, padding='SAME', name='conv2')  # , act=tf.nn.leaky_relu
        net = tf.nn.leaky_relu(net, name='lrelu2')

        net = conv2d(net, 16, filter_size=(1, 1), strides=(1, 1), w_init=w_init,
                     b_init=b_init, padding='SAME', name='conv3')  # , act=tf.nn.leaky_relu
        net = tf.nn.leaky_relu(net, name='lrelu3')

        net = conv2d(net, 16, filter_size=(1, 1), strides=(1, 1), w_init=w_init,
                     b_init=b_init, padding='SAME', name='conv4')  # , act=tf.nn.leaky_relu
        net = tf.nn.leaky_relu(net, name='lrelu4')

        outputs = conv2d(net, 1, filter_size=(1, 1), strides=(1, 1), w_init=w_init,
                         b_init=b_init, padding='SAME', name='outputs')  # , act=tf.nn.sigmoid

    return tf.nn.sigmoid(outputs), outputs, net


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
tg_real_ = tf.placeholder(dtype=tf.float32, shape=(None, img_w, img_h, 8))
src_real_ = tf.placeholder(dtype=tf.float32, shape=(None, img_w, img_h, 8))
# src_gt_ = tf.placeholder(dtype=tf.float32, shape=(None, img_w, img_h, 8))
src_ivim_ = tf.placeholder(dtype=tf.float32, shape=(None, img_w, img_h, 3))
# mask = tf.placeholder(dtype=tf.float32, shape=(None, img_w, img_h, 1))
b_values_fit = tf.constant(b_fit, dtype=tf.float32)

# ************** bulit networks **************
src_ft, src_ep = feature_extractor(src_real_)
tg_ft, tg_ep = feature_extractor(tg_real_, reuse=True)

src_rec, src_ivim = estimator(src_ft, src_ep, b_values_fit)
tg_rec, tg_ivim = estimator(tg_ft, tg_ep, b_values_fit, reuse=True)

tg_rec_dis, tg_rec_dis_logit, tg_rec_feature = discriminator(tg_rec)
src_rec_dis, src_rec_dis_logit, _ = discriminator(src_rec, reuse=True)  # src_rec
tg_dis, tg_dis_logit, tg_feature = discriminator(tg_real_, reuse=True)
# src_dis_real, _ = discriminator(src_real_)
# src_dis_fake, src_dis_fake_tg = discriminator(src_fake, reuse=True)
# _, tg_dis_real = discriminator(tg_real_, reuse=True)
# tg_dis_fake_src, tg_dis_fake = discriminator(tg_fake, reuse=True)

# ************** estimator loss **************
# loss_e_src = tf.reduce_mean(tf.abs(src_ivim - src_ivim_))
# loss_e_tg = tf.reduce_mean(tf.abs(tg_rec - tg_real_))
# loss_e = loss_e_src + loss_e_tg

loss_e = tf.reduce_mean(tf.abs(src_ivim - src_ivim_))  # + tf.reduce_mean(tf.abs(tg_rec - tg_real_))
# loss_ft = tf.reduce_mean(tf.abs(tg_rec - tg_real_))

# # ************** discriminator loss **************
# # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))

loss_d_src = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.zeros_like(tg_rec_dis_logit) + 0.0, logits=tg_rec_dis_logit))  # src_rec_dis_logit
# loss_d_src_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
# labels=tf.zeros_like(src_dis_real), logits=src_dis_real))
loss_d_tg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(tg_dis_logit) * 1.0, logits=tg_dis_logit))

# loss_d_src = tf.reduce_mean(tf.abs(tg_rec_dis_logit - tf.zeros_like(tg_rec_dis_logit)))  # src_rec_dis_logit
# # loss_d_src_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
# # labels=tf.zeros_like(src_dis_real), logits=src_dis_real))
# loss_d_tg = tf.reduce_mean(tf.abs(tg_dis_logit - tf.ones_like(tg_dis_logit)))

loss_d = loss_d_src + loss_d_tg  # + loss_d_src_real

# loss_d = tf.constant(0.0)
# loss_d_src_real = tf.reduce_mean(-tf.log(src_dis_real))
# loss_d_src_fake = tf.reduce_mean(-tf.log(1 - src_dis_fake))
# loss_d_src = loss_d_src_real + loss_d_src_fake
#
# loss_d_tg_real = tf.reduce_mean(-tf.log(tg_dis_real))
# loss_d_tg_fake = tf.reduce_mean(-tf.log(1 - tg_dis_fake))
# loss_d_tg = loss_d_tg_real + loss_d_tg_fake
#
# loss_d = loss_d_src + loss_d_tg

# # ************** generator loss **************
# loss_g = tf.constant(0.0)
# loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#     labels=tf.ones_like(src_rec_dis_logit) * 1.0, logits=src_rec_dis_logit)) + \
#          tf.reduce_mean(tf.abs(tg_rec_feature - tg_feature))

loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    labels=tf.ones_like(tg_rec_dis_logit) * 1.0, logits=tg_rec_dis_logit))

# loss_g = tf.reduce_mean(tf.abs(tg_rec_dis_logit - tf.ones_like(tg_rec_dis_logit)))

# # ************** feature extractor loss **************
# loss_f_src = tf.reduce_mean(-tf.log(src_dis_fake_tg))
# loss_f_tg = tf.reduce_mean(-tf.log(tg_dis_fake_src))
# loss_f = loss_f_src + loss_f_tg
loss_t = loss_e + 1e-1 * loss_g  # + loss_ft

saver = tf.train.Saver()
# d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
# saver_d = tf.train.Saver(d_vars)

trainable_vars = tf.trainable_variables()
# var_f = [v for v in trainable_vars if 'feature_extractor' in v.name]
# var_g = [v for v in trainable_vars if 'generator' or 'feature_extractor' in v.name]
var_d = [v for v in trainable_vars if 'discriminator' in v.name]
var_e = [v for v in trainable_vars if 'estimator' in v.name or 'feature_extractor' in v.name]
var_ft = [v for v in trainable_vars if 'estimator' in v.name or 'feature_extractor' in v.name]

# global_step_f = tf.Variable(0, trainable=False, name='global_step_f')

# optimizer_d = tf.train.RMSPropOptimizer(learning_rate=learning_rate_d).minimize(
#     loss_d, var_list=var_d)
optimizer_d = tf.train.AdamOptimizer(learning_rate=learning_rate_d, beta1=0.9, beta2=0.999).minimize(
    loss_d, var_list=var_d)
# optimizer_g = tf.train.AdamOptimizer(learning_rate=learning_rate_g, beta1=0.9, beta2=0.999).minimize(
#     loss_g, var_list=var_g)
# optimizer_f = tf.train.AdamOptimizer(learning_rate=learning_rate_f, beta1=0.9, beta2=0.999).minimize(
#     loss_f, var_list=var_f)
optimizer_e = tf.train.AdamOptimizer(learning_rate=learning_rate_e, beta1=0.9, beta2=0.999).minimize(
    loss_t, var_list=var_e)
# optimizer_ft = tf.train.AdamOptimizer(learning_rate=learning_rate_e, beta1=0.9, beta2=0.999).minimize(
#     loss_ft, var_list=var_ft)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

iter_time = []
with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()

    # # ************************************************************************
    # # ********************** train dis ***************************************
    # # ************************************************************************
    # idx = np.arange(num_samples)
    # current_eta = None
    # best = 1e16
    # for epoch in range(num_pre_epochs):
    #     np.random.shuffle(idx)
    #     running_loss = 0.
    #     for n_batch in range(num_pre_batches):
    #         step_time = time.time()
    #         sub_idx = idx[n_batch * pre_batch_size:n_batch * pre_batch_size + pre_batch_size]
    #         batch_fit_src = x_fit_fake[sub_idx]
    #         batch_fit_tg = x_fit_real[sub_idx]  # [np.random.randint(0, x_fit_real.shape[0], pre_batch_size)]
    #
    #         _, ld, ldt, lds, sdr, tgd = sess.run(  #
    #             fetches=[optimizer_d, loss_d, loss_d_tg, loss_d_src, src_rec_dis, tg_dis],  #
    #             feed_dict={src_real_: batch_fit_src, tg_real_: batch_fit_tg}  # , src_gt_: batch_gt ,
    #         )
    #
    #         time_per_iter = time.time() - step_time
    #         n_iter_remain = (num_pre_epochs - epoch - 1) * num_pre_batches + num_pre_batches - n_batch
    #         eta_str, eta_ = eta(time_per_iter, n_iter_remain, current_eta)
    #         current_eta = eta_
    #         running_loss += ld
    #
    #         if (n_batch + 1) % int(num_pre_batches) == 0:
    #             print('Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\tld %.4f\tldt %.4f\tlds %.4f\n'
    #                   '\tsdr %.4f %.4f %.4f\ttgd %.4f %.4f %.4f\n' %
    #                   (epoch + 1, num_pre_epochs, n_batch + 1, num_pre_batches, eta_str, running_loss, ldt, lds,
    #                    np.max(sdr), np.mean(sdr), np.min(sdr), np.max(tgd), np.mean(tgd), np.min(tgd)))
    #
    #     if running_loss < best:
    #         saver.save(sess, "Model/%s/model.ckpt" % save_model_path)
    #         # saver_d.save(sess, "Model/dis/model.ckpt")
    #         best = running_loss
    #         # print('saved')
    #     # else:
    #     #     num_bad_epochs = num_bad_epochs + 1
    #     #     if num_bad_epochs == patience:
    #     #         print("Done, best loss: {}".format(best))
    #     #         break
    #
    # # saver_d.restore(sess, "Model/dis/model.ckpt")
    # saver.restore(sess, "./Model/%s/model.ckpt" % save_model_path)

    # ************************************************************************
    # ********************** train all ***************************************
    # ************************************************************************
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
            src_idx = np.random.randint(0, x_fit_fake.shape[0], batch_size)
            batch_fit_src = x_fit_fake[src_idx]
            batch_ivim = ivim_train[src_idx]
            batch_fit_tg = x_fit_real[sub_idx]
            # batch_gt = x_fit_gt[sub_idx]

            _, ld, ldt, lds, srd, tgd, trd = sess.run(  #
                fetches=[optimizer_d, loss_d, loss_d_tg, loss_d_src, src_rec_dis, tg_dis, tg_rec_dis],  #
                feed_dict={src_real_: batch_fit_src, tg_real_: batch_fit_tg}  # , src_gt_: batch_gt ,
            )
            # ld, ldt, lds, srd, tgd, trd = 0, 0, 0, 0, 0, 0

            _, le, lg, tg_ivim_pre = sess.run(  #
                fetches=[optimizer_e, loss_e, loss_g, tg_ivim],  #
                feed_dict={src_real_: batch_fit_src, src_ivim_: batch_ivim, tg_real_: batch_fit_tg}
                # , src_gt_: batch_gt ,
            )

            time_per_iter = time.time() - step_time
            n_iter_remain = (num_epochs - epoch - 1) * num_batches + num_batches - n_batch
            eta_str, eta_ = eta(time_per_iter, n_iter_remain, current_eta)
            current_eta = eta_
            running_loss += le

            if (n_batch + 1) % int(num_batches) == 0:
                # lft, tg_ivim_pre = sess.run(
                #     fetches=[loss_ft, tg_ivim],
                #     feed_dict={tg_real_: X_test_real}  # , src_gt_: batch_gt
                # )
                # running_loss = lft
                print('Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\tld %.4f\tldt %.4f\tlds %.4f\n'
                      '\tsdr %.4f %.4f %.4f\ttgd %.4f %.4f %.4f\ttrd %.4f %.4f %.4f' %
                      (epoch + 1, num_epochs, n_batch + 1, num_batches, eta_str, ld, ldt, lds,
                       np.max(srd), np.mean(srd), np.min(srd), np.max(tgd), np.mean(tgd), np.min(tgd),
                       np.max(trd), np.mean(trd), np.min(trd)))
                print('le %.4f\tlg %.4f\tmse = %.8f\tdp = %.5f\tdt = %.4f\tfp = %.4f\tbad epochs = %d\n' %
                      (le, lg, running_loss, tg_ivim_pre[..., 0].mean(), tg_ivim_pre[..., 1].mean(),
                       tg_ivim_pre[..., 2].mean(), num_bad_epochs))
                # print('Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\tle %.4f\tlg %.4f\n'
                #       '\tmse = %.8f\tdp = %.5f\tdt = %.4f\tfp = %.4f\tbad epochs = %d\n' %
                #       (epoch + 1, num_epochs, n_batch + 1, num_batches, eta_str,
                #        le, lg, running_loss, tg_ivim_pre[..., 0].mean(), tg_ivim_pre[..., 1].mean(),
                #        tg_ivim_pre[..., 2].mean(), num_bad_epochs))

        # early stopping
        if running_loss < best:
            saver.save(sess, "Model/%s/model.ckpt" % save_model_path)
            # saver_d.save(sess, "Model/dis/model.ckpt")
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best))
                break

    saver.restore(sess, "./Model/%s/model.ckpt" % save_model_path)

    # # ************************************************************************
    # # ********************** fine-tuning *************************************
    # # ************************************************************************
    # num_samples = x_fit_real.shape[0]
    # num_batches = int(num_samples / batch_size)
    # idx = np.arange(num_samples)
    # current_eta = None
    # best = 1e16
    # num_bad_epochs = 0
    # for epoch in range(num_epochs):
    #     np.random.shuffle(idx)
    #     running_loss = 0.
    #     # x_fit, ivim_train = simulate_data(img_w, img_h, batch_size * num_batches, b_fit)
    #     for n_batch in range(num_batches):
    #         step_time = time.time()
    #         sub_idx = idx[n_batch * batch_size:n_batch * batch_size + batch_size]
    #         # batch_fit_src = x_fit_fake[sub_idx]
    #         batch_fit_tg = x_fit_real[sub_idx]
    #         # batch_ivim = ivim_train[sub_idx]
    #         # batch_gt = x_fit_gt[sub_idx]
    #
    #         _, lft = sess.run(
    #             fetches=[optimizer_ft, loss_ft],
    #             feed_dict={tg_real_: batch_fit_tg}  # , src_gt_: batch_gt
    #         )
    #
    #         time_per_iter = time.time() - step_time
    #         n_iter_remain = (num_epochs - epoch - 1) * num_batches + num_batches - n_batch
    #         eta_str, eta_ = eta(time_per_iter, n_iter_remain, current_eta)
    #         current_eta = eta_
    #         # running_loss += lft
    #
    #         if (n_batch + 1) % int(num_batches) == 0:
    #             lft, tg_ivim_pre = sess.run(
    #                 fetches=[loss_ft, tg_ivim],
    #                 feed_dict={tg_real_: X_test_real}  # , src_gt_: batch_gt
    #             )
    #             running_loss = lft
    #             print('Finetune Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\tlft %.4f\n'
    #                   '\tmse = %.8f\tdp = %.5f\tdt = %.4f\tfp = %.4f\tbad epochs = %d\n' %
    #                   (epoch + 1, num_epochs, n_batch + 1, num_batches, eta_str, lft,
    #                    running_loss, tg_ivim_pre[..., 0].mean(), tg_ivim_pre[..., 1].mean(),
    #                    tg_ivim_pre[..., 2].mean(), num_bad_epochs))
    #
    #     # early stopping
    #     if running_loss < best:
    #         saver.save(sess, "Model/%s/model.ckpt" % save_model_path)
    #         best = running_loss
    #         num_bad_epochs = 0
    #     else:
    #         num_bad_epochs = num_bad_epochs + 1
    #         if num_bad_epochs == patience:
    #             print("Done, best loss: {}".format(best))
    #             break
    #
    # saver.restore(sess, "./Model/%s/model.ckpt" % save_model_path)

    # x_fit_pre_all, ivim_pre_all = [], []
    # for idxt, x_test in enumerate(X_test):
    #     x_test_fit = x_test[..., 1:]
    #
    #     x_test_fit = patch(x_test_fit, img_w, img_h)
    #     # _, le, tg_ivim_pre = sess.run(
    #     #     fetches=[optimizer_e, loss_e, tg_ivim],
    #     #     feed_dict={src_real_: batch_fit_src}
    #     # )
    #     ivim_pre_, x_fit_pre = sess.run(
    #         fetches=[ivim_pre, x_pre_fit],
    #         feed_dict={x_fit_: x_test_fit}
    #     )
    #
    #     # make sure Dp is the larger value between Dp and Dt
    #     if ivim_pre_[..., 0].mean() < ivim_pre_[..., 1].mean():
    #         print('sawp')
    #         temp = copy.deepcopy(ivim_pre_[..., 0])
    #         ivim_pre_[..., 0] = copy.deepcopy(ivim_pre_[..., 1])
    #         ivim_pre_[..., 1] = temp
    #         ivim_pre_[..., 2] = 1 - ivim_pre_[..., 2]
    #
    #     x_fit_pre_all.append(unpatch(x_fit_pre, 160, 192))  #
    #     ivim_pre_all.append(unpatch(ivim_pre_, 160, 192))  #
    #
    # np.savez(join(test_save_path, 'test.npz'), ivim=ivim_pre_all, x_fit_pre=x_fit_pre_all)

    tg_rec_dis_all, src_rec_dis_all, tg_dis_all = [], [], []
    for ii in range(0, x_fit_real.shape[0], 18):
        tg_rec_dis_, src_rec_dis_, tg_dis_ = sess.run(
            fetches=[tg_rec_dis, src_rec_dis, tg_dis],
            feed_dict={tg_real_: x_fit_real[ii: ii+18], src_real_: x_fit_fake[ii: ii+18]}
        )
        tg_rec_dis_all.append(tg_rec_dis_)
        src_rec_dis_all.append(src_rec_dis_)
        tg_dis_all.append(tg_dis_)
    tg_rec_dis_all = np.concatenate(tg_rec_dis_all, 0)
    src_rec_dis_all = np.concatenate(src_rec_dis_all, 0)
    tg_dis_all = np.concatenate(tg_dis_all, 0)
    d_list = np.array([tg_rec_dis_all, src_rec_dis_all, tg_dis_all])
    np.savez('./dis1.npz', d=d_list)

    for idxt, x_test in enumerate(X_test_real):
        x_test_fit = x_test  # [..., 1:]

        x_test_fit = patch(x_test_fit, img_w, img_h)
        ivim_pre_, x_fit_pre = sess.run(
            fetches=[tg_ivim, tg_rec],
            feed_dict={tg_real_: x_test_fit}
        )

        # make sure Dp is the larger value between Dp and Dt
        if ivim_pre_[..., 0].mean() < ivim_pre_[..., 1].mean():
            print('sawp')
            temp = copy.deepcopy(ivim_pre_[..., 0])
            ivim_pre_[..., 0] = copy.deepcopy(ivim_pre_[..., 1])
            ivim_pre_[..., 1] = temp
            ivim_pre_[..., 2] = 1 - ivim_pre_[..., 2]

        x_fit_pre = unpatch(x_fit_pre, 160, 192)
        ivim_pre_ = unpatch(ivim_pre_, 160, 192)

        np.savez(join(save_path_real, b_files[idxt].split('/')[-1]), ivim=ivim_pre_, x_fit_pre=x_fit_pre)

    sess.close()
