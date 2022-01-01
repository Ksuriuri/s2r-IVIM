import tensorflow as tf
from os.path import join
from glob import glob
import numpy as np
from tensorflow.python.training import moving_averages


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


def get_fit_and_no_fit(inputs, no_fit_num):
    if len(inputs.shape) == 1:
        return np.concatenate((inputs[1: no_fit_num], inputs[no_fit_num + 1:]), -1), inputs[no_fit_num: no_fit_num + 1]
    elif len(inputs.shape) > 1:
        return np.concatenate((inputs[..., 1: no_fit_num],
                               inputs[..., no_fit_num + 1:]), -1), inputs[..., no_fit_num: no_fit_num + 1]


def simulate_data(img_w, img_h, batch_size, b_values):
    def ivim(b, dp, dt, fp):
        return fp * np.exp(np.matmul(-dp, b)) + (1 - fp) * np.exp(np.matmul(-dt, b))

    Dp_train = np.random.uniform(0.01, 0.4, (batch_size, img_w * img_h, 1))  # 0.01, 0.1  0.5
    Dt_train = np.random.uniform(0.0, 0.01, (batch_size, img_w * img_h, 1))  # 0.0005, 0.002  0.008
    Fp_train = np.random.uniform(0.0, 0.7, (batch_size, img_w * img_h, 1))  # 0.1, 0.4  0.7
    b_values_ = np.expand_dims(b_values, 0)
    X_train = ivim(b_values_, Dp_train, Dt_train, Fp_train)  # range 0 - 1

    Dp_train = np.reshape(Dp_train, (batch_size, img_w, img_h, 1))
    Dt_train = np.reshape(Dt_train, (batch_size, img_w, img_h, 1))
    Fp_train = np.reshape(Fp_train, (batch_size, img_w, img_h, 1))
    X_train = np.reshape(X_train, (batch_size, img_w, img_h, len(b_values)))
    ivim_train = np.concatenate([Dp_train, Dt_train, Fp_train], axis=-1)  # clear map

    # # rg = np.random.RandomState(hp.rs_1)
    # noise_sd = np.random.uniform(0, 0.11, (batch_size, img_w, img_h, 1))
    # # rg = np.random.RandomState(456)
    # # add noise
    # for i in range(batch_size):
    #     for j in range(img_w):
    #         for k in range(img_h):
    #             nosiy = np.random.normal(scale=noise_sd[i][j][k], size=len(b_values))
    #             X_train_real = X_train[i][j][k] + nosiy
    #             X_train_imag = nosiy
    #             X_train[i][j][k] = np.sqrt(X_train_real ** 2 + X_train_imag ** 2)  # / s0
    # noise_scale = np.random.uniform(0.1, 0.3)
    nosiy = np.random.normal(scale=0.3, size=X_train.shape)
    # X_train_real = X_train + nosiy
    # X_train_imag = nosiy
    # X_train = np.sqrt(X_train_real ** 2 + X_train_imag ** 2)  # / s0
    X_train = X_train + nosiy
    # print(X_train.max(), ' ', X_train.mean(), ' ', X_train.min())
    return X_train, ivim_train


def convolutional(input_data, filters_shape, name, trainable=True, downsample=False, activate=True, bn=False):
    with tf.variable_scope(name):
        if downsample:
            pad_h, pad_w = (filters_shape[0] - 2) // 2 + 1, (filters_shape[1] - 2) // 2 + 1
            paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
            input_data = tf.pad(input_data, paddings, 'CONSTANT')
            strides = (1, 2, 2, 1)
            padding = 'VALID'
        else:
            strides = (1, 1, 1, 1)
            padding = "SAME"
        weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                 shape=filters_shape, initializer=tf.random_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input=input_data, filter=weight, strides=strides, padding=padding)
        if bn:
            conv = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(), training=trainable)
        else:
            bias = tf.get_variable(name='bias', shape=filters_shape[-1], trainable=True,
                                   dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, bias)
        if activate is True:
            # conv = tf.nn.leaky_relu(conv, alpha=0.1)
            conv = tf.nn.relu(conv)
        return conv


def upsample(input_data, shortcut_size, name, method="deconv"):
    assert method in ["resize", "deconv"]
    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))
    if method == "deconv":
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        num_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, num_filter//2, kernel_size=4, padding='same',
                                            strides=(2, 2), kernel_initializer=tf.random_normal_initializer())
        # output = tf.image.resize_nearest_neighbor(output, shortcut_size)
    return output


def resblock(net, in_channels, out_channels, stride=1, res_scale=1, name='resblock'):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name) as vs:
        res = net
        net = conv2d(net, in_channels, filter_size=(3, 3), strides=(stride, stride), w_init=w_init, b_init=b_init,
                     padding='SAME', name='conv1', act=tf.nn.relu)
        net = conv2d(net, out_channels, filter_size=(3, 3), strides=(1, 1), w_init=w_init, b_init=b_init,
                     padding='SAME', name='conv2')

        if stride > 1:
            res = conv2d(res, out_channels, filter_size=(1, 1), strides=(stride, stride), w_init=w_init, b_init=b_init,
                         padding='SAME', name='shortcut')

        out = net * res_scale + res
        return out


def conv2d(net, n_filter=32, filter_size=(3, 3), strides=(1, 1), act=tf.identity, padding='SAME',
           w_init=tf.truncated_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(value=0.0),
           name='conv2d'):
    kernel_shape = [filter_size[0], filter_size[1], int(net.get_shape()[-1]), n_filter]
    with tf.variable_scope(name) as vs:
        w = tf.get_variable(name='W_conv2d', shape=kernel_shape, initializer=w_init)
        if b_init:
            b = tf.get_variable(name='b_conv2d', shape=(kernel_shape[-1]), initializer=b_init)
            output = act(tf.nn.conv2d(net, w, strides=[1, strides[0], strides[1], 1], padding=padding) + b)
        else:
            output = act(tf.nn.conv2d(net, w, strides=[1, strides[0], strides[1], 1], padding=padding))
        return output


def deconv2d(net, n_filter=32, filter_size=(3, 3), strides=(1, 1), out_size=(28, 28), act=tf.identity, padding='SAME',
           w_init=tf.truncated_normal_initializer(stddev=0.02), b_init=tf.constant_initializer(value=0.0),
           name='conv2d'):
    kernel_shape = [filter_size[0], filter_size[1], n_filter, int(net.get_shape()[-1])]

    fixed_batch_size = net.get_shape().with_rank_at_least(1)[0]
    if fixed_batch_size.value:
        batch_size = fixed_batch_size.value
    else:
        from tensorflow.python.ops import array_ops
        batch_size = array_ops.shape(net)[0]
    out_shape = [batch_size, out_size[0], out_size[1], n_filter]
    with tf.variable_scope(name) as vs:
        w = tf.get_variable(name='W_conv2d', shape=kernel_shape, initializer=w_init)
        if b_init:
            b = tf.get_variable(name='b_conv2d', shape=(kernel_shape[-2]), initializer=b_init)
            output = act(tf.nn.conv2d_transpose(net, w, strides=[1, strides[0], strides[1], 1], output_shape=out_shape,
                                                padding=padding) + b)
        else:
            output = act(tf.nn.conv2d_transpose(net, w, strides=[1, strides[0], strides[1], 1], output_shape=out_shape,
                                                padding=padding))
        return output


def dense(net, n_units=256, act=tf.identity, w_init=tf.truncated_normal_initializer(stddev=0.1),
          b_init=tf.constant_initializer(value=0.0), name='dense1'):
    n_in = int(net.get_shape()[-1])
    with tf.variable_scope(name) as vs:
        w = tf.get_variable(name='W', shape=(n_in, n_units), initializer=w_init)
        if b_init is not None:
            b = tf.get_variable(name='b', shape=(n_units), initializer=b_init)
            outputs = act(tf.matmul(net, w) + b)
        else:
            outputs = act(tf.matmul(net, w))

        return outputs


def subpixel(net, scale=2, n_out_channel=None, act=tf.identity, name='subpixel_conv2d'):
    _err_log = "SubpixelConv2d: The number of input channels == (scale x scale) x The number of output channels"
    def _PS(X, r, n_out_channel):
        if n_out_channel >= 1:
            assert int(X.get_shape()[-1]) == (r ** 2) * n_out_channel, _err_log
            #bsize, a, b, c = X.get_shape().as_list()
            #bsize = tf.shape(X)[0] # Handling Dimension(None) type for undefined batch dim
            #Xs=tf.split(X,r,3) #b*h*w*r*r
            #Xr=tf.concat(Xs,2) #b*h*(r*w)*r
            #X=tf.reshape(Xr,(bsize,r*a,r*b,n_out_channel)) # b*(r*h)*(r*w)*c
            X = tf.depth_to_space(X, r)
        else:
            print(_err_log)
        return X

    if n_out_channel is None:
        assert int(net.get_shape()[-1]) / (scale ** 2) % 1 == 0, _err_log
        n_out_channel = int(int(net.get_shape()[-1]) / (scale ** 2))

    # with tf.name_scope(name):
    with tf.variable_scope(name) as vs:
        net = act(_PS(net, r=scale, n_out_channel=n_out_channel))
        return net


def batch_norm(x, name, _ops, is_train=True):
    """Batch normalization."""
    with tf.variable_scope(name):
        params_shape = [x.get_shape()[-1]]

        beta = tf.get_variable('beta', params_shape, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', params_shape, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))

        if is_train is True:
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

            moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                          initializer=tf.constant_initializer(0.0, tf.float32),
                                          trainable=False)
            moving_variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                              initializer=tf.constant_initializer(1.0, tf.float32),
                                              trainable=False)

            _ops.append(moving_averages.assign_moving_average(moving_mean, mean, 0.9))
            _ops.append(moving_averages.assign_moving_average(moving_variance, variance, 0.9))
        else:
            mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
            variance = tf.get_variable('moving_variance', params_shape, tf.float32, trainable=False)

        # epsilon used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
        y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5)
        y.set_shape(x.get_shape())

        return y


def flatten(net, name='flatten'):
    dim = 1
    for d in net.get_shape()[1:].as_list():
        dim *= d
    return tf.reshape(net, shape=[-1, dim], name=name)


def eta(time_per_iter, n_iter_remain, current_eta=None, alpha=.8):
    eta_ = time_per_iter * n_iter_remain
    if current_eta is not None:
        eta_ = (current_eta - time_per_iter) * alpha + eta_ * (1 - alpha)
    new_eta = eta_

    days = eta_ // (3600 * 24)
    eta_ -= days * (3600 * 24)

    hours = eta_ // 3600
    eta_ -= hours * 3600

    minutes = eta_ // 60
    eta_ -= minutes * 60

    seconds = eta_

    if days > 0:
        if days > 1:
            time_str = '%2d days %2d hr' % (days, hours)
        else:
            time_str = '%2d day %2d hr' % (days, hours)
    elif hours > 0 or minutes > 0:
        time_str = '%02d:%02d' % (hours, minutes)
    else:
        time_str = '%02d sec' % seconds

    return time_str, new_eta


# def get_file_name(dir, type, one_sample_num):
#     # one_sample_num = self.one_sample_num
#     high_test_num = 22 * one_sample_num  # 21
#     low_test_num = 18 * one_sample_num  # 19
#     files_input = sorted(glob(join(dir, '*' + type)))
#     files_input.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), int(x.split('_')[-1].split('.')[0])))
#     high_files = [hf for hf in files_input if int(hf.split('/')[-1].split('_')[0]) == 1]
#     print('high_files ', len(high_files))
#     low_files = [lf for lf in files_input if int(lf.split('/')[-1].split('_')[0]) == 0]
#     print('low_files ', len(low_files))
#     high_num, low_num = len(high_files) - high_test_num, len(low_files) - low_test_num
#     train_files = np.concatenate((high_files[:high_num], low_files[:low_num]))
#     train_labels = np.array([np.eye(2)[lb] for lb in np.concatenate(
#         (np.ones(shape=high_num), np.zeros(shape=low_num))).astype(np.uint8)]).astype(np.float32)
#
#     test_files = np.concatenate((high_files[high_num:], low_files[low_num:]))
#     test_labels = np.array([np.eye(2)[lb] for lb in np.concatenate(
#         (np.ones(shape=high_test_num), np.zeros(shape=low_test_num))).astype(np.uint8)]).astype(np.float32)
#
#     return train_files, test_files, train_labels, test_labels


def get_file_name(dir_, type_, one_sample_num):
    files_input = sorted(glob(join(dir_, '*' + type_)))
    files_input.sort(key=lambda x: (int(x.split('_')[-1].split('.')[0])))
    files_input = np.array(files_input)

    label_ = []
    for fi in files_input:
        if int(fi.split('/')[-1].split('_')[0]) == 1:
            label_.append([1, 0])
        elif int(fi.split('/')[-1].split('_')[0]) == 0:
            label_.append([0, 1])
    label_ = np.array(label_).astype(np.float32)

    train_num = len(files_input) - 40 * one_sample_num
    train_files = files_input[:train_num]
    train_labels = label_[:train_num]
    test_files = files_input[train_num:]
    test_labels = label_[train_num:]

    return train_files, test_files, train_labels, test_labels


def sen_spe(op, eq):
    positive_position = 0
    negative_position = 1
    staticity_T = [0, 0]
    staticity_F = [0, 0]

    for i in range(len(eq)):
        if eq[i]:
            staticity_T[op[i]] += 1
        else:
            staticity_F[op[i]] += 1
    sensitivity = staticity_T[positive_position] / (staticity_T[positive_position] + staticity_F[negative_position])
    specificity = staticity_T[negative_position] / (staticity_T[negative_position] + staticity_F[positive_position])
    return sensitivity, specificity
