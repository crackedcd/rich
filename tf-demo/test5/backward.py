"""
反向传播

def backward():
    x = tf.placeholder()
    y_ = tf.placeholder()
    # 使用前向传播的网络结构求预测值y
    y = forward.forward(x, r)

    # 训练轮数计数器 global_step
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数, 可以是:
    # > test4/lossTest
    # 均方误差 loss_base = tf.reduce_mean(tf.square(y - y_))
    # 交叉熵 loss_base = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 加入正则化
    # > test4/regularizationTest
    loss_real = loss_base + tf.add_n(tf.get_collection("losses"))

    # 学习率
    # > test4/learningRateTest
    lr = tf.train.exponential_decay(lr_base, global_step, total_count / batch_size, lr_decay, staircase=True)
    # 训练过程
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)

    # 滑动平均
    # > test4/emaTest
    moving_average_decay =
    ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    ema_op = ema.apply(tf.trainable_variables())

    # graph到session
    with tf.Session() as sess:
        # 初始化值
        sess.run(tf.global_variables_initializer())

        for i in range(steps):
            # 喂入参数
            feeds =
            sess.run(train_step, feed_dict=feeds)

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import forward
import generate


def backward():
    steps = 50000
    batch_size = 30
    lr_base = 0.001
    lr_decay = 0.999
    regularizer = 0.01
    data_count = generate.data_count()
    global_step = tf.Variable(0, trainable=False)

    # nn
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    mat_x, mat_y_, y_color = generate.generate()
    y = forward.forward(x, regularizer)

    lr = tf.train.exponential_decay(
        learning_rate=lr_base,
        global_step=global_step,
        decay_steps=data_count/batch_size,
        decay_rate=lr_decay,
        staircase=True)

    loss_mse = tf.reduce_mean(tf.square(y - y_))
    loss_regular = loss_mse + tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.AdamOptimizer(lr).minimize(loss_regular)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(steps):
            start = (i * batch_size) % data_count
            end = start + batch_size
            feeds = {
                x: mat_x[start:end],
                y_: mat_y_[start:end]
            }
            sess.run(train_step, feed_dict=feeds)
            if i % 5000 == 0:
                loss_mse_val = sess.run(loss_mse, feed_dict={x: mat_x, y_:mat_y_})
                print("第%d步的损失为%f" % (i, loss_mse_val))

        xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)

    plt.scatter(mat_x[:, 0], mat_x[:, 1], c=np.squeeze(y_color))
    plt.contour(xx, yy, probs, levels=[0.5])
    plt.show()

    return None


if __name__ == '__main__':
    backward()
