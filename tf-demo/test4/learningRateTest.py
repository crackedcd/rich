"""
学习率 learning rate

更新后的参数 = 更新前的参数 - 学习率 * 损失函数的导数

优化学习率, 是为了找到损失函数梯度最低的点

"""

import tensorflow as tf


def learning_rate_test1():
    """
    设损失函数 loss = (w + 1) ^ 2, 此时要使得loss梯度最低, 则肉眼可见, w = -1时, loss最小.
    通过学习, 也可以得出这个值.
    :return:
    """
    # 设w初始值为5(随便设一个)
    w = tf.Variable(tf.constant(5, dtype=tf.float32))
    # 定义损失函数
    loss = tf.square(w + 1)
    # 定义反向传播方法
    lr = 0.2
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            sess.run(train_step)
            w_val = sess.run(w)
            loss_val = sess.run(loss)
            print("第%d步: w是 %f , loss是 %f" % (i, w_val, loss_val))

    return None


def learning_rate_test2():
    """
    在learning_rate_test1()中, 可以反复修改lr的值调试, 会发现:
    lr较大, 会震荡
    lr较小, 收敛速度很慢

    指数衰减学习率
    base基数
    decay衰减率
    lr = 学习率基数 * 学习率衰减率 ^ (训练轮数 / 多少轮更新一次学习率)
    一般多少次更新一次学习率? 使用总样本数 / 一批的大小batch_size

    :return:
    """
    # 定义最初学习率
    lr_base = 0.4
    # 定义学习率衰减率
    lr_decay = 0.95
    # 定义多少轮更新一次学习率
    lr_step = 1
    # 训练轮数计数器, 标记为不被训练
    global_step = tf.Variable(0, trainable=False)
    # staircase为真, "多少轮更新一次学习率"取整, 学习率梯度衰减, 为假, 学习率平滑下降
    lr = tf.train.exponential_decay(
        learning_rate=lr_base, global_step=global_step, decay_steps=lr_step, decay_rate=lr_decay, staircase=True)

    # 设w初始值为10(随便设一个)
    w = tf.Variable(tf.constant(10, dtype=tf.float32))
    # 定义损失函数
    loss = tf.square(w + 1)
    # 定义反向传播方法
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step=global_step)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(50):
            sess.run(train_step)
            w_val = sess.run(w)
            lr_val = sess.run(lr)
            loss_val = sess.run(loss)
            print("第%d步: 学习率是%f, w是 %f , loss是 %f" % (i+1, lr_val, w_val, loss_val))

    return None


if __name__ == '__main__':
    # learning_rate_test1()
    learning_rate_test2()
