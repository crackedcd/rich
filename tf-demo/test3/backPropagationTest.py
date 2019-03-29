"""
损失函数 loss
预测值(predict)(y)与已知答案(target)(y_)的差距

均方误差 MSE mean-square error
MSE(y, y_) = sigma ((y - y_)^2 / n)
loss = tf.reduce_mean(tf.square(y, y_))

反向传播 BP back propagation
为训练模型参数, 在所有参数上用梯度下降, 使NN模型在训练数据上的损失最小.
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

学习率 LR learning rate
                学习率 大                学习率 小
学习速度            快                      慢
使用时间点       刚开始训练时              一定轮数过后
副作用     1.易损失值爆炸; 2.易振荡.	    1.易过拟合; 2.收敛速度慢.

刚开始训练时: 学习率以 0.01 ~ 0.001 为宜.
一定轮数过后: 逐渐减缓.
接近训练结束: 学习速率的衰减应该在100倍以上.

"""

import tensorflow as tf
import numpy as np


def back_propagation_test():
    # 训练次数
    steps = 3000
    # 每次喂入数据数量
    batch_size = 8
    # 随机种子
    seed = 8731
    # 基于seed产生随机数
    rng = np.random.RandomState(seed)
    # 生成32组重量和体积作为输入数据集, 32行2列的矩阵
    mat_x = rng.rand(32, 2)
    mat_y = []
    # print(mat_x)
    # 假设"体积 + 重量 < 1"的零件合格, 构造mat_y. 从X中取出一行, 判断如果两者的和小于1, 给Y赋值1, 否则赋值0.
    # 神经网络判断的依据是"数据"和"概率", 它并不知道人为标注y是0或1的方法.
    # pythonic code: mat_y = [[int(x0 + x1 < 1)] for (x0, x1) in mat_x]
    for x0, x1 in mat_x:
        if x0 + x1 < 1:
            mat_y.append([1])
        else:
            mat_y.append([0])
    # print(mat_y)

    # 前向传播
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    # 反向传播
    loss = tf.reduce_mean(tf.square(y - y_))
    lr = 0.001
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    # 训练
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 输出未训练的参数取值
        print("训练前的结果")
        print(sess.run(w1))
        print(sess.run(w2))

        for i in range(steps):
            # 数据集只有32个(行), 所以对32取余, 让start在数据集范围内, i * batch_size让每次训练跨度batch_size个数据
            start = (i * batch_size) % 32
            end = start + batch_size
            feeds = {
                x: mat_x[start:end],
                y_: mat_y[start:end]
            }
            # 每次循环中, 代入输入特征(data)和标准答案(target)
            sess.run(train_step, feed_dict=feeds)
            if i % 500 == 0:
                total_loss = sess.run(loss, feed_dict={x: mat_x, y_: mat_y})
                print("在%d次训练后, 损失为%g" % (i, total_loss))

        print("训练后的结果")
        print(sess.run(w1))
        print(sess.run(w2))

    return None


if __name__ == '__main__':
    back_propagation_test()
