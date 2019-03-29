"""

损失函数 loss
预测值(predict)(y)与已知答案(target)(y_)的差距
            | -> 均方误差 MSE Mean Squared Error        tf.reduce_mean(tf.square(y, y_))
loss最小 -> | -> 交叉熵 Cross Entropy                   tf.reduce_sum(tf.where(tf.greater(y, y_), if_true, if_false))
           | -> 自定义                                 -tf.reduce_mean(y_ * tf.log(y))

"""

import tensorflow as tf
import numpy as np


def loss_test1():
    batch_size = 8
    # 固定seed便于调试, 使得每次结果一致, 实际使用中不需要固定seed
    seed = 8731
    rdm = np.random.RandomState(seed)
    mat_x = rdm.rand(32, 2)
    mat_y_ = []
    # 指定 x 和 y_ 关系造数据
    for (x1, x2) in mat_x:
        # RandomState().rand(row, col)生成[0, 1)区间的随机值矩阵, 利用这个来生成 -0.05 ~ 0.05 之间的n
        n = rdm.rand(1, 1)[0][0] / 10.0 - 0.05
        # 也就是说, 生成的结果(目的)是 x1 + x2, 正负误差0.05
        # 如果按 w1 * x1 + w2 * x2 = y 来看, 应该拟合出的模型要满足 w1 -> 1, w2 -> 1
        mat_y_.append([x1 + x2 + n])

    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
    y = tf.matmul(x, w)

    print(mat_x)
    print(mat_y_)

    # mse
    loss = tf.reduce_mean(tf.square(y_ - y))
    lr = 0.001
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        # 测试数据, 根据 y_ 的算法, 应该拟合出 y -> 2
        feed_test = {x: [[1, 1]]}
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        steps = 30000
        print("训练前的w是")
        print(sess.run(w))
        for i in range(steps):
            start = (i * batch_size) % 32
            end = start + batch_size
            feeds = {
                x: mat_x[start:end],
                y_: mat_y_[start:end]
            }
            sess.run(train_step, feed_dict=feeds)
            if i % 1000 == 0:
                print("经过%d步后" % i)
                print("w是")
                print(sess.run(w))
                print("测试数据的预测结果是")
                print(sess.run(y, feed_dict=feed_test))
        print("最终的w是")
        print(sess.run(w))
        print("最终对测试数据的预测结果是")
        print(sess.run(y, feed_dict=feed_test))

    return None


def loss_test2():
    """
    使用MSE进行预测的前提, 是预测多了或预测少了对于结果而言都一样.
    但实际场景中, 预测多了和预测少了可能是不一样的.
    例如预测销量, 成本是1元, 利润是9元. 如果预测少了, 生产不足, 会损失的利润是9元, 预测多了, 生产多了, 损失的成本仅有1元.
    在这种情况下, 我们希望能尽可能预测多, 需要自定义损失函数.

    使用自定义损失函数计算每一个结果y和标准答案y_的损失的累计和:
    my_loss(y, y_) = sigma(ori_loss(y, y_))
    如果y < y_, 也就是预测的结果小于标准答案的结果, 预测少了, 则损失利润(profit)
    如果y > y_, 也就是预测的结果大于标准答案的结果, 预测多了, 则损失成本(cost)

    tf.reduce_sum(y > y_, if_true, if_false)
    ↓
    loss = tf.reduce_sum(tf.where(tf.greater(y, y_), cost(y - y_), profit(y_ - y))

    在这个损失函数的定义下, 最终的w1和w2都会在尽可能准确的前提下往多预测, 也就是都略大于1.

    :return:
    """
    batch_size = 8
    # 固定seed便于调试, 使得每次结果一致, 实际使用中不需要固定seed
    seed = 8731
    rdm = np.random.RandomState(seed)
    mat_x = rdm.rand(32, 2)
    mat_y_ = []
    # 指定 x 和 y_ 关系造数据
    for (x1, x2) in mat_x:
        # RandomState().rand(row, col)生成[0, 1)区间的随机值矩阵, 利用这个来生成 -0.05 ~ 0.05 之间的n
        n = rdm.rand(1, 1)[0][0] / 10.0 - 0.05
        # 也就是说, 生成的结果(目的)是 x1 + x2, 正负误差0.05
        # 如果按 w1 * x1 + w2 * x2 = y 来看, 应该拟合出的模型要满足 w1 -> 1, w2 -> 1
        mat_y_.append([x1 + x2 + n])

    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
    y = tf.matmul(x, w)

    print(mat_x)
    print(mat_y_)

    # 在loss_test1()的基础上, 仅修改损失函数loss, 假设成本为1, 利润为9.
    cost = 9
    profit = 1
    # 如果 y > y_, 即预测结果大于真实值, 会扩大生产, 那么会损失更多的成本 * cost
    # 否则, 即预测结果大于真实值, 会减少生产, 那么会损失更多的利润 * profit
    loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * cost, (y_ - y) * profit))
    lr = 0.001
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    with tf.Session() as sess:
        # 测试数据, 根据 y_ 的算法, 应该拟合出 y -> 2
        feed_test = {x: [[1, 1]]}
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        steps = 30000
        print("训练前的w是")
        print(sess.run(w))
        for i in range(steps):
            start = (i * batch_size) % 32
            end = start + batch_size
            feeds = {
                x: mat_x[start:end],
                y_: mat_y_[start:end]
            }
            sess.run(train_step, feed_dict=feeds)
            if i % 1000 == 0:
                print("经过%d步后" % i)
                print("w是")
                print(sess.run(w))
                print("测试数据的预测结果是")
                print(sess.run(y, feed_dict=feed_test))
        print("最终的w是")
        print(sess.run(w))
        print("最终对测试数据的预测结果是")
        print(sess.run(y, feed_dict=feed_test))
    return None


def loss_test3():
    """
    交叉熵 Cross Entropy
    表征两个概率分布之间的关系
    f(y, y_) = - sigma(y_ * log(y))

    通过交叉熵, 可以计算二分类问题, 例如:
    将"二分问题"视作第一种情况出现概率100%, 第二种0%, 也就是(100%, 0%),
    设已知标准答案是(1, 0), 也就是判断(0.6, 0.4)和(0.8, 0.2)哪个更接近标准答案.
    f(0.6, 0.4) = - (1 * log(0.6) + 0 * log(0.4)) = - (-0.222 + 0) = 0.222
    f(0.8, 0.2) = - (1 * log(0.8) + 0 * log(0.2)) = - (-0.097 + 0) = 0.097
    所以(0.8, 0.2)比(0.6, 0.4)更接近(1, 0)

    假设存在max(a, b)方法, 取a和b种的更大值, 这个max()方法永远只会取最大一个值.
    如果需要最大值"经常取到", 最小值"偶尔取到", 这个方法称作softmax().
    softmax()可以使得在n分类问题种, 将n个值全部映射到(0, 1)之间, 并且其累加和正好等于1, 满足概率的性质.
    即, 如果是n分类问题, 设第n种情况出现的概率是yn, 则视作n分类问题的输出是(y1, y2, ..., yn),
    那么softmax(y1, y2, y3, ... yn)可以把原本的输出转换成"概率"的输出(可以理解成不仅仅选"最大", 而是所有项参与, 再归一化).
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)

    sigmoid用于二分类, softmax用于多分类.

    :return:
    """
    return None


if __name__ == '__main__':
    loss_test1()
    # loss_test2()

