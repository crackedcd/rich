"""
正则化缓解过拟合

正则化在损失函数中引入模型复杂度指标, 利用给w加权, 抑制训练数据的噪声
loss = loss(y, y_) + regularizer * loss(w)
其中, regularizer是参数w在总loss中的比例, 也就是正则化的权重

L1正则化 loss = sigma(|w|)
loss = tf.contrib.layers.l1_regularizer(regularizer)(w)
L2正则化 loss = sigma(w^2)
loss = tf.contrib.layers.l2_regularizer(regularizer)(w)

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def regularization_test():
    batch_size = 30
    seed = 2
    rdm = np.random.RandomState(seed)
    # 随机生成300行2列的矩阵, 表示300个坐标点, 作为输入数据集
    mat_x = rdm.randn(300, 2)
    # 区分出两种类型的点, 一种"靠近中心"(x0^2 + x1^2) < 2, 一种"远离中心"(x0^2 + x1^2) >= 2
    # 将"靠近中心"的点标记为1
    # mat_y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in mat_x]
    mat_y_ = []
    for x0, x1 in mat_x:
        if x0 * x0 + x1 * x1 < 2:
            mat_y_.append(1)
        else:
            mat_y_.append(0)
    # 将mat_y中的1和0分别指定颜色, "靠近中心"的为红色, "远离中心"的为蓝色, 便于matplotlib画图
    y_color = [["red" if y == 1 else "blue"] for y in mat_y_]

    # print(mat_x)
    # print(mat_y_)
    # 进行reshape处理
    """
    np.vstack() 沿着竖直方向将矩阵堆叠起来
    ** 
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    res = np.vstack((arr1, arr2))
    结果:
    array([[1, 2, 3],
           [4, 5, 6]])
    
    np.hstack() 沿着水平方向将数组堆叠起来
    **
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    res = np.hstack((arr1, arr2))
    结果:
    [1 2 3 4 5 6]
    **
    arr1 = np.array([[1, 2], [3, 4], [5, 6]])
    arr2 = np.array([[7, 8], [9, 0], [0, 1]])
    res = np.hstack((arr1, arr2))
    结果:
    array([[1 2 7 8]
          [3 4 9 0]
          [5 6 0 1]])
    
    np.reshape(row, col) 更改数组的形状, -1在行数表示不限制行数, 只用col限制列数
    reshape生成的新数组和原始数组公用一个内存
    """
    mat_x = np.vstack(mat_x).reshape(-1, 2)
    mat_y_ = np.vstack(mat_y_).reshape(-1, 1)

    # plt绘图
    # plt.scatter(x坐标, y坐标, c="颜色")
    plt.scatter(mat_x[:, 0], mat_x[:, 1], c=np.squeeze(y_color))
    plt.show()

    # nn
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    # 第一层, 隐藏层
    # w1是2行11列, 这里的行是根据x的坐标点是x0和x1两个值来的, 11是随便设置的神经元数量
    # 正则化权重是0.01
    w1 = get_weight([2, 11], 0.01)
    # 这里的11和weight对应
    b1 = get_bias([11])
    # 使用激活函数tf.nn.relu
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # 第二层, 输出层
    # w2是11行1列, 因为第一层有11个神经元, 但输出层只有1个神经元
    w2 = get_weight([11, 1], 0.01)
    b2 = get_bias([1])
    # 输出层不过激活函数
    y2 = tf.matmul(y1, w2) + b2
    # 损失函数
    loss_mse = tf.reduce_mean(tf.square(y2 - y_))
    loss_regular = loss_mse + tf.add_n(tf.get_collection("losses"))
    # 反向传播方法
    lr = 0.0001
    # 不含正则化
    train_step = tf.train.AdamOptimizer(lr).minimize(loss_mse)
    # 含正则化
    # train_step = tf.train.AdamOptimizer(lr).minimize(loss_regular)

    # 运行
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = 50000
        for i in range(steps):
            start = (i * batch_size) % 300
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
        probs = sess.run(y2, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)
        print("w1: ", sess.run(w1))
        print("b1: ", sess.run(b1))
        print("w2: ", sess.run(w2))
        print("b2: ", sess.run(b2))

    plt.scatter(mat_x[:, 0], mat_x[:, 1], c=np.squeeze(y_color))
    # plt.contour(x轴坐标值, y轴坐标值, 点的高度, levels=[等高线的高度])
    plt.contour(xx, yy, probs, levels=[0.5])
    plt.show()

    return None


def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # 将每个w的正则化损失(l2正则化), 加到总损失"losses"中
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


if __name__ == '__main__':
    # 修改反向传播方法train_step使用的损失函数, 对比使用正则化与否的结果
    regularization_test()
