"""
急切执行 eager execution
该模式下, 一旦被调用就会立即执行, 省去了构造graph并传给session执行的过程, 无需Session.run(), tf2.0后会当成默认模式.
开启:
tf.enable_eager_execution(), 一旦开启, 在这个python会话中就无法关闭.
禁用:
https://www.tensorflow.org/api_docs/python/tf/disable_eager_execution
v1.13 -> tf.disable_eager_execution()
https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/compat/v1/disable_eager_execution
v2.0 -> tf.compat.v1.disable_eager_execution()

前向传播 FP forward propagation
搭建模型, 实现推理(输入层 -> 隐藏层 -> 输出层)
设w为待优化参数, x为输入数据, y为输出结果

"""

import tensorflow as tf


def forward_propagation_test1():
    """
    零件重量体积demo

    假设生产一批零件, 有体积和重量两个元素, 经过下面的神经网络后得到一个数值. (输入2个数字, 隐藏层3个数字, 输出1个数字)

            ?
    体积
            ?       ?
    重量
            ?

    用一个[1, 2]的矩阵表示x, 也就是一行两列的矩阵表示一次输入的两个特征(体积和重量)
    用w(前节点, 后节点)(层数)表示待优化的参数, 假设只有一层隐藏层, 在第一层w的前节点有2个, 后节点有3个, 那么w为一个[2, 3]的矩阵
        w(1) = [[w(1,1)(1), w(1,2)(1), w(1,3)(1)], [w(2,1)(1), w(2,2)(1), w(2,3)(1)]]
    第一层的结果 a(1) = x * w(1) = tf.matmul(x, w1)
        w(2) = [[w(1, 1)(2)], [[w(1, 2)(2)], [[w(1, 3)(2)]]
    输出的结果 y = a(1) * w(2) = tf.matmul(a, w2)
    :return:
    """

    # 定义输入和参数
    x = tf.constant([[0.7, 0.5]])
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # 定义前向传播过程
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    # session
    with tf.Session() as sess:
        # 变量初始化, 因为上面的"定义"只生成了graph, 需要传到session中执行真正的赋值, 相当于init_op是初始节点
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        result = sess.run(y)
        print(result)
    return None


def forward_propagation_test2():
    """
    先预留x的占位, 使用feed喂入数据, 其余同上.
    :return:
    """
    # tf.placeholder 占位, shape(一次喂几组数据, 一组数据几个数), 当数据组数不确定时, 可以shape(None, 一组数据几个数)
    x = tf.placeholder(tf.float32, shape=(1, 2))
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # feed_dict喂数据
        result = sess.run(y, feed_dict={x: [[0.7, 0.5]]})
        print(result)
    return None


def forward_propagation_test3():
    """
    喂入多组数据, 其余同上.
    :return:
    """
    x = tf.placeholder(tf.float32, shape=(None, 2))
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        feeds = {
            x: [
                [0.7, 0.5],
                [0.2, 0.3],
                [0.3, 0.4],
                [0.4, 0.5]
            ]
        }
        result = sess.run(y, feed_dict=feeds)
        print(result)
        print("第一层的参数w1为")
        print(sess.run(w1))
        print("第二层的参数w2为")
        print(sess.run(w2))
    return None


if __name__ == '__main__':
    forward_propagation_test1()
    forward_propagation_test2()
    forward_propagation_test3()
