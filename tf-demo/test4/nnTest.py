"""
x1 --w1--> |
x2 --w2--> |
x3 --w3--> | --> x_new
...        |
b  ------> |

x_new = f(sigma(x * w) + b)
f activation function 激活函数, 引入激活函数, 是为了引入非线性, 提高模型表达力.
w weight 权重
b bias 偏置

relu: f(x) = max(x, 0)
sigmoid: f(x) = 1 / (1+e^(-x))
tanh: f(x) = (1-e^(-2x)) / (1+e^(-2x))

NN的层数: 因为输入层没有计算, 所以层数 = 隐藏层 + 输出层
NN的复杂度: NN的总参数 = 总w + 总b

输入层3个节点     隐藏层4个节点     输出层2个节点
    *               *               *
    *               *               *
    *               *
                    *
以此为例
层数为2
总参数有26个:
(第一层, 即隐藏层有3行4列个w, 4个b) (第二层, 即输出层有4行2列个w, 2个b)
(3*4 + 4) + (4*2 + 2) = 16 + 10 = 26

"""

import tensorflow as tf


def nn_test():
    """
    构建最简单的神经网络, 假设存在一个模型, 能够在输入x1和x2之后, 得到结果y. 在满足x1和x2是1的情况下, y=0.3
    即nn为:
    x1 -> |
          | -> y
    x2 -> |
    (无中间隐藏层)
    :return:
    """

    # 所有x在一个矩阵里, 这里是一个1行2列的矩阵, 表示1组数据, 这组数据有2个值
    x = tf.placeholder(tf.float32, [1, 2])
    # y_已知结果
    y_ = tf.placeholder(tf.float32, [1])
    # 所有w也在一个矩阵里, 这里是一个2行1列的矩阵
    w = tf.Variable(tf.random_normal([2, 1]), name="weight")
    b = tf.Variable(tf.random_normal([1]), name="biases")
    # sigmoid
    y = tf.sigmoid(tf.matmul(x, w) + b)

    loss = -tf.reduce_sum(y_ * tf.log(y) + (1 - y_) * tf.log(1 - y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        print("训练前的预期")
        print(sess.run(y, feed_dict={x: [[1, 1]]}))
        feeds = {
            x: [
                [1, 1]
            ],
            y_: [
                0.3
            ]
        }
        for i in range(10000):
            sess.run(train_step, feed_dict=feeds)
            if i % 1000 == 0:
                print("训练%d次后的预期" % i)
                print(sess.run(y, feed_dict={x: [[1, 1]]}))

        print("训练完成后的预期")
        print(sess.run(y, feed_dict={x: [[1, 1]]}))

    return None


if __name__ == '__main__':
    nn_test()
