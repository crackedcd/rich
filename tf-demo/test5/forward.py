"""
前向传播

# 搭建结构(输入参数x, 正则化权重r), 定义整个过程, 返回输出y
# > test4/nnTest
def forward(x, regularizer):
    w =
    b =
    y =
    return y

# 生成权重(w形状, 正则化权重)
def get_weight(shape, regularizer):
    w = tf.Variable()
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

# 生成偏置(b形状)
def get_bias(shape):
    b = tf.Variable()
    return b

"""

import tensorflow as tf


def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # 将每个w的正则化损失(l2正则化), 加到总损失"losses"中
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


def forward(x, r):
    """
    :param x: 输入层x
    :param r: 正则化权重regularizer
    :return: 输出层y
    """
    # 第一层, 隐藏层
    # w1是2行11列, 这里的行是根据x的坐标点是x0和x1两个值来的, 11是随便设置的神经元数量
    w1 = get_weight([2, 11], r)
    # 这里的11和weight对应
    b1 = get_bias([11])
    # 使用激活函数tf.nn.relu
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # 第二层, 输出层
    # w2是11行1列, 因为第一层有11个神经元, 但输出层只有1个神经元
    w2 = get_weight([11, 1], r)
    b2 = get_bias([1])
    # 输出层不过激活函数
    y2 = tf.matmul(y1, w2) + b2
    return y2


if __name__ == '__main__':
    forward()
