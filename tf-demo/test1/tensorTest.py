"""
张量 tensor
"张量"即是多维数组, 用"阶"(D)表示张量的维数
0D -> 0阶张量 -> 标量 scalar -> s = 1
1D -> 1阶张量 -> 矢量 vector -> v = [1, 2, 3]
2D -> 2阶张量 -> 矩阵 matrix -> m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
nD -> n阶张量 -> 张量 tensor -> t = [[[ ... ]]]

数据类型
tf.float32  tf.int32 ...

神经网络(neural network, NN)实现过程
1. 准备数据, 提取特征
2. 数据输入到NN, 先搭建graph, 然后执行session(前向传播)
3. 大量输入得到大量输出, 把每次的输出与标准答案的差异再次传给NN, 调整NN的参数直到模型达到要求(反向传播)
4. 得到优化好参数的模型(训练好的模型), 对新数据进行模型预测

准备 -> 前传 -> 反传 -> 迭代

神经元 neuron
在NN neural network中, 一个节点被称作一个neuron

计算图 graph
搭建神经网络的计算过程, 只搭建, 不运算

会话 session
执行计算图中的节点运算, 执行global_variables_initializer()这个赋值相当于执行了输入节点.

"""


import tensorflow as tf


def test1():
    """
    简单计算
    :return:
    """
    a = tf.constant([1.0, 2.0])
    b = tf.constant([3.0, 4.0])
    y = a + b
    # 这里的y是graph
    print(y)
    with tf.Session() as sess:
        result = sess.run(y)
    # 这里的result是session控制graph计算的结果
    print(result)

    # x 一行两列
    x = tf.constant([[1.0, 2.0]])
    # w 两行一列
    w = tf.constant([[3.0], [4.0]])
    # 矩阵乘法
    y = tf.matmul(x, w)
    with tf.Session() as sess:
        """
        run()该方法运行TensorFlow计算的一个"步骤(step)"
        评估`fetches`中的tensor, 替换`feed_dict`中的输入值, 执行指定的`Operation`来运行graph.
        `fetches`参数可以是单个graph, 也可以是任意的嵌套list, tuple, namedtuple, dict或是叶子上包含图形的OrderedDict.
        """
        result = sess.run(y)
    print(y)
    print(result)


def test2():
    """
    生成参数
    :return:
    """
    """
    tf.random_normal(): 正态分布(normal distribution),
        类似的还有tf.truncated_normal()去掉过大偏离点的正态分布, tf.random_uniform()平均分布
        除了随机数, 还可以生成常量, 例如:
            tf.zeros 全0
                tf.zeros([3, 2], int32) 生成 [[0, 0], [0, 0], [0, 0]]
            tf.ones 全1
                tf.zeros([3, 2], int32) 生成 [[1, 1], [1, 1], [1, 1]]
            tf.fill 全定值
                tf.fill([3, 2], 7) 生成 [[7, 7], [7, 7], [7, 7]]
            tf.constant 直接赋值
                tf.constant([3, 2, 1]) 直接生成[3, 2, 1]
    [2, 3] 2行3列
    stddev 标准差
    mean 平均数
    seed 随机数种子, 如果去掉, 每次生成的随机数就不一致
    """
    a = tf.random_normal([2, 3], stddev=2, mean=0, seed=1)
    print(a)
    w = tf.Variable(a)
    print(w)
    return None


if __name__ == '__main__':
    print("------------------------------------------------")
    test1()
    print("------------------------------------------------")
    test2()
    print("------------------------------------------------")
