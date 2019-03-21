"""
k nearest neighbors
欧式距离 distance = sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z3)^2)

k太小, 容易受到异常点影响
k太大, 容易受到样本不均衡的影响

优点是无需训练, 简单易用
缺点是计算量大, 内存开销大, 不易确定k的精度
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def knn_test():
    """
    sepal length 萼片长度
    sepal width 萼片宽度
    petal length 花瓣长度
    petal width 花瓣宽度
    iris-setosa 山鸢尾
    iris-versicolour 变色鸢尾
    iris-virginica 维吉尼亚鸢尾
    """
    # 加载数据 data特征值(判断的参数), target目标值(判断的结果/标签)
    iris = load_iris()
    # print(iris)
    # 划分数据集
    # train_test_split的参数是, data特征值(判断的参数), target目标值(判断的结果/标签), test_size是百分比, random_state是随机数种子
    # 返回的的变量是 训练集, 测试集, 训练集目标, 测试集目标
    data_train, data_test, target_train, target_test = \
        train_test_split(iris.data, iris.target, test_size=0.2, random_state=4)
    ss = StandardScaler()
    # 对数据集进行标准化, 抛开浮动较大数据对整体的影响
    data_train = ss.fit_transform(data_train)
    # 在这里, 期望使用[训练集]的平均值和标准差, 来计算[测试集]的标准化结果
    # 也即是, 通过一套相同的平均值和标准差, 来分别计算[训练集]和[测试集], 使得二者被处理的情况一致
    # 所以, 要保留[训练集]用过的[fit], 也就是这里的ss, 用在[测试集]的[transform]上
    data_test = ss.transform(data_test)
    # 使用knn, n_neighbors的值是算法所指的邻居数量, 得到预估结果
    k = KNeighborsClassifier(n_neighbors=3)
    # 这里的k, 是一个预估器, 使用k.fit(训练集, 训练目标)进行训练, 得到拟合模型
    k.fit(data_train, target_train)
    # 使用已经训练好的预估器k, 通过predict()来测试测试集数据, 得到测试集数据的预估目标标签
    target_predict = k.predict(data_test)
    print("原始数据的测试集的标签target_test是")
    print(target_test)
    print("训练得出的预估目标标签target_predict是")
    print(target_predict)
    # 比对预估器生成的目标和原始数据的测试集目标是否一致
    print(target_predict == target_test)
    # 返回测试数据和测试标签的平均精度
    print(k.score(data_test, target_test))
    return None


if __name__ == '__main__':
    knn_test()
