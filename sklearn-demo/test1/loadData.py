# coding: utf-8

from sklearn import datasets
from sklearn.model_selection import train_test_split


def main():
    iris = datasets.load_iris()
    # print(iris)
    # print(iris.DESCR)
    # 特征值 -> 数据属性
    # 目标值 -> 标签
    # 训练集特征值, 测试集特征值, 训练集目标值, 测试集目标值
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=1)
    print(x_train)
    print(x_test)
    print(x_train.shape)
    print(x_test.shape)
    print(y_train)
    print(y_test)


if __name__ == '__main__':
    main()
