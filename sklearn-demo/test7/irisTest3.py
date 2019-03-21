"""
信息的定义: 消除随机不确定性
信息熵: 消除不确定性的程度的大小, 单位bit, 当测量的是抛硬币形式的两种等概率事件时, 测量的信息量是1bit.
类似于质量, 当我们知道质量的标准单位是1g时, 想知道某个物体A的质量m, 是看它相当于多少(n)个1g的物体,
也就是m = 1 * n, 要计算n, 则使用乘法的反函数, 得到m / 1.

同理可以简单得出结论, 因为相互独立的事件的概率是用指数方法来计算的, 例如3次抛硬币的某种结果概率应该是(1/2)^3,
所以要计算信息的"数量", 应该用指数的反函数, 也就是对数来计算.
例如4个不确定情况是几个硬币抛出的结果? 是log以2为底, 4的对数, 答案是2.
再如8个不确定情况是几个硬币抛出的结果? 是log以2为底, 8的对数, 答案是3.
以此类推.
换成信息论的概念, 也就是, 4个相互独立概率的答案的不确定性, 如果能得到确定, 则信息量是2bit. 8个不确定的答案信息是3bit.
如果各个答案的概率不一致, 那么就分别计算各个答案的信息量, 再乘以各自的概率, 然后相加得到熵.
即 熵 = 概率 * log2为底(信息量的对数)
又因为1%的概率相当于100个等概率事件中确定1个实际情况. 也就是p=1/100, 可以得到结论, 1/p=100, 即概率的倒数等于情况的个数.
用信息概率的倒数替换信息个数, 熵 = 概率 * log2为底(1/概率)
例如:
A B C D四个选项, 都无法确定, 视作概率都是25%, 此时总的熵是2bit. 如果告知正确答案, 正确答案的信息量是2bit.
现在得到一个信息, "有一半的概率是选C", 这个信息会使得A B C D选项的概率变成1/6 1/6 1/2 1/6, 此时,
熵A = 1/6 * log2为底(6) = 0.4308270834535260302422898239913
熵B = 1/6 * log2为底(6) = 0.4308270834535260302422898239913
熵C = 1/2 * log2为底(2) = 0.5
熵D = 1/6 * log2为底(6) = 0.4308270834535260302422898239913
4者的和为1.7924812503605780907268694719739, 也就是知道"有一半的概率是选C"之后, 熵是1.79bit.
所以"有一半的概率是选C"这个信息的信息量是 2bit - 1.79bit = 0.21bit.

决策树 decision tree
决策树的划分依据之一是信息增益, 也就是如果选择一个特征后, 信息增益最大(信息不确定性减少的程度最大), 那么我们就选取这个特征.
在已知结论的情况下, 随机变量x的信息熵为:
H(x) = - sigma(P(x) * log(P(x)))

决策树容易产生过拟合(overfitting)的问题, 解决方法有:
减枝cart
随机森林
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


def decision_tree_test():
    iris = load_iris()
    data_train, data_test, target_train, target_test = \
        train_test_split(iris.data, iris.target, test_size=0.15, random_state=77)
    dt = DecisionTreeClassifier(criterion="entropy")
    dt.fit(data_train, target_train)
    target_predict = dt.predict(data_test)
    print(target_predict == target_test)
    print(dt.score(data_test, target_test))
    # 可视化决策树
    # export_graphviz(dt, out_file=r"./iris_tree.dot", feature_names=iris.feature_names)
    # .dot图文件可以在http://www.webgraphviz.com/查看
    export_graphviz(dt,
                    out_file=r"./iris_tree.dot",
                    feature_names=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    print("graphviz finished!")
    return None


if __name__ == '__main__':
    decision_tree_test()
