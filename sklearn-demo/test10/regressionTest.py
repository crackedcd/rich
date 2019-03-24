"""
目标值为连续数据时的问题被称作回归问题 regression
线性回归 Linear Regression
逻辑回归 Logistic Regression

房价/销售额度/贷款额度等等问题, 目标值都是连续的

假定一个线性回归函数
y = w1x1 + w2x2 + w3x3 + ... + b
在统计学中, y称作因变量(目标值)(标签), x称作自变量(特征值)(属性), w称作回归系数(权重), b称作偏置
这里的w和x的关系, 其实是一个矩阵乘法, 即 y = wTx + b
(偏重可以也算在矩阵中, b = b * 1 -> 权重是b, 自变量是1)

例如:
期末成绩 = 平时成绩 * 0.3 + 考试成绩 * 0.7
也就是矩阵
[[学生A平时成绩, 学生A考试成绩],
[学生B平时成绩, 学生B考试成绩],
[学生C平时成绩, 学生C考试成绩]]
乘以
[0.3, 0.7]的结果.

狭义的线性模型:
    指满足线性关系的模型, 从简单的几何知识可以知道, 单个特征与目标值的关系肯定是线性关系(y = f(x)是平面直角坐标系)
广义的线性模型:
    可能是非线性关系, 多个特征与目标值的关系不是线性关系, 例如两个特征与目标的关系, 就是平面的关系(z = f(x, y)是笛卡尔坐标系)
    可能是"折线"关系, 例如 y = w1x1 + w2x2^2 + w3x3^3 + ... + b
    也即是, 自变量或回归系数之一是一次(幂)的, 都可以视作线性模型, 不一定是线性关系.

损失函数
假设真实计算方法为 期末成绩 = 平时成绩 * 0.3 + 考试成绩 * 0.7
先任意猜测一些权重, 得到计算方法 期末成绩 = 平时成绩 * 0.82 + 考试成绩 * 0.18
然后通过某种方法, 调整预测值和真实值, 当猜测值和真实值一样时, 就得到了真实计算方法.
猜测值和真实值之间的"差距", 被称为成本函数/损失函数.
在平面直角坐标系中, 线性关系的图形是一根斜线, 也就是猜测方法和真实方法分别是两根斜线,
把每一个样本点在两根斜线上投影的距离求出总和, 即可得到这个"差距". 也就是最小二乘法.

逐渐优化损失函数的方法, 有:
1. 正规方程:
最小二乘法可以将误差方程转化为有确定解的代数方程组(其方程式数目正好等于未知数的个数), 这个有确定解的代数方程组称为最小二乘法估计的正规方程.
(搜索AndrewNG Normal Equation)
2. 梯度下降
反复迭代改进, 直到逼近结果.

GD gradient descent
原始的梯度下降需要计算所有样本的值, 计算量大.
SDG stochastic gradient descent
随机梯度下降在一次迭代时只考虑一个样本, 需要设定正则项参数和迭代数等等, 对特征标准化敏感.
SAG stochastic average gradient
随机平均梯度法优化了收敛速度.
"""

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


def get_data():
    # 数据
    boston = load_boston()
    # 这里可以看到总共有13个特征, 求出的权重数量也应该是13个.
    print("特征数量")
    print(boston.data.shape)
    data_train, data_test, target_train, target_test = \
        train_test_split(boston.data, boston.target, test_size=0.2, random_state=22)
    # 标准化
    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    return data_train, data_test, target_train, target_test


def linear_test():
    data_train, data_test, target_train, target_test = get_data()
    # 预估器
    lr = LinearRegression()
    lr.fit(data_train, target_train)
    # 结果
    print("正规方程")
    print("权重系数")
    print(lr.coef_)
    print("偏置")
    print(lr.intercept_)
    print("均方误差")
    target_predict = lr.predict(data_test)
    err = mean_squared_error(target_test, target_predict)
    print(err)
    return None


def sgd_test():
    data_train, data_test, target_train, target_test = get_data()
    # 预估器
    sgd = SGDRegressor()
    sgd.fit(data_train, target_train)
    # 结果
    print("梯度下降")
    print("权重系数")
    print(sgd.coef_)
    print("偏置")
    print(sgd.intercept_)
    print("均方误差")
    target_predict = sgd.predict(data_test)
    err = mean_squared_error(target_test, target_predict)
    print(err)
    return None


if __name__ == '__main__':
    linear_test()
    sgd_test()
