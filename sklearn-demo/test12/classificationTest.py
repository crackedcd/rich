"""
目标值为离散数据时的问题被称作分类问题 classification

逻辑回归 logistic regression
这是一种分类算法
广告点击率(是否被点击)/垃圾邮件识别/是否患病/虚假账号等等问题, 目标值都是离散的(甚至是二分的)

sigmoid函数(激活函数)
1 / (1 + e^(-x))
其中, x是线性回归的输出(即 y = w1x1 + w2x2 + w3x3 + ... + b, 即:
1 / (1 + e^(-(w1x1 + w2x2 + w3x3 + ... + b)))
逻辑回归的最终分类(二分)是基于概率的. 通过sigmoid函数计算后, 结果会被落在0~1这个区间.
然后, 设定一个阈值, 假设是0.5, 把>=0.5的视作1, <0.5的视作0, 就将一个线性的问题转化成了二分的问题.

对数似然损失
在线性回归中, 损失函数的计算方式可以视作 sigma (y_predict - y_true)^2 / count,
但在二分的问题中, 无法找到具体的y_true, 只能找到"类别", 无法确定线性的损失函数, 只能使用分段函数(对数似然损失).
若 y = 0
则 cost = - log(1 - 回归(x))
若 y = 1
则 cost = - log(回归(x))
也就是, 先用线性回归, 然后把线性回归的结果使用sigmoid函数映射到0~1区间, 再根据阈值确定二分, 损失使用对数似然随时函数进行计算.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def classification_test():
    """
    数据来源:
    https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
    """
    df = pd.read_csv(r"./breast-cancer-wisconsin_data.csv",
                     names=["Sample code number",
                            "Clump Thickness",
                            "Uniformity of Cell Size",
                            "Uniformity of Cell Shape",
                            "Marginal Adhesion",
                            "Single Epithelial Cell Size",
                            "Bare Nuclei",
                            "Bland Chromatin",
                            "Normal Nucleoli",
                            "Mitoses",
                            "Class"])
    df = df.replace(to_replace="?", value=np.nan)
    # 删除空值
    df.dropna(inplace=True)
    # 检查是否包含空值
    df.isnull().any()
    # 分出data和target
    data = df.iloc[:, :-1]
    target = df["Class"]
    # 拆分
    data_train, data_test, target_train, target_test = train_test_split(data, target)
    # 标准化
    sd_scaler = StandardScaler()
    data_train = sd_scaler.fit_transform(data_train)
    data_test = sd_scaler.transform(data_test)
    # 线性回归
    lr = LogisticRegression()
    lr.fit(data_train, target_train)
    # 权重系数
    print(lr.coef_)
    # 偏置
    print(lr.intercept_)
    # 模型评估
    target_predict = lr.predict(data_test)
    print(target_predict == target_test)
    print(lr.score(data_test, target_test))
    return None


if __name__ == '__main__':
    classification_test()
