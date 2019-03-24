"""
预测结果 Predicted Condition
正确标记 True Condition
都为真, 称作真正例TP True Positive
都为假, 称作真反例TN True Negative
若预测为真, 但正确为假, 称作伪正例 FP False Positive
若预测为假, 但正确为真, 称作伪反例 TN True Negative

精确率 Precision
预测结果为正例的结果中, 真正例的比率, 即 TP / (TP + FP), 预测的准确率

召回率 Recall
真实结果为正例的结果中, 真正例的比率, 即 TP / (TP + FN), 预测的覆盖率

F1-score
F1 = (2 * 精确率 * 召回率) / (精确率 + 召回率)
解得 F1 = 2TP / (2TP + FN + FP)
反映模型的稳健性, 如果F1值大, 说明精确率和召回率都相对较高
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def true_condition():
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

    # 精确率 召回率 F1-score
    # labels 指target结果的范围, target_names 指target结果的名称
    report = classification_report(target_test, target_predict, labels=[2, 4], target_names=["良性", "恶性"])
    # 结果中的support表示数量
    print(report)
    return None


if __name__ == '__main__':
    true_condition()
