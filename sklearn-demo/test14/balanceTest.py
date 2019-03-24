"""
以癌症判断为例, 假设原本的样本有100个, 真实的正确标记是99个癌症, 1个正常, 将癌症视作正例的话,
假设不进行任何科学预测, 而是直接把100个都胡乱猜测为正例, 则:
精确率为 99%
召回率为 100%
F1-score为 2 * 0.99 * 1 / (0.99 + 1) = 1.98 / 1.99 = 0.99497487437185929648241206030151
在这种情况下, 因为样本不够均衡, 精确率/召回率/F1-score都无法用来判断真实的模型能力, 会认为胡乱猜测的结果预测很准确.

TPR True Positive Rate 真正例比率
TP / (TP + FN)
所有真实为1的样本中, 预测也为1的比率(其实就是召回率)
FPR False Positive Rate 伪正例比率
FP / (FP + TN)
所有真实为0的样本中, 预测为1的比率

以FPR为横坐标, TPR为纵坐标, 构造平面直角坐标系. 绘制每一个点的FPR和TPR, 得到的曲线被称作ROC(Receiver Operating Characteristic)曲线
ROC曲线和横纵坐标围成的面积, 被称作AUC(Area Under the Curve)指标, AUC也就是预测正样本大于预测负样本的比率
如果在某一点上,
若TPR == FPR, 该点的斜率为1, 则斜线正好是与横纵轴夹角都为45°的斜线,
    该情况可视作是无论真实类别为0或1, 预测成1的概率都相等, 也就是random guess
    该情况下, ROC曲线也就是random guess线与横纵坐标围成的面积是直角三角形, 面积为0.5
若TPR > FPR, 则斜线陡峭,
    该情况下, 若所有真实为1的样本预测为1, 且所有真实为0的样本中, 都不预测为1, 则斜线变成Y轴, AUC值达到最大, 即面积为正方形, 面积为1
若TPR < FPR, 则斜线平缓
    该情况下, AUC小于0.5, 可以反向使用模型.
最终可以认为, AUC的范围是在0.5~1之间, 越接近1越好.

以上面的极端情况为例(假设原本的样本有100个, 真实的正确标记是99个癌症, 1个正常),
此时的TPR为召回率 100%,
此时的FPR为(真实为0的样本数1中, 预测为1的样本数1) 1 / 1 = 100%
此时的ROC曲线为random guess, AUC为0.5, 可以看出其实是胡乱猜测的.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


def balance_test():
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

    # y_true参数必须c处理成取值为0或1的正反例
    # 该例子中将患癌症作为正例1
    target_predict_positive = np.where(target_predict == 4, 1, 0)
    auc = roc_auc_score(target_predict_positive, target_predict)
    print("auc指标为")
    print(auc)
    return None


if __name__ == '__main__':
    balance_test()
