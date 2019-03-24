import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib


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

    # ---------------------------------------------------------
    # 保存拟合好的模型
    joblib.dump(lr, r"./lr.pkl")
    lr_model = joblib.load(r"./lr.pkl")
    # ---------------------------------------------------------

    # 权重系数
    print(lr_model.coef_)
    # 偏置
    print(lr_model.intercept_)
    # 模型评估
    target_predict = lr_model.predict(data_test)
    print(target_predict == target_test)
    print(lr_model.score(data_test, target_test))

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
