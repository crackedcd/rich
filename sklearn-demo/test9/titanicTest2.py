"""
集成学习方法
生成多种分类器/模型, 各自独立学习和预测, 最后再结合成组合预测, 使得最终结果优于单一预测结果.

随机森林
包含多个决策树的分类器, 其输出是多个决策树输出的众数决定(投票). 其特点是训练集随机, 特征也随机.

bootstrap抽样, 随机有放回的抽样:
原始训练集有N个样本, M个特征.
训练集随机, 是指从N个样本中, 随机有放回抽样出N个.
特征随机, 是指从M个特征中, 随机抽取m个特征, 且m远远小于M, 形成决策树. m相对小的话, 一方面可以降维, 另一方面使得错误的决策树互相抵消.
假设训练集包含5个样本[1, 2, 3, 4, 5], 随机抽出一个样本, 假设是2, 得到测试样本2, 再将2放回, 重抽, 也就是抽出的测试样本可能因为"放回"出现重复,
这样可以使得每棵树的结果都有选择到随机样本的权利, 不"片面", 最后再投票表决.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import pandas as pd


def get_data():
    df = pd.read_csv(r"./titanic.txt")
    data = df[["pclass", "age", "sex"]]
    target = df["survived"]
    data["age"].fillna(data["age"].mean(), inplace=True)
    data_dict = data.to_dict(orient="records")
    dv = DictVectorizer()
    dv_data = dv.fit_transform(data_dict)
    data_train, data_test, target_train, target_test = train_test_split(dv_data, target, test_size=0.15, random_state=6)
    return data_train, data_test, target_train, target_test


def titanic_gs_test2():
    data_train, data_test, target_train, target_test = get_data()
    rf = RandomForestClassifier()
    rf_params = {
        "n_estimators": [60, 80, 100, 120, 140, 160],
        "criterion": ["gini", "entropy"],
        "max_depth": [1, 3, 5, 7, 9]
    }
    gs = GridSearchCV(rf, param_grid=rf_params, cv=3)
    gs.fit(data_train, target_train)
    target_predict = gs.predict(data_test)
    print(target_predict == target_test)
    print(gs.score(data_test, target_test))
    print("最佳参数")
    print(gs.best_params_)
    print("最佳结果")
    print(gs.best_score_)
    print("最佳预估器")
    print(gs.best_estimator_)
    print("交叉验证结果")
    print(gs.cv_results_)
    return None


def titanic_test2():
    data_train, data_test, target_train, target_test = get_data()
    rf = RandomForestClassifier(n_estimators=120, max_depth=5, criterion="gini")
    rf.fit(data_train, target_train)
    target_predict = rf.predict(data_test)
    print(target_predict == target_test)
    print(rf.score(data_test, target_test))
    return None


if __name__ == '__main__':
    # titanic_gs_test2()
    titanic_test2()
