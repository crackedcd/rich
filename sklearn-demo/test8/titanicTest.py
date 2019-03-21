"""
思路:
目标是找出拟合模型, 找出拟合模型的前提是得到特征值(指标)和目标值(标签)
1. 获取数据后, 需要将数据处理成可拟合的形式, 包括:
    1.1. 缺失值处理
    1.2. 类别 -> 字典one-hot编码
2. 数据集划分成样本和测试集
3. 标准化
4. 模型评估
"""


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import export_graphviz
import pandas as pd


def titanic_test():
    df = pd.read_csv(r"./titanic.txt")
    data = df[["pclass", "age", "sex"]]
    target = df["survived"]
    """
    pandas方法                                     说明
    count                      非NA值的数量
    describe                  针对Series或各DataFrame列计算汇总统计
    min,max                 计算最小值和最大值
    argmin,argmax        计算能够获取到最小值和最大值的索引位置（整数)
    idxmin,idxmax         计算能够获取到最小值和最大值的索引值
    quantile                   计算样本的分位数（0到 1） 
    sum                        值的总和
    mean                      值的平均数， a.mean() 默认对每一列的数据求平均值；若加上参数a.mean(1)则对每一行求平均值
    media                      值的算术中位数（50%分位数)
    mad                         根据平均值计算平均绝对离差
    var                          样本值的方差 
    std                        样本值的标准差
    skew                     样本值的偏度（三阶矩）
    kurt                       样本值的峰度（四阶矩）
    cumsum                 样本值的累计和
    cummin,cummax    样本值的累计最大值和累计最小
    cumprod                样本值的累计积
    diff                        计算一阶差分（对时间序列很有用) 
    pct_change            计算百分数变化
    """
    data["age"].fillna(data["age"].mean(), inplace=True)
    data_dict = data.to_dict(orient="records")
    dv = DictVectorizer()
    dv_data = dv.fit_transform(data_dict)
    data_train, data_test, target_train, target_test = train_test_split(dv_data, target, test_size=0.15, random_state=6)
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    dt.fit(data_train, target_train)
    target_predict = dt.predict(data_test)
    print(target_predict == target_test)
    print(dt.score(data_test, target_test))
    export_graphviz(dt, out_file=r"./titanic_tree.dot", feature_names=dv.get_feature_names())
    print("graphviz finished!")
    return None


if __name__ == '__main__':
    titanic_test()
