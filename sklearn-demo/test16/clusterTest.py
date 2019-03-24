"""
利用训练数据(已经做过的题)，使机器能够利用它们(解题方法)分析未知数据(考试题目)的能力, 叫做"学习"(learning).
以分类(classification)为例, 对于分类问题, 输入的训练数据有特征(feature), 有标签(label),
所谓的学习, 其本质就是找到特征(data)和标签(target)间的关系(mapping).

如果所有训练数据都有标签, 则为有监督学习(supervised learning).
如果数据没有标签, 则为无监督学习(unsupervised learning), 也即聚类(clustering).
如果部分数据有标签, 但大部分都没有, 则称为半监督学习(semi-supervised learning).

K-means K均值聚类
步骤:
1. 随机设置k个特征空间内的点作为初始的聚类中心(k的数量可以根据需求指定, 或根据网格搜索调节超参数选择合适的k值)
2. 找到剩余所有样本点与k点的距离, 距离最近的标为一类, 分为k个类别
3. 对标出的类别取均值, 得到新的k个中心
4. 如果步骤3中选取出的中心和步骤1中的中心完全一致(视具体情况也可以取相对近似), 则完成, 若不是, 则以步骤3中选取出的中心重复整套计算.

K-means一般用于没有目标值的情况, 通过聚类得到目标值(将无监督学习转为监督学习).
K-means是一个迭代式的算法, 算法简单, 但容易陷入到局部最优(可以使用多次聚类或者模拟退火等方式规避).
"""

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score


def k_means_test():
    df = pd.read_csv(r"./order_product.csv", encoding="gbk", low_memory=False)
    df_new = pd.DataFrame(df)
    df_new = df_new.fillna("0.0")
    data_dict = df_new.to_dict(orient="records")
    dv = DictVectorizer()
    data = dv.fit_transform(data_dict)
    tsvd = TruncatedSVD(n_components=2, algorithm="randomized")
    data_tsvd = tsvd.fit_transform(data)
    print(data_tsvd)
    print(data_tsvd.shape)
    # n_cluster是k的取值
    km = KMeans(n_clusters=3)
    km.fit(data_tsvd)
    km_predict = km.predict(data_tsvd)
    print(km_predict)
    """
    轮廓系数
    聚类的结果好, 直观上可以认为是, 类与类之间的距离较远, 同一类中的各个样本距离较近.
    设某个样本i到本类的样本点的平均值为a
    设某个样本i到其他类的样本点的最小值为b
    轮廓系数为 s = (b - a) / max(a, b)
    若b远远大于a, 则说明样本点到其他类的距离远远大于其与同一类的样本点的距离, 说明聚类效果好, s可以近似为1,
    若b远远小于a, 则说明样本点到其他类的距离远远小于其与同一类的样本点的距离, 说明聚类效果差, s可以近似为-1,
    可以看出, 轮廓系数的范围是-1~1, 越接近-1, 聚类效果越差, 越接近1, 聚类效果越好.
    """
    s = silhouette_score(data_tsvd, km_predict)
    print("轮廓系数为")
    print(s)
    return None


if __name__ == '__main__':
    k_means_test()
