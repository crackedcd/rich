"""
找到用户和商品之间的关系
PCA处理的是协方差矩阵, 如果原始矩阵较大, 无法传入稀疏矩阵, 会MemoryError, 要使用SVD代替
"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA


def main():
    # df = pd.read_csv(r"./order_product.csv", encoding="gbk", low_memory=False)
    df_reader = pd.read_csv(r"./order_product.csv", encoding="gbk", iterator=True, chunksize=2000)
    df = pd.DataFrame()
    for chunk in df_reader:
        df = df.append(chunk)
    # 需要处理空值
    print("df完成")
    df_new = pd.DataFrame(df)
    df_new = df_new.fillna("0.0")
    print("df fillna完成")

    # 内容转成字典, 使用DictVectorizer()
    data_dict = df_new.to_dict(orient="records")
    dv = DictVectorizer()
    data = dv.fit_transform(data_dict)
    """
    np包含空值的处理
    读取数据
    train = pd.read_csv('./data/train.csv')
    检查数据中是否有缺失值, False无缺失, True有缺失
    print(pd.isna(train).sum())
    删除有缺失值的行
    train = train.dropna(inplace=True)
    对缺失值进行填充处理
    train = train.fillna('0')
    """
    print(data)
    print(data.shape)
    pca = PCA(n_components=0.95)
    # 一旦data稍大, 这里的.toarray()必然内存不够, 使用SVD代替
    # TruncatedSVD 类似于 PCA, 但tsvd可以使用scipy.sparse稀疏矩阵, 不需要还原成标准矩阵
    # 类似 data_tsvd = tsvd.fit_transform(data)
    data_pca = pca.fit_transform(data.toarray())
    print(data_pca)
    print(data_pca.shape)
    return None


if __name__ == '__main__':
    main()
