# coding: utf-8

from sklearn.feature_extraction import DictVectorizer


def dict_demo():
    data = [
        {
            "city": "北京",
            "value": 100
        },
        {
            "city": "上海",
            "value": 99
        },
        {
            "city": "贵阳",
            "value": 98
        }
    ]
    print(data)
    # 要保证各个"类别"在计算种的"公平", 需要将"类别"转成"数值"进行使用, 也就是进行离散化, 需要进行one-hot编码
    # 在字典转矢量使用one-hot编码时, 若存在大量类别, 则可能有大量0值,
    # 为节约内存, 使用sparse 稀疏矩阵, 只显示非0值, 并且指出保存非0值的位置
    # 一般都直接输出稀疏矩阵, 再用.toarray()方法查看
    dv1 = DictVectorizer(sparse=False)
    data_new1 = dv1.fit_transform(data)
    print(dv1.get_feature_names())
    print(data_new1)
    dv2 = DictVectorizer(sparse=True)
    data_new2 = dv2.fit_transform(data)
    print(data_new2)
    return None


if __name__ == '__main__':
    dict_demo()
