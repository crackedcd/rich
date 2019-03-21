from sklearn.preprocessing import StandardScaler
from pandas.core.frame import DataFrame


def main():
    """
    适合存在可能影响最大最小值的异常点的大量数据
    归一化处理异常值偏差较大的情况时容易出问题, 需要使用标准化
    标准化将原始数据处理到均值为0, 标准差为1的范围
    x_final = (x - avg) / sigma
    也即是, (x - 该列平均值) / 标准差
    标准差的计算为:
    1. 计算平均值 avg
    2. 计算方差, 即该列所有值 (x1 - avg)^2 + (x2 - avg)^2 + (x3 - avg)^2 + ... + (xn - avg)^2 / (n - 1)
    3. 方差开方的结果即是标准差 sigma
    可以理解成, 计算单个数值的过程是fit, 合并结果的过程是transform
    """
    data = [
        [8994, 1, 1, 1, 1, 1],
        [17988, 2, 1, 1, 1, 1],
        [17988, 1, 1, 1, 1, 1],
        [62958, 2, 1, 1, 1, 1],
        [53964, 3, 1, 3, 1, 2],
        [2998, 1, 1, 1, 1, 1],
        [17988, 1, 1, 1, 1, 1],
        [17988, 1, 1, 1, 1, 1],
        [17988, 1, 1, 1, 1, 1],
        [7495, 1, 1, 1, 1, 1],
        [10494, 1, 1, 1, 1, 1],
        [134910, 11, 1, 1, 1, 1],
        [17988, 1, 1, 1, 1, 1],
        [2998, 1, 1, 1, 1, 1],
        [8994, 1, 1, 1, 1, 1],
        [46469, 7, 1, 1, 1, 1],
        [8994, 2, 1, 1, 1, 1],
        [2998, 1, 1, 1, 1, 1],
        [2998, 2, 1, 1, 1, 1],
        [2998, 1, 1, 1, 1, 1]
    ]
    df = DataFrame(data, columns=["pm_count", "ord_count", "name_count", "addr_count", "chan_count", "item_count"]) # print(df)
    ss = StandardScaler()
    data_new = ss.fit_transform(df)
    print(data_new)
    return None


if __name__ == '__main__':
    main()
