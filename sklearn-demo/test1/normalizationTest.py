from sklearn.preprocessing import MinMaxScaler
from pandas.core.frame import DataFrame


def normalization_demo():
    """
    适合小量精准数据
    range指归一化的区间, 一般使用0~1, 即range_max = 1, range_min = 0
    x_new = (x - min) / (max - min)
    x_final = x_new * (range_max - range_min) + range_min
    在这个一般例子中, x_final = x_new

    例如 100, 80, 60, 10 归一化
    结果是:
    (100 - 10) / (100 - 10) = 1
    (80 - 10) / (100 - 10) = 0.7778
    (60 - 10) / (100 - 10) = 0.5556
    (10 - 10) / (100 - 10) = 0
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
        [10493, 1, 1, 1, 1, 1],
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
    df = DataFrame(data, columns=["pm_count", "ord_count", "name_count", "addr_count", "chan_count", "item_count"])
    # print(df)
    # 不设feature_range默认就是0~1
    mms = MinMaxScaler(feature_range=[0, 1])
    data_new = mms.fit_transform(data)
    print(data_new)
    return None


if __name__ == '__main__':
    normalization_demo()
