"""
欠拟合 指 模型过于简单
通过增加数据特征数量来优化

过拟合 指 模型过于复杂
通过正则化的方法, 减少嘈杂特征
L1正则化(LASSO回归)
使得部分w权重直接变成0, 剔除这些特征的影响
L2正则化(Ridge回归)
使得部分w权重减小, 削弱这些特征的影响
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def ridge_test():
    # 数据
    boston = load_boston()
    data_train, data_test, target_train, target_test = \
        train_test_split(boston.data, boston.target, test_size=0.15, random_state=7)
    # 标准化
    scaler = StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    # alpha 正则化力度 惩罚项系数 0~1 1~10
    # solver 选择优化方法 auto自动 sag优化收敛的随机梯度下降
    # normalize 进行标准化
    # 预估器
    ridge = Ridge(alpha=0.5, max_iter=10000)
    ridge.fit(data_train, target_train)
    # 结果
    print("权重系数")
    print(ridge.coef_)
    print("偏置")
    print(ridge.intercept_)
    print("均方误差")
    target_predict = ridge.predict(data_test)
    err = mean_squared_error(target_test, target_predict)
    print(err)
    return None


if __name__ == '__main__':
    ridge_test()
