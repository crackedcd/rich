"""
交叉验证 cross validation
多次更换不同的测试集和验证集, 取平均值作为最终结果
网格搜索 grid search
遍历可能的k值, 进行交叉验证, 最终得出最合适的k值
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def knn_test2():
    iris = load_iris()
    data_train, data_test, target_train, target_test = \
        train_test_split(iris.data, iris.target, test_size=0.2, random_state=4)
    ss = StandardScaler()
    data_train = ss.fit_transform(data_train)
    data_test = ss.transform(data_test)

    # 原本这里的n_neighbors是根据"经验"设置的3, 这里使用grid search cv进行处理.
    k = KNeighborsClassifier()
    neighbor_dict = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "p": [1, 2]}
    gs = GridSearchCV(k, param_grid=neighbor_dict, cv=30)
    gs.fit(data_train, target_train)
    target_predict = gs.predict(data_test)
    print("最佳参数")
    print(gs.best_params_)
    print("最佳结果")
    print(gs.best_score_)
    print("最佳预估器")
    print(gs.best_estimator_)
    print("交叉验证结果")
    print(gs.cv_results_)
    return None


if __name__ == '__main__':
    knn_test2()
