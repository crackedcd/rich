import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def grid_search_test():
    # df = pd.read_csv(r"./data2.csv", encoding="gbk")
    df = pd.read_csv(r"./data.csv", encoding="gbk")
    time_value = pd.DatetimeIndex(df["create_time"])
    df["day"] = time_value.day
    df.loc[:, "month"] = time_value.month
    df.loc[:, "hour"] = time_value.hour
    df.loc[:, "minute"] = time_value.minute
    df_new = df[["month", "day", "hour", "minute", "final_score"]]
    data = df_new[["month", "day", "hour", "minute"]]
    target = df_new["final_score"]
    data_train, data_test, target_train, target_test = train_test_split(data, target)
    ss = StandardScaler()
    data_train = ss.fit_transform(data_train)
    data_test = ss.transform(data_test)

    k = KNeighborsClassifier()
    neighbor_dict = {"n_neighbors": [3, 6, 9]}
    gs = GridSearchCV(k, param_grid=neighbor_dict, cv=10)
    gs.fit(data_train, target_train)
    target_predict = gs.predict(data_test)
    print(target_predict)
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
    grid_search_test()
