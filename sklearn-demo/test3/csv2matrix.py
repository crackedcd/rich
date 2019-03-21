import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def main():
    df = pd.read_csv(r"./1.csv", encoding="gbk")
    print(df)
    dd = df.to_dict(orient="records")
    print(dd)
    dv = DictVectorizer()
    data = dv.fit_transform(dd)
    print(dv.get_feature_names())
    print(data.toarray())
    return None


if __name__ == '__main__':
    main()
