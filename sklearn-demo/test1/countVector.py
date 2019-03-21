from sklearn.feature_extraction.text import CountVectorizer


def count_demo():
    data = [
        "this is an apple",
        "this is a pear",
        "these are oranges"
    ]
    # 使用stop_words传入列表, 使用停用词
    # cv = CountVectorizer(stop_words=["apple"])
    cv = CountVectorizer(stop_words=["apple"])
    data_new = cv.fit_transform(data)
    print(cv.get_feature_names())
    print(data_new.toarray())
    print(data_new)
    return None


if __name__ == '__main__':
    count_demo()
