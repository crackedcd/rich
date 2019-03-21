from sklearn.feature_extraction.text import CountVectorizer
import jieba


def count_demo():
    data = [
        "中文文本特征这里是一个苹果苹果苹果",
        "中文文本特征这里是一个梨",
        "中文文本特征这些是一些桔子桔子"
    ]
    data = cut_word_list(data)
    # 使用stop_words传入列表, 使用停用词
    # cv = CountVectorizer(stop_words=["apple"])
    cv = CountVectorizer()
    data_new = cv.fit_transform(data)
    print(cv.get_feature_names())
    print(data_new.toarray())
    # print(data_new)
    return None


def cut_word(text):
    it = jieba.cut(text)
    return " ".join(it)


def cut_word_list(text_list):
    result = []
    for line in text_list:
        result.append(cut_word(line))
    return result


if __name__ == '__main__':
    count_demo()
