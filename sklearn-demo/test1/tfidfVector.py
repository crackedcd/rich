from sklearn.feature_extraction.text import TfidfVectorizer
import jieba


def tfidf_demo():
    """
    tf: term frequency 词频, 给定词语在文本中出现的频率
    idf: inverse document frequency 逆向文档频率, 用总文件数目含该词语的文件数目, 得到的商取以10为底的对数
    df-idf: 两者相乘

    假设在1000篇文章中, 每篇文章都是10000字,
    有500篇出现了"非常", 100篇出现了"经济", 50篇出现了"计算机".
    出现"非常"的文章中, "非常"出现了1000次,
    出现"经济"的文章中, "经济"出现了500次,
    出现"计算机"的文章中, "计算机"出现了500次,
    则在这些文章中:

    "非常"的tf是 1000次 / 10000字 = 0.1
    "经济"的tf是 500次 / 10000字 = 0.05
    "计算机"的tf是 500次 / 10000字 = 0.05

    "非常"的idf是1000篇 / 500篇 = 2, 取对数的结果是(log以10为底, 取2的对数) ln2/ln10 = 0.301
    "经济"的idf是1000篇 / 100篇 = 10, 取对数的结果是(log以10为底, 取10的对数) ln10/ln10 = 1
    "计算机"的idf是1000篇 / 50篇 = 20, 取对数的结果是(log以10为底, 取20的对数) ln20/ln10 = 1.301

    "非常"的tf-idf是 0.1 * 0.301 = 0.0301
    "经济"的tf-idf是 0.05 * 1 = 0.05
    "计算机"的tf-idf是 0.05 * 1.301 = 0.06505
    """
    data = [
        "中文文本特征这里是一个苹果苹果苹果",
        "中文文本特征这里是一个梨",
        "中文文本特征这些是一些桔子桔子"
    ]
    data = cut_word_list(data)
    tfidfv = TfidfVectorizer()
    data_new = tfidfv.fit_transform(data)
    print(tfidfv.get_feature_names())
    print(data_new.toarray())
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
    tfidf_demo()
