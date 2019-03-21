from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import jieba


def naive_bayes_test():
    """
    朴素 + 贝叶斯
    联合概率: 所有条件同时成立的概率 P(A, B)
    条件概率: A在B已经发生的情况下出现的概率 P(A|B)
    相互独立: A和B相互独立, 满足P(A, B) = P(A) * P(B)
    朴素 -> 假设所有条件的概率都是独立概率
    贝叶斯公式:
    W是给定文档的特征值, C是文档类别
    在已经确定了W的前提下, 发生C的概率, 也就是, 在知道文档的词频的情况下, 给出文档分类的"概率"
    P(C|W) = P(W|C) * P(C) / P (W)
    """
    """
    # 总共4个样本, 前3个是中文, 第4个是日文
    data = [
        ["china", "chinese", "shanghai"],
        ["china", "chinese", "guiyang"],
        ["china", "beijing"],
        ["japan", "tokyo"],
    ]
    # 测试样本
    test = ["guiyang", "china", "tokyo"]
    求:
    在"guiyang", "china", "tokyo"的条件下, 文档是中文(C)或者日文(非C)的概率
    P(C|"guiyang", "china", "tokyo") 与 P(非C|"guiyang", "china", "tokyo") 与 
    ↓
    P(C|"guiyang", "china", "tokyo") = 
        P("guiyang", "china", "tokyo"|C) * P(C) / P("guiyang", "china", "tokyo")
    求非C的概率分母是一致的, 所以只算C和非C哪个分子更大即可
    ↓
    其中,
    P("guiyang", "china", "tokyo"|C) = P("guiyang"|C) * P("china"|C) * P("tokyo"|C)
        类型被判定为中文, 也就是C的文档的总词数是8个
        "guiyang": "guiyang"在所有类型判定为C的文档中的词数比例, 也就是1/8
        "china": "china"在所有类型判定为C的文档中的词数比例, 也就是3/8
        "tokyo": "tokyo"在所有类型判定为C的文档中的词数比例, 也就是0/8
        同时, 加入拉普拉斯平滑系数, 也就是 分子 + α, 分母 + α * m, 其中, α一般使用1, m使用训练文档中的特征词个数
        在本例中, 特征词china, chinese, shanghai, guiyang, beijing这5个
        即:
        "guiyang": (1 + 1) / (8 + 5) = 2/13
        "china": (3 + 1) / (8 + 5) = 4/13
        "tokyo": (0 + 1) / (8 + 5) = 1/13
        2/13 * 4/13 * 1/13 = 8/39
    P(C)是"4个文档中有3个是中文文档的概率", 为3/4
    最终结果为 3/4 * 8/39 = 0.15384615384615384615384615384615
    ↓
    
    同理计算非C
    P("guiyang", "china", "tokyo"|非C) = P("guiyang"|非C) * P("china"|非C) * P("tokyo"|非C)
        "guiyang": 0/2 -> 平滑 -> (0+1)/(2+2) -> 1/4
        "china": 0/2 -> 平滑 -> (0+1)/(2+2) -> 1/4
        "tokyo": 1/2 -> 平滑 -> (1+1)/(2+2) -> 2/4
        1/4 * 1/4 * 2/4 = 2/12
    P(非C)是"4个文档中有3个是中文文档的概率", 为1/4
    最终结果为 2/12 * 1/4 = 0.04166666666666666666666666666667
    ↓
    因为C的概率大于非C的概率, 所以该文档是中文文档的概率更大.
    """
    # 加载文件, 第一列是地址, 第二列是分数
    df = pd.read_csv(r"./data.csv", encoding="gbk")
    # 切分样本和测试数据 -> 样本数据, 测试数据, 样本标签, 测试标签
    data_train, data_test, target_train, target_test = train_test_split(df["receipt_address"], df["final_score"])
    # 分词
    data_train = cut_word_list(data_train)
    data_test = cut_word_list(data_test)
    # tf-idf词频计算
    tfv = TfidfVectorizer()
    tfv_train = tfv.fit_transform(data_train)
    tfv_test = tfv.transform(data_test)
    # native bayes
    nb = MultinomialNB()
    # fit样本的数据集和标签集, 得到拟合模型
    nb.fit(tfv_train, target_train)
    # 使用拟合模型计算测试数据的结果
    test_predict = nb.predict(tfv_test)
    # 对比计算结果和测试标签
    print(test_predict == target_test)
    # 使用拟合模型分数对比拟合数据和目标标签匹配的分数
    print(nb.score(tfv_test, target_test))
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
    naive_bayes_test()
