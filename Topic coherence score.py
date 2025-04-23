import os  # 用于操作系统功能，如文件路径操作
import pandas as pd  # 数据处理库，用于处理数据框
import re  # 正则表达式库，用于文本处理
from gensim.corpora.dictionary import Dictionary  # Gensim中的字典类，用于创建词汇表
from gensim.models.ldamodel import LdaModel  # Gensim中的LDA模型类
from gensim.models.coherencemodel import CoherenceModel  # Gensim中的一致性模型类，用于计算主题模型的一致性得分
import matplotlib.pyplot as plt  # 绘图库，用于绘制一致性得分图

def preprocess_text(text, stop_file):
    """
    对英文文本进行预处理：去除停用词、标点符号，并仅保留有意义的词
    :param text: 输入的文本字符串
    :param stop_file: 停用词文件路径
    :return: 预处理后的词列表
    """
    # 如果文本是非字符串类型（例如float或者NaN），将其转换为空字符串
    if not isinstance(text, str):
        text = ""

    # 读取停用词文件
    try:
        with open(stop_file, 'r', encoding='UTF-8') as f:
            stop_words = set(line.strip() for line in f)  # 读取停用词文件并存储在集合中
    except IOError:
        stop_words = set()  # 如果文件读取失败，使用空的停用词集合
        print("Error reading stop_file")

    # 分词处理，去除停用词和非字母字符
    words = re.sub(r'\W+', ' ', text).lower().split()  # 使用正则表达式去除非字母字符，并将文本转为小写后分词
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]  # 去除停用词并过滤长度小于等于1的词
    return filtered_words  # 返回处理后的词列表


def run_lda_analysis(data_path, stop_file, output_path):
    """
    运行LDA分析并计算一致性得分
    :param data_path: 数据文件路径
    :param stop_file: 停用词文件路径
    :param output_path: 结果输出路径
    """
    # 读取Excel文件
    data = pd.read_excel(data_path)  # 使用Pandas读取Excel文件

    # 对文本进行预处理
    data["content_cutted"] = data['content'].apply(lambda text: preprocess_text(text, stop_file))  # 对每个文档进行预处理

    # 生成Gensim字典和语料库
    dictionary = Dictionary(data['content_cutted'])  # 创建词汇表
    corpus = [dictionary.doc2bow(text) for text in data['content_cutted']]  # 将文档转换为词袋模型（Bag of Words）

    # 计算一致性得分
    coherence_values = []
    for n_topics in range(2, 25):  # 对主题数量从2到17进行迭代
        lda_model = LdaModel(corpus=corpus, num_topics=n_topics, id2word=dictionary, passes=40, random_state=0)  # 训练LDA模型
        coherence_model = CoherenceModel(model=lda_model, texts=data['content_cutted'], dictionary=dictionary, coherence='c_v')  # 计算一致性得分
        coherence_values.append(coherence_model.get_coherence())  # 将一致性得分存入列表

    # 绘制一致性得分图
    plt.plot(range(2, 25), coherence_values)  # 绘制主题数量与一致性得分的关系图
    plt.xlabel("Number of Topics")  # 设置X轴标签
    plt.ylabel("Coherence Score")  # 设置Y轴标签
    plt.title("LDA Topic Consistency Score")  # 设置图表标题
    plt.savefig(os.path.join(output_path, 'coherence_scores.png'))  # 保存图表为PNG文件
    plt.show()  # 显示图表

    # 保存一致性得分到文件
    with open(os.path.join(output_path, 'coherence_scores.txt'), 'w') as f:  # 将一致性得分保存到文本文件
        for i, score in enumerate(coherence_values, start=2):  # 枚举并写入每个主题数量对应的一致性得分
            f.write(f"Topics: {i}, Coherence Score: {score}\n")

if __name__ == '__main__':
    # 文件路径（新手只需替换这些路径即可）
    data_path = 'C:/LDA/zhutijianmo/data/dataZT2.xlsx'  # 数据文件路径
    stop_file = 'C:/LDA/zhutijianmo/stop_dic/stopwords.txt'  # 停用词文件路径
    output_path = 'C:/LDA/zhutijianmo/result'  # 输出结果路径

    # 创建输出目录（如果不存在）
    os.makedirs(output_path, exist_ok=True)  # 创建输出目录

    # 运行LDA分析
    run_lda_analysis(data_path, stop_file, output_path)  # 调用主函数执行LDA分析
