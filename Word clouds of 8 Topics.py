import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis
import numpy as np

# 指定输出文件夹路径
output_path = 'C:/LDA/zhutijianmo/result'  # 输出路径
# 指定数据文件夹路径
file_path = 'C:/LDA/zhutijianmo/data'  # 数据路径

# 切换工作目录到数据路径
os.chdir(file_path)

# 读取 Excel 文件中的数据
try:
    data = pd.read_excel("dataZT1.xlsx")  # 文件名及后缀
except FileNotFoundError:
    print("Error: Data file not found.")
    exit()

# 切换工作目录到输出路径
os.chdir(output_path)

# 加载停用词文件
stop_file = "C:/LDA/zhutijianmo/stop_dic/stopwords.txt"
try:
    with open(stop_file, 'r', encoding='UTF-8') as f:
        stop_words = f.read().splitlines()
except FileNotFoundError:
    print("Error: Stopwords file not found.")
    exit()
stop_words = stop_words + ['102', '104', '12', '20', '302', '304', 'ain', 'aren', 'called', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'll', 'mon', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']
# 将停用词转换为小写并去除多余空白
stop_words = [word.strip().lower() for word in stop_words if word.strip()]

# 加载自定义词汇表（从 dict.txt 文件中），这些词是需要重点分析的
dict_file = "C:/LDA/zhutijianmo/stop_dic/dict.txt"
try:
    with open(dict_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        # 使用正则表达式按任何空白字符分割（包括空格、制表符、换行符）
        custom_words = re.split(r'\s+', content)
        custom_words = [word.strip().lower() for word in custom_words if word.strip()]
        # 移除重复的词汇
        custom_words = list(set(custom_words))
except FileNotFoundError:
    print("Error: Dictionary file not found.")
    exit()

# 从停用词列表中移除 dict.txt 文件中的词，确保这些词不会被过滤掉
stop_words = [word for word in stop_words if word not in custom_words]

# 删除所有数字
data['content'] = data['content'].astype(str).apply(lambda x: re.sub(r'\d+', '', x))

# 统计 dict.txt 中词的出现频率
def get_word_frequencies(corpus, custom_words):
    # 使用正则表达式确保仅匹配完整词汇
    vectorizer = CountVectorizer(vocabulary=custom_words)
    word_matrix = vectorizer.fit_transform(corpus)
    word_counts = word_matrix.sum(axis=0).A1
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts))
    return word_freq

# 获取 dict.txt 中词汇的出现频率
word_frequencies = get_word_frequencies(data['content'], custom_words)

# 输出 dict.txt 中词的高频出现情况
print("High frequency words from dict.txt:")
for word, freq in sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True):
    print(f"{word}: {freq}")

# ## 2. LDA分析

def print_top_words(model, feature_names, n_top_words):
    """
    输出每个主题的关键词
    :param model: LDA 模型
    :param feature_names: 词汇表
    :param n_top_words: 每个主题中显示的前 n 个词
    :return: 每个主题的关键词列表
    """
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        tword.append(" ".join(topic_words))
        print(" ".join(topic_words))
    return tword

# 设置特征提取器的参数，min_df=10 表示至少出现在10个文档中的词将被保留
n_features = 2000  # 提取1000个特征词语
tf_vectorizer = CountVectorizer(
    strip_accents='unicode',
    max_features=n_features,
    stop_words=stop_words,  # 使用自定义停用词，确保保留 dict.txt 中的词
    max_df=0.5,
    min_df=10,  # 至少出现在10个文档中的词才会保留
    lowercase=True
)

# 对文本进行词频向量化处理
tf = tf_vectorizer.fit_transform(data['content'])

# 设置主题数
n_topics = 8

# 创建 LDA 模型
lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=50,
    learning_method='batch',
    learning_offset=50,
    random_state=0
)
# 拟合 LDA 模型
lda.fit(tf)

# ### 2.1 输出每个主题对应的关键词及其频率和概率

n_top_words = 50  # 每个主题的前 n 个关键词
tf_feature_names = tf_vectorizer.get_feature_names_out()  # 获取词汇表

# 保存每个主题的关键词及其频率和概率
topic_details = []

for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic #{topic_idx + 1}:")
    topic_words = []
    for i in topic.argsort()[:-n_top_words - 1:-1]:
        word = tf_feature_names[i]
        freq = topic[i]
        prob = freq / topic.sum()
        topic_words.append((word, freq, prob))
        print(f"Word: {word}, Frequency: {freq:.2f}, Probability: {prob:.4f}")
    topic_details.append(pd.DataFrame(topic_words, columns=["Word", "Frequency", "Probability"]))

# 保存每个主题的关键词及其频率和概率
with pd.ExcelWriter("processed_dataZT1_topic_words_details.xlsx") as writer:
    for idx, df in enumerate(topic_details):
        print(f"Saving Topic #{idx+1} to Excel")
        df.to_excel(writer, sheet_name=f"Topic_{idx+1}", index=False)

# ### 2.2 输出每篇文章对应的主题

# 对每篇文章进行主题分布预测
topics = lda.transform(tf)

# 将每篇文章分配到概率最大的主题
topic = []
for t in topics:
    topic.append("Topic #" + str(list(t).index(np.max(t)) + 1))  # 加1使主题编号从1开始
# 添加列：每篇文章的概率最大的主题序号
data['概率最大的主题序号'] = topic
# 添加列：每篇文章的主题概率分布
data['每个主题对应概率'] = list(topics)

# 保存每篇文章的主题分布
print("Saving document-topic assignments to Excel")
data.to_excel("processed_dataZT1_word.xlsx", index=False)

# ### 2.3 可视化

# 获取词汇表
vocab = tf_vectorizer.get_feature_names_out()

# 计算词频
term_frequency = np.array(tf.sum(axis=0)).flatten()

# 计算文档长度
doc_lengths = np.array(tf.sum(axis=1)).flatten()

# 计算主题-词分布矩阵
topic_term_dists = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

# 计算文档-主题分布矩阵
doc_topic_dists = lda.transform(tf)

# 准备 pyLDAvis 数据
pic = pyLDAvis.prepare(
    topic_term_dists,
    doc_topic_dists,
    doc_lengths,
    vocab,
    term_frequency
)

# 保存 pyLDAvis 可视化为 HTML 文件
pyLDAvis_output_file = os.path.join(output_path, f'lda_pass_{n_topics}.html')
pyLDAvis.save_html(pic, pyLDAvis_output_file)

# ### 3. 生成词云图

# 生成词云图的部分
from PIL import Image

# 创建词云输出文件夹
wordcloud_output_path = "C:/LDA/zhutijianmo/result"
os.makedirs(wordcloud_output_path, exist_ok=True)

# 生成每个主题的词云图
for i, topic in enumerate(lda.components_):
    # 生成词汇表与主题词频的字典
    topic_words = {tf_feature_names[j]: topic[j] for j in topic.argsort()[:-n_top_words - 1:-1]}

    # 创建词云对象
    wordcloud = WordCloud(
        width=300,  # 词云图宽度
        height=300,  # 词云图高度
        background_color='white',  # 背景颜色
        max_words=n_top_words,  # 最大显示的词数量
        max_font_size=80,  # 最大字体大小
        scale=3,  # 缩放比例
        random_state=42  # 随机种子
    ).generate_from_frequencies(topic_words)

    # 将词云转换为PIL图像对象
    wordcloud_image = wordcloud.to_image()

    # 显示词云图
    plt.figure(figsize=(5, 5))  # 设置图形大小
    plt.imshow(wordcloud_image, interpolation='bilinear')  # 显示词云图
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图形

    # 保存词云图到文件
    wordcloud_image.save(os.path.join(wordcloud_output_path, f"topic_processed_dataZT1_{i+1}.png"))
