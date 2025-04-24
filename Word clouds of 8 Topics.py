import os
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pyLDAvis
import numpy as np

output_path = 'C:/LDA/zhutijianmo/result'  
file_path = 'C:/LDA/zhutijianmo/data'  

os.chdir(file_path)

try:
    data = pd.read_excel("dataZT1.xlsx")  
except FileNotFoundError:
    print("Error: Data file not found.")
    exit()

os.chdir(output_path)

stop_file = "C:/LDA/zhutijianmo/stop_dic/stopwords.txt"
try:
    with open(stop_file, 'r', encoding='UTF-8') as f:
        stop_words = f.read().splitlines()
except FileNotFoundError:
    print("Error: Stopwords file not found.")
    exit()
stop_words = stop_words + ['102', '104', '12', '20', '302', '304', 'ain', 'aren', 'called', 'couldn', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'll', 'mon', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn']
stop_words = [word.strip().lower() for word in stop_words if word.strip()]

dict_file = "C:/LDA/zhutijianmo/stop_dic/dict.txt"
try:
    with open(dict_file, 'r', encoding='UTF-8') as f:
        content = f.read()
        custom_words = re.split(r'\s+', content)
        custom_words = [word.strip().lower() for word in custom_words if word.strip()]
        custom_words = list(set(custom_words))
except FileNotFoundError:
    print("Error: Dictionary file not found.")
    exit()

stop_words = [word for word in stop_words if word not in custom_words]

data['content'] = data['content'].astype(str).apply(lambda x: re.sub(r'\d+', '', x))

def get_word_frequencies(corpus, custom_words):
    vectorizer = CountVectorizer(vocabulary=custom_words)
    word_matrix = vectorizer.fit_transform(corpus)
    word_counts = word_matrix.sum(axis=0).A1
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts))
    return word_freq

word_frequencies = get_word_frequencies(data['content'], custom_words)

print("High frequency words from dict.txt:")
for word, freq in sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True):
    print(f"{word}: {freq}")

def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        topic_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        tword.append(" ".join(topic_words))
        print(" ".join(topic_words))
    return tword

n_features = 2000  
tf_vectorizer = CountVectorizer(
    strip_accents='unicode',
    max_features=n_features,
    stop_words=stop_words,
    max_df=0.5,
    min_df=10,
    lowercase=True
)

tf = tf_vectorizer.fit_transform(data['content'])

n_topics = 8

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=50,
    learning_method='batch',
    learning_offset=50,
    random_state=0
)
lda.fit(tf)

n_top_words = 50  
tf_feature_names = tf_vectorizer.get_feature_names_out()

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

with pd.ExcelWriter("processed_dataZT1_topic_words_details.xlsx") as writer:
    for idx, df in enumerate(topic_details):
        print(f"Saving Topic #{idx+1} to Excel")
        df.to_excel(writer, sheet_name=f"Topic_{idx+1}", index=False)

topics = lda.transform(tf)

topic = []
for t in topics:
    topic.append("Topic #" + str(list(t).index(np.max(t)) + 1))
data['概率最大的主题序号'] = topic
data['每个主题对应概率'] = list(topics)

print("Saving document-topic assignments to Excel")
data.to_excel("processed_dataZT1_word.xlsx", index=False)

vocab = tf_vectorizer.get_feature_names_out()

term_frequency = np.array(tf.sum(axis=0)).flatten()

doc_lengths = np.array(tf.sum(axis=1)).flatten()

topic_term_dists = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

doc_topic_dists = lda.transform(tf)

pic = pyLDAvis.prepare(
    topic_term_dists,
    doc_topic_dists,
    doc_lengths,
    vocab,
    term_frequency
)

pyLDAvis_output_file = os.path.join(output_path, f'lda_pass_{n_topics}.html')
pyLDAvis.save_html(pic, pyLDAvis_output_file)

from PIL import Image

wordcloud_output_path = "C:/LDA/zhutijianmo/result"
os.makedirs(wordcloud_output_path, exist_ok=True)

for i, topic in enumerate(lda.components_):
    topic_words = {tf_feature_names[j]: topic[j] for j in topic.argsort()[:-n_top_words - 1:-1]}

    wordcloud = WordCloud(
        width=300,
        height=300,
        background_color='white',
        max_words=n_top_words,
        max_font_size=80,
        scale=3,
        random_state=42
    ).generate_from_frequencies(topic_words)

    wordcloud_image = wordcloud.to_image()

    plt.figure(figsize=(5, 5))
    plt.imshow(wordcloud_image, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    wordcloud_image.save(os.path.join(wordcloud_output_path, f"topic_processed_dataZT1_{i+1}.png"))
