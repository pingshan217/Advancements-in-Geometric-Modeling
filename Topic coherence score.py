import os
import pandas as pd
import re
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt

def preprocess_text(text, stop_file):
    """
    Preprocess text by removing stopwords/punctuation and retaining meaningful words
    :param text: Input text string
    :param stop_file: Path to stopwords file
    :return: List of processed tokens
    """
    if not isinstance(text, str):
        text = ""

    try:
        with open(stop_file, 'r', encoding='UTF-8') as f:
            stop_words = set(line.strip() for line in f)
    except IOError:
        stop_words = set()
        print("Error reading stop_file")

    words = re.sub(r'\W+', ' ', text).lower().split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    return filtered_words

def run_lda_analysis(data_path, stop_file, output_path):
    data = pd.read_excel(data_path)
    data["content_cutted"] = data['content'].apply(lambda text: preprocess_text(text, stop_file))
    
    dictionary = Dictionary(data['content_cutted'])
    corpus = [dictionary.doc2bow(text) for text in data['content_cutted']]

    coherence_values = []
    for n_topics in range(2, 25):
        lda_model = LdaModel(
            corpus=corpus,
            num_topics=n_topics,
            id2word=dictionary,
            passes=40,
            random_state=0
        )
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=data['content_cutted'],
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_values.append(coherence_model.get_coherence())

    plt.plot(range(2, 25), coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("LDA Topic Consistency Score")
    plt.savefig(os.path.join(output_path, 'coherence_scores.png'))
    plt.show()

    with open(os.path.join(output_path, 'coherence_scores.txt'), 'w') as f:
        for i, score in enumerate(coherence_values, start=2):
            f.write(f"Topics: {i}, Coherence Score: {score}\n")

if __name__ == '__main__':
    data_path = 'C:/LDA/zhutijianmo/data/dataZT2.xlsx'
    stop_file = 'C:/LDA/zhutijianmo/stop_dic/stopwords.txt'
    output_path = 'C:/LDA/zhutijianmo/result'
    
    os.makedirs(output_path, exist_ok=True)
    run_lda_analysis(data_path, stop_file, output_path)
