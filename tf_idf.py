import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def generate_topic_df(data, DOC_COL, cluster):
    topic_df = (
        pd.DataFrame({
            'document': data[DOC_COL],
            'topic': cluster.labels_
        })
        .groupby('topic', as_index=False)
        .agg({'document': ' '.join})
    )
    return topic_df


def c_tf_idf(documents, M, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(M, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_by_topic(data, DOC_COL, cluster, n=20, ngram_range=(1,1)):
    # obtain concatenated documents by topic
    topic_df = generate_topic_df(data, DOC_COL, cluster)

    # list of strings - each element is a concatenation of documents of the same topic
    documents = topic_df['document'].to_list()

    # number of (raw, unconcatenated) documents
    M = data.shape[0]

    # compute c-tf-idf and count
    tf_idf, count = c_tf_idf(documents, M, ngram_range=ngram_range)

    # list of all words/terms
    words = count.get_feature_names_out()

    return {
        topic_df['topic'].iloc[i]: pd.DataFrame(
            {'words': words, 'c_tf_idf':  tf_idf[:,i]}
        ).sort_values('c_tf_idf', ascending=False).head(n).reset_index(drop=True) for i in range(topic_df.shape[0])
    }

