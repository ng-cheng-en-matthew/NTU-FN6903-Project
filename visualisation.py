import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from tf_idf import generate_topic_df, extract_top_n_words_by_topic



def get_symptom_count_df(data, query_str=None, SYMP_COL='SYMP_STR'):
    # regex pattern to split symptoms
    pattern = re.compile("^\s+|\s*,\s*|\s+$")
    if query_str != None:
        data = data.query(query_str)

    # collapse all reported symptoms into single array
    symptom_lst = np.concatenate(
        data[SYMP_COL]
        .astype(str)
        .str.split(pattern)
        .to_numpy()
    )

    # obtain counts of symptoms
    symptom_count_dict = dict(Counter(symptom_lst))
    symptom_count_df = (
        pd.DataFrame.from_dict(symptom_count_dict, orient='index', columns=['Count'])
        .reset_index()
        .sort_values('Count', ascending=False)
    )

    return symptom_count_df


def visualise_topics(data, DOC_COL, cluster, n=10, n_split=3, figsize=(10,20),
                     img_dir=None, file_name=None):
    # create dataframe of concatenated documents by topics
    topic_df = generate_topic_df(data, DOC_COL, cluster)

    # create dictionary of dataframes of tf-idf scores by topic
    top_n_words_dict = extract_top_n_words_by_topic(data, DOC_COL, cluster, n=n)

    # number of topics in a chunk
    chunk_size = -(topic_df.shape[0] // -n_split)

    # split number of topics into chunks
    for m in range(n_split):
        print('Printing chunk')
        start_idx, end_idx = m*chunk_size, min((m+1)*chunk_size, topic_df.shape[0])
        fig, axes = plt.subplots(end_idx-start_idx, 2, figsize=figsize)

        for i, topic in enumerate(list(top_n_words_dict.keys())[start_idx:end_idx]):
            ax = axes[(i, 0)]
            sns.barplot(y='words', x='c_tf_idf', data=top_n_words_dict[topic], ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(f'Topic {topic}')

            ax = axes[(i, 1)]
            wc = WordCloud(background_color='white').generate(
                topic_df.loc[topic_df['topic'] == topic, 'document'].iloc[0])
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")

        # save images
        if (img_dir != None) and (file_name != None):
            plt.savefig(os.path.join(img_dir, f'{file_name}_chunk_{m+1}.pdf'), bbox_inches='tight')

        plt.show()

