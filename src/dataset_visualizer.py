from preprocessing import Preprocessing, Dataset
from sklearn.datasets import fetch_20newsgroups
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# top 10 words for + vs -


# positive review vs negative review word count

# top 20 words before and after removing stop words
def countplot(x, y, title=""):
    x = np.array(x.sum(axis=0)).flatten()
    x, y = zip(*sorted(zip(x, y), reverse=True))
    df = pd.DataFrame(columns=["Class", "Count"])
    df["Class"] = y
    df["Count"] = x
    df.plot.bar(x="Class", y="Count")
    plt.show()


#  imdb_p = Preprocessing(Dataset.imdb_reviews, vectorize=False)
#  twenty_news_p = Preprocessing(Dataset.twenty_news, vectorize=False)
#
#  imdb_words_before, imdb_vect_before = imdb_p.custom_vectorizer(
#  max_features=20, strip_accents="unicode"
#  )
#  imdb_words_after, imdb_vect_after = imdb_p.custom_vectorizer(
#  max_features=20, stop_words="english", strip_accents="unicode", lowercase=True
#  )
#  twenty_words_before, twenty_vect_before = twenty_news_p.custom_vectorizer(
#  max_features=20, strip_accents="unicode"
#  )
#  twenty_words_after, twenty_vect_after = twenty_news_p.custom_vectorizer(
#  max_features=20, stop_words="english", strip_accents="unicode", lowercase=True
#  )
#  countplot(imdb_vect_before, imdb_words_before)  # Fig1
#  countplot(imdb_vect_after, imdb_words_after)  # Fig2
#  countplot(twenty_vect_before, twenty_words_before)  # Fig3
#  countplot(twenty_vect_after, twenty_words_after)  # Fig4

# total count article per newsgroup
#  preproc = Preprocessing(Dataset.twenty_news)
#  y = preproc.get_train_test()[1]
#  counter = Counter(y)
#
#  labels, values = zip(*counter.items())
#  labels, values = zip(*sorted(zip(labels, values)))
#  indexes = np.arange(len(labels))
#  plt.bar(indexes, values)
#  plt.xticks(indexes, labels)
#  plt.show()  # Fig5
