from preprocessing import Preprocessing, Dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from imdb_reader import get_subset

# top words for + vs -
stop_words = text.ENGLISH_STOP_WORDS.union(
    [
        "br",
        "ax",
        "character",
        "characters",
        "did",
        "does",
        "don",
        "film",
        "films",
        "movie",
        "movies",
        "really",
        "like",
        "just",
        "time",
    ]
)


def wordcloud(txt, title):
    cloud = WordCloud(
        stopwords=stop_words,
        collocations=False,
        max_font_size=80,
        background_color="white",
    ).generate(txt)
    plt.figure(figsize=(16, 12))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


imdb_train = get_subset("train")
pos_txt = "".join(imdb_train["data"][12500:])
neg_txt = "".join(imdb_train["data"][:12500])
wordcloud(pos_txt, "positive reviews wordcloud")  # Fig7
wordcloud(neg_txt, "negative reviews wordcloud")  # Fig8
# --------------------------------------------------------------------------

# positive review vs negative review word count
# --------------------------------------------------------------------------

# top 20 words before and after removing stop words
def countplot(x, y, title=""):
    x = np.array(x.sum(axis=0)).flatten()
    x, y = zip(*sorted(zip(x, y), reverse=True))
    df = pd.DataFrame(columns=["Class", "Count"])
    df["Word"] = y
    df["Count"] = x
    df.plot.bar(x="Word", y="Count")
    plt.title(title)
    plt.show()


imdb_p = Preprocessing(Dataset.imdb_reviews, vectorize=False)
twenty_news_p = Preprocessing(Dataset.twenty_news, vectorize=False)

imdb_words_before, imdb_vect_before = imdb_p.custom_vectorizer(
    max_features=20, strip_accents="unicode"
)
imdb_words_after, imdb_vect_after = imdb_p.custom_vectorizer(
    max_features=20,
    stop_words=Preprocessing.stop_words,
    strip_accents="unicode",
    lowercase=True,
)
twenty_words_before, twenty_vect_before = twenty_news_p.custom_vectorizer(
    max_features=20, strip_accents="unicode"
)
twenty_words_after, twenty_vect_after = twenty_news_p.custom_vectorizer(
    max_features=20,
    stop_words=Preprocessing.stop_words,
    strip_accents="unicode",
    lowercase=True,
)

countplot(
    imdb_vect_before,
    imdb_words_before,
    "IMDB dataset before feature engineering and removing stop words",
)  # Fig1
countplot(
    imdb_vect_after,
    imdb_words_after,
    "IMDB dataset after feature engineering and removing stop words",
)  # Fig2
countplot(
    twenty_vect_before,
    twenty_words_before,
    "20news dataset before feature engineering and removing stop words",
)  # Fig3
countplot(
    twenty_vect_after,
    twenty_words_after,
    "20news dataset after feature engineering and removing stop words",
)  # Fig4

imdb_words_after, imdb_vect_after = imdb_p.custom_vectorizer(
    max_features=20,
    ngram_range=(3, 3),
    stop_words=Preprocessing.stop_words,
    strip_accents="unicode",
    lowercase=True,
)
#  Most frequent trigrams IMDB reviews
countplot(imdb_vect_after, imdb_words_after, "IMDB dataset trigrams")  # Fig5

# --------------------------------------------------------------------------
# total count article per newsgroup
preproc = Preprocessing(Dataset.twenty_news)
y = preproc.get_train_test()[1]
counter = Counter(y)

labels, values = zip(*counter.items())
labels, values = zip(*sorted(zip(labels, values)))
indexes = np.arange(len(labels))
plt.bar(indexes, values)
plt.title("number of articles per newsgroup")
plt.xticks(indexes, labels)
plt.show()  # Fig6
