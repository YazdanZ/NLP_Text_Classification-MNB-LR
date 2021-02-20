from enum import Enum
import pandas as pd
import imdb_reader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


class Dataset(Enum):
    twenty_news = 1
    imdb_reviews = 2


class Preprocessing:
    """Aquire, Analyze and Clean data
    We are using the following datasets:
    - Dataset 1: 20 news group dataset:
    - Dataset 2: IMDB Reviews:
    """

    def __init__(self, dataset):
        """aquire data"""
        self.dataset = dataset
        if dataset == Dataset.imdb_reviews:
            self._preprocess_imbd()
        elif dataset == Dataset.twenty_news:
            self._preprocess_20news()
        else:
            raise IndexError("Dataset enumeration between 1 and 2")

        self._vectorize_transform()

    def _preprocess_20news(self):
        print("Preprocessing the 20 newsgroup dataset...\n")

        self.train = fetch_20newsgroups(
            subset="train",
            shuffle=True,
            random_state=0,
            remove=(["headers", "footers", "quotes"]),
        )
        self.test = fetch_20newsgroups(
            subset="test",
            shuffle=True,
            random_state=0,
            remove=(["headers", "footers", "quotes"]),
        )
        print("Training set contains {n} instances".format(n=len(self.train.data)))
        print("Testing set contains {n} instances".format(n=len(self.test.data)))

    def _preprocess_imbd(self):
        print("Preprocessing the IMDB reviews dataset...\n")

        self.train = imdb_reader.get_subset("train")
        self.test = imdb_reader.get_subset("test")
        print("Training set contains {n} instances".format(n=len(self.train["data"])))
        print("Testing set contains {n} instances".format(n=len(self.test["data"])))

    def _vectorize_transform(self):
        """Turn the text data into numerical feature using the bags of words representation"""

        # count vectorization
        # use the built-in english stopwords
        print("Vectorizing dataset...")
        self.vectorizer = CountVectorizer(stop_words="english")
        vectorized_train = self.vectorizer.fit_transform(self.train["data"])
        vectorized_test = self.vectorizer.transform(self.test["data"])
        train_cols = self.vectorizer.get_feature_names()

        # term frequency–inverse document frequency tranformation
        print(
            "Transforming vector using term frequency–inverse document frequency transformation..."
        )
        self.tfidf_transformer = TfidfTransformer()
        self.transformed_train = self.tfidf_transformer.fit_transform(vectorized_train)
        self.transformed_test = self.tfidf_transformer.transform(vectorized_test)

        # create the dataframes
        self.train_df = pd.DataFrame(self.transformed_train.toarray())
        self.test_df = pd.DataFrame(self.transformed_test.toarray())
        self.train_df.columns = train_cols
        self.test_df.columns = train_cols

        # also add the labels associated with each text
        self.train_df["Class"] = self.train["target"]
        self.test_df["Class"] = self.test["target"]


p = Preprocessing(Dataset.imdb_reviews)
# print(p.train_df)
# print(p.test_df)
# y = p.train_df.pop("Class").to_numpy()
# X = p.train_df.to_numpy()
# y_test = p.test_df.pop("Class").to_numpy()
# X_test = p.test_df.to_numpy()

# clf = MultinomialNB()
# clf.fit(X, y)
#
# pred = clf.predict(X_test[110:150])
# print(pred)
# print(accuracy_score(y_test[110:150], pred))
