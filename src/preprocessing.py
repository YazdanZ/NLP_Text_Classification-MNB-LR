from enum import Enum
import pandas as pd
import imdb_reader
import random
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

# Lemmatisation tokenizer from the Natural Language Toolkit (nlk)
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class Dataset(Enum):
    twenty_news = 1
    imdb_reviews = 2


class Preprocessing:
    """Aquire, Analyze and Clean data
    We are using the following datasets:
    - Dataset 1: 20 news group dataset:
    - Dataset 2: IMDB Reviews:
    """

    def __init__(self, dataset, frac=1, vectorize=True):
        """aquire data"""
        # fraction of the training set to take (used in part3 experiments)
        self.frac = frac

        self.dataset = dataset
        if dataset == Dataset.imdb_reviews:
            self._preprocess_imbd()
        elif dataset == Dataset.twenty_news:
            self._preprocess_20news()
        else:
            raise IndexError("Dataset enumeration between 1 and 2")

        # feature engineering before vectorization and lemmatizing
        self._clean_train()

        # take a random sample (fraction = self.frac) of the train data
        sample_n = len(self.train["data"])
        idx = np.random.choice(
            np.arange(sample_n), int(sample_n * self.frac), replace=False
        )
        self.train_X = np.array(self.train["data"])[idx]
        self.train_label = np.array(self.train["target"])[idx]
        self.test_label = self.test["target"]

        if vectorize:
            self._vectorize_transform()

    def _clean_train(self):
        # convert all tabs, endlines, carriage returns to simple space delimiters
        self.train["data"] = [word.replace("\n", " ") for word in self.train["data"]]
        self.train["data"] = [word.replace("\r", " ") for word in self.train["data"]]
        self.train["data"] = [word.replace("\t", " ") for word in self.train["data"]]
        self.train["data"] = [word.replace('"', " ") for word in self.train["data"]]
        self.train["data"] = [word.lower() for word in self.train["data"]]

        # replace punctuation signs with space delimiters
        punctuation_signs = list("?:!.,;@*-=")
        for sign in punctuation_signs:
            self.train["data"] = [
                word.replace(sign, " ") for word in self.train["data"]
            ]
        # 's => " "
        self.train["data"] = [word.replace("'s", " ") for word in self.train["data"]]

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
        print("\nVectorizing and lemmatizing dataset...")
        vectorizer = CountVectorizer(
            # A good idea would be to analyze feature selection over here
            # also add a regex pattern
            #  token_pattern=r'(?u)\b\w\w+\b',
            #  max_features=20,
            stop_words="english",
            strip_accents="unicode",
            lowercase=True,
            #  tokenizer=LemmaTokenizer(),
        )
        vectorized_train = vectorizer.fit_transform(self.train_X)
        vectorized_test = vectorizer.transform(self.test["data"])
        train_cols = vectorizer.get_feature_names()
        print("Number of unique words :", len(train_cols))

        # term frequency–inverse document frequency tranformation
        print(
            "Transforming vector using term frequency–inverse document frequency transformation..."
        )
        tfidf_transformer = TfidfTransformer()
        self.transformed_train = tfidf_transformer.fit_transform(vectorized_train)
        self.transformed_test = tfidf_transformer.transform(vectorized_test)

        try:
            # create the dataframes
            self.train_df = pd.DataFrame.sparse.from_spmatrix(self.transformed_train)
            self.test_df = pd.DataFrame.sparse.from_spmatrix(self.transformed_test)
            self.train_df.columns = train_cols
            self.test_df.columns = train_cols

            # also add the labels associated with each text
            self.train_df["Class"] = self.train_label
            self.test_df["Class"] = self.test["target"]
        except MemoryError:
            print("Not enought memory to create pandas dataframe")

    def get_train_test(self):
        """get train and test transformed features (bag of words)
        and their respective labels"""
        return (
            self.transformed_train,
            self.train_label,
            self.transformed_test,
            self.test_label,
        )

    def custom_vectorizer(self, **kwargs):
        vectorizer = CountVectorizer(**kwargs)
        vectorized_train = vectorizer.fit_transform(self.train_X)
        feature_names = vectorizer.get_feature_names()
        return feature_names, vectorized_train
