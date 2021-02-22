from enum import Enum
import pandas as pd
import imdb_reader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from classifier import Classifier
from cross_validation import cross_validation_split, kfoldCV
from multinomial_nb import MultinomialNB as mnb


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

    def __init__(self, dataset):
        """aquire data"""
        self.dataset = dataset
        if dataset == Dataset.imdb_reviews:
            self._preprocess_imbd()
        elif dataset == Dataset.twenty_news:
            self._preprocess_20news()
        else:
            raise IndexError("Dataset enumeration between 1 and 2")

        self.train_label = self.train["target"]
        self.test_label = self.test["target"]
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
        self.vectorizer = CountVectorizer(
            # A good idea would be to analyze feature selection over here
            # also add a regex pattern
            #  max_features=2000,
            stop_words="english",
            strip_accents="unicode",
            lowercase=True,
            tokenizer=LemmaTokenizer(),
        )
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

        try:
            # create the dataframes
            self.train_df = pd.DataFrame.sparse.from_spmatrix(self.transformed_train)
            self.test_df = pd.DataFrame.sparse.from_spmatrix(self.transformed_test)
            self.train_df.columns = train_cols
            self.test_df.columns = train_cols

            # also add the labels associated with each text
            self.train_df["Class"] = self.train["target"]
            self.test_df["Class"] = self.test["target"]
        except MemoryError:
            print("Not enought memory to create pandas dataframe")


p = Preprocessing(Dataset.imdb_reviews)
m = mnb()
m.fit(p.transformed_train, p.train_label)
print(p.transformed_train.shape[1])
print(p.transformed_test.shape[1])
c = m.predict(p.transformed_test)
print(Classifier.evaluate_acc(p.test_label, c))

#  sets = cross_validation_split(p.transformed_train, p.train_label, 5)
#  print(kfoldCV(MultinomialNB, sets, alpha=0.05))
#
#  clf = MultinomialNB(alpha=0.05)
#  clf.fit(p.transformed_train, p.train_label)
#  pred = clf.predict(p.transformed_test)
#  print(Classifier.evaluate_acc(p.test_label, pred))
