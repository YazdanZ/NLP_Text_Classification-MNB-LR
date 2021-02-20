from enum import Enum
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import dic_maker


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
            self.preprocess_imbd()
        elif dataset == Dataset.twenty_news:
            self.preprocess_20news()
        else:
            raise IndexError("Dataset enumeration between 1 and 2")
        
        self.vectorize_transform()
    
    def preprocess_20news(self):
        print("Preprocessing the 20 newsgroup dataset...\n")

        self.train = twenty_train_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=0, remove=(["headers", "footers", "quotes"]))
        self.test = twenty_train_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=0, remove=(["headers", "footers", "quotes"]))
        print("Training set contains {n} instances".format(n=len(self.train.data)))
        print("Testing set contains {n} instances".format(n=len(self.test.data)))
    
    def preprocess_imbd(self):
        print("Preprocessing the IMDB reviews dataset...\n")

        #TODO: assign test and train subsets to the instance variable 

    def vectorize_transform(self):
        """Turn the text data into numerical feature using the bags of words representation"""

        # count vectorization
        # use the built-in english stopwords  
        print("Vectorizing dataset...")
        vectorizer_train = CountVectorizer(stop_words="english")
        vectorizer_test = CountVectorizer(stop_words="english")
        vectorized_train = vectorizer_train.fit_transform(self.train["data"])
        vectorized_test = vectorizer_test.fit_transform(self.test["data"])
        train_cols = vectorizer_train.get_feature_names()
        test_cols = vectorizer_test.get_feature_names()

        # term frequency–inverse document frequency tranformation
        print("Transforming vector using term frequency–inverse document frequency transformation...")
        tfidf_transformer_train = TfidfTransformer() 
        tfidf_transformer_test = TfidfTransformer() 
        self.transformed_train = tfidf_transformer_train.fit_transform(vectorized_train)
        self.transformed_test = tfidf_transformer_test.fit_transform(vectorized_test)

        # create the dataframes        
        self.train_df = pd.DataFrame(self.transformed_train.toarray())
        self.test_df = pd.DataFrame(self.transformed_test.toarray())
        self.train_df.columns = train_cols
        self.test_df.columns = test_cols
        
        # also add the labels associated with each text
        self.train_df["Class"] = self.train["target"]
        self.test_df["Class"] = self.test["target"]

p = Preprocessing(Dataset.twenty_news)
print(p.train_df)

# create a dictionary for the second data set with below format:
# {'Data': list of documents strings, 'target_names': ["positive", "negative"], 'targets' = [ 0 , ..., 0, 1, 1, ..., 1]}
# dict_maker = dic_maker.dic_maker()
# imdb_dic = dict_maker.list_maker()

y = p.train_df.pop("Class").to_numpy()
X = p.train_df.to_numpy()
y_test = p.test_df.pop("Class").to_numpy()
X_test = p.test_df.to_numpy()
clf = MultinomialNB()
clf.fit(X, y)
pred = clf.predict(X_test)
print(accuracy_score(y, pred))
# for doc, category in zip(docs_new, predicted):
#     print("%r => %s" % (doc, imdb_dic["target_names"][category]))
