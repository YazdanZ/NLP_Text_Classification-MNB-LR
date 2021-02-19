from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import dic_maker

# categories = ['alt.atheism', 'soc.religion.christian',
#               'comp.graphics', 'sci.med']
# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42,
#                                   remove=(["headers", "footers", "quotes"]))

# create a dictionary for the second data set with below format:
# {'Data': list of documents strings, 'target_names': ["positive", "negative"], 'targets' = [ 0 , ..., 0, 1, 1, ..., 1]}
dict_maker = dic_maker.dic_maker()
imdb_dic = dict_maker.list_maker()

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(imdb_dic['data'])

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, imdb_dic['target'])

docs_new = ['I did not like it', 'I loved it']

X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, imdb_dic['target_names'][category]))