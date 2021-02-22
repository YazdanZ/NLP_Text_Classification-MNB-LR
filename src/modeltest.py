from preprocessing import Preprocessing, Dataset
from multinomial_nb import MultinomialNB

p = Preprocessing(Dataset.imdb_reviews)
clf = MultinomialNB(alpha=0.05)
X_train, y_train, X_test, y_test = p.get_train_test()
clf.fit(X_train, y_train)
clf.predict(X_test)
print(clf.evaluate_acc(y_test))
