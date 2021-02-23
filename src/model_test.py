from preprocessing import Preprocessing, Dataset
from multinomial_nb import MultinomialNB
from cross_validation import cross_validation_split, kfoldCV
from sklearn.metrics import accuracy_score
from logistic_regression import LogisticRegression
from logistic_regression import LogisticRegression

p = Preprocessing(Dataset.imdb_reviews)
X_train, y_train, X_test, y_test = p.get_train_test()
#  accuracies = MultinomialNB.tune_hyperparams(X_train, y_train, 2)
#  print(accuracies)
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
print(accuracy_score(pred, y_test))

#  clf = MultinomialNB(alpha=0.05)
#  X_train, y_train, X_test, y_test = p.get_train_test()
#  clf.fit(X_train, y_train)
#  clf.predict(X_test)
#  print(clf.evaluate_acc(y_test))
