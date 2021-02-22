from preprocessing import Preprocessing, Dataset
from multinomial_nb import MultinomialNB
from cross_validation import cross_validation_split, kfoldCV

p = Preprocessing(Dataset.twenty_news)
X_train, y_train, X_test, y_test = p.get_train_test()
accuracies = MultinomialNB.tune_hyperparams(X_train, y_train, 2)
print(accuracies)

#  clf = MultinomialNB(alpha=0.05)
#  X_train, y_train, X_test, y_test = p.get_train_test()
#  clf.fit(X_train, y_train)
#  clf.predict(X_test)
#  print(clf.evaluate_acc(y_test))
