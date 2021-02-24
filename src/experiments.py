import numpy as np
import pandas as pd
import operator
from preprocessing import Preprocessing, Dataset
from multinomial_nb import MultinomialNB
from cross_validation import cross_validation_split, kfoldCV
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression

SEPARATOR = "\n" + 40 * "-" + "\n" + 40 * "-" + "\n"

# Preprocess textual data
imdb_preprocessor = Preprocessing(Dataset.imdb_reviews)
print(SEPARATOR)
twenty_news_preprocessor = Preprocessing(Dataset.twenty_news)

# Get lemmatized, cleaned and vectorized train and test sets as sparse arrays
(
    X_train_imdb,
    y_train_imdb,
    X_test_imdb,
    y_test_imdb,
) = imdb_preprocessor.get_train_test()
(
    X_train_20news,
    y_train_20news,
    X_test_20news,
    y_test_20news,
) = twenty_news_preprocessor.get_train_test()


print(SEPARATOR)
#  # hyperparameter tuning for naive bayes
#  # test for alphas with 0.5 in [0.5, 5[
#  alphas = np.arange(0.5, 3, 0.5)
#
#  # 20news dataset
#  scores_20news = MultinomialNB.tune_hyperparams(
#      X_train_20news, y_train_20news, 5, alphas
#  )
#  index, max_acc = max(enumerate(scores_20news), key=operator.itemgetter(1))
#  best_alpha_20news = alphas[index]
#
#  # IMDB dataset
#  scores_imdb = MultinomialNB.tune_hyperparams(X_train_imdb, y_train_imdb, 5, alphas)
#  index, max_acc = max(enumerate(scores_imdb), key=operator.itemgetter(1))
#  best_alpha_imdb = alphas[index]

# hyperparameter tuning for Logistic Regression
# 20news dataset
#  rs = RandomizedSearchCV(
#  LogisticRegression(solver="lbfgs"),
#  {
#  "max_iter": np.arange(100, 500, 20),
#  "C": [0.001, 0.01, 0.1, 1, 10, 100],
#  },
#  return_train_score=False,
#  n_iter=10,
#  cv=5,
#  )
#  rs.fit(X_train_20news, y_train_20news)
#  resuts = pd.DataFrame(rs.cv_results_)
#  print(resuts)
#  resuts.to_csv("out.csv", index=False)


# ---------------------------------------------------------------------------
# 1- Conduct multiclass classification on the 20 news group dataset,
# and binary classification on the IMDB Reviews.
#  mnb = MultinomialNB(alpha=0.05)
#  mnb.fit(X_train_20news, y_train_20news)
#  mnb.predict(X_test_20news)
#  mnb.evaluate_acc(y_test_20news)

mnb = MultinomialNB(alpha=0.05)
mnb.fit(X_train_imdb, y_train_imdb)
mnb.predict(X_test_imdb)
print(mnb.evaluate_acc(y_test_imdb))
# ---------------------------------------------------------------------------
# 2- In a single table, compare and report the performance of the
# performance of naive Bayes and logistic regression on each of the two
# datasets (with their best hyperparameters), and highlight the winner for
# each dataset and overall.

# ---------------------------------------------------------------------------
# 3- With a plot, compare the accuracy of the two models as a
# function of the size of dataset (by controlling the training size).

# ---------------------------------------------------------------------------
# For this part, you might implement logistic regression from scratch
# (and try different learning rates,investigate different stopping criteria
# for the gradient descent), try linear regression for predicting ratings in
# the IMDB data, try different text embedding methods as alternatives to bag
# of words. You are also welcome and encouraged to try any other model covered
# in the class, and you are free to implement them yourself or use any Python
# library that has their implementation, e.g. the from the scikit-learn
# package. Of course, you do not need to do all of these things, but look
# at them as suggestions and try to demonstrate curiosity, creativity, rigour,
# and an understanding of the course material in how you run your chosen
# experiments and how you report on them in your write-up.
