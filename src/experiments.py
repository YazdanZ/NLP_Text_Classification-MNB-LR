import numpy as np
import pandas as pd
import operator
from preprocessing import Preprocessing, Dataset
from logistic_regression import LogisticRegression
from multinomial_nb import MultinomialNB
from cross_validation import cross_validation_split, kfoldCV
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


SEPARATOR = "\n" + 40 * "-" + "\n" + 40 * "-" + "\n"

# Preprocess textual data
imdb_preprocessor = Preprocessing(Dataset.imdb_reviews, verbose=True)
print(SEPARATOR)
twenty_news_preprocessor = Preprocessing(Dataset.twenty_news, verbose=True)

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


# hyperparameter tuning for naive bayes
# test for alphas with 0.5 in [0.5, 5[
alphas = np.arange(0.1, 2, 0.5)

# 20news dataset
#  scores_20news = MultinomialNB.tune_hyperparams(
#  X_train_20news, y_train_20news, 5, alphas
#  )
#  index, max_acc = max(enumerate(scores_20news), key=operator.itemgetter(1))
#  best_alpha_20news = alphas[index]
#  print(best_alpha_20news)

# IMDB dataset
#  scores_imdb = MultinomialNB.tune_hyperparams(X_train_imdb, y_train_imdb, 5, alphas)
#  index, max_acc = max(enumerate(scores_imdb), key=operator.itemgetter(1))
#  best_alpha_imdb = alphas[index]
#  print(best_alpha_imdb)

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


# ---------------------------------------------------------------------------
# 1- Conduct multiclass classification on the 20 news group dataset,
# and binary classification on the IMDB Reviews.
best_alpha_imdb = best_alpha_20news = 0.04


def evaluate_model(model, dataset, sets=None, verbose=True, show_plot=True, **kwargs):
    clf = model(**kwargs)
    if dataset == Dataset.twenty_news:
        X_train, y_train, X_test, y_test = (
            X_train_20news,
            y_train_20news,
            X_test_20news,
            y_test_20news,
        )
        rg = 20
    elif dataset == Dataset.imdb_reviews:
        X_train, y_train, X_test, y_test = (
            X_train_imdb,
            y_train_imdb,
            X_test_imdb,
            y_test_imdb,
        )
        rg = 2
    else:
        raise ValueError("Dataset doesn't exist")
    if sets is not None:
        X_train, y_train, X_test, y_test = sets
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(pred, y_test, average="weighted")
    if verbose:
        msg1 = "Accuracy of the {} model on the {} dataset: {}".format(
            model, dataset, acc
        )
        msg2 = "F1 score of the {} model on the {} dataset: {}".format(
            model, dataset, f1
        )
        print(msg1)
        print(msg2)
    if show_plot:
        con_matrix = confusion_matrix(y_test, pred)
        df_cm = pd.DataFrame(con_matrix, range(rg), range(rg))
        sns.heatmap(df_cm, annot=True)
        plt.ylabel("Predicted Class")
        plt.xlabel("True Class")
        plt.show()  # Fig10
    return acc


def part1():
    evaluate_model(MultinomialNB, Dataset.imdb_reviews, alpha=0.05)
    evaluate_model(MultinomialNB, Dataset.twenty_news, alpha=0.05)
    evaluate_model(LR, Dataset.imdb_reviews)
    evaluate_model(LR, Dataset.twenty_news)
    evaluate_model(
        LogisticRegression, Dataset.imdb_reviews, learning_rate=0.01, num_iters=10000
    )


# ---------------------------------------------------------------------------
# 2- With a plot, compare the accuracy of the two models as a
# function of the size of dataset (by controlling the training size).
def part2():
    def dataset_size_acc(dataset, model, **kwargs):
        fractions = [0.2, 0.4, 0.6, 0.8, 1]
        accuracies = []

        for frac in fractions:
            preprocessor = Preprocessing(dataset, frac=frac)
            set_ = preprocessor.get_train_test()
            acc = evaluate_model(
                model, dataset, sets=set_, verbose=False, show_plot=False, **kwargs
            )
            accuracies.append(acc)
        df = pd.DataFrame(columns=["Training set fraction", "Accuracy"])
        df["Training set fraction"] = fractions
        df["Accuracy"] = accuracies
        df.plot.line(x="Training set fraction", y="Accuracy")
        plt.title("training set fraction size vs accuracy {}-{}".format(dataset, model))
        plt.show()

    dataset_size_acc(Dataset.imdb_reviews, MultinomialNB, alpha=0.04)
    dataset_size_acc(Dataset.twenty_news, MultinomialNB, alpha=0.04)
    dataset_size_acc(Dataset.imdb_reviews, LR)
    dataset_size_acc(Dataset.twenty_news, LR)


# ---------------------------------------------------------------------------
# 3- Compare the accuracy of models with reduced features
