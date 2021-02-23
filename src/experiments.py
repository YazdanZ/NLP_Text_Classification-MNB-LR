from preprocessing import Preprocessing, Dataset
from multinomial_nb import MultinomialNB
from cross_validation import cross_validation_split, kfoldCV

# ---------------------------------------------------------------------------
# 1- Conduct multiclass classification on the 20 news group dataset,
# and binary classification on the IMDB Reviews.

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
