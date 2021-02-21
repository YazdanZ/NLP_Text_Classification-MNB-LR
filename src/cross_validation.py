import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import vstack
from sklearn.naive_bayes import MultinomialNB
from classifier import Classifier


def cross_validation_split(train_csr, train_labels, k):
    """Takes training data as input and splits it into k folds.
    Each fold is then used once as validation while the k − 1
    remaining folds form the training set"""
    # get a random sample of items

    train_csr, train_labels = shuffle(train_csr, train_labels)

    num_rows = train_csr.shape[0]
    num_folds_row = num_rows // k

    sets = []
    for f in range(k):
        start = f * num_folds_row
        stop = start + num_folds_row

        validation_x = train_csr[start:stop]
        validation_y = train_labels[start:stop]

        train1_x, train2_x = train_csr[0:start], train_csr[stop:]
        train1_y, train2_y = train_labels[0:start], train_labels[stop:]
        train_x = vstack([train1_x, train2_x])
        train_y = np.concatenate((train1_y, train2_y))

        sets.append((train_x, train_y, validation_x, validation_y))

    return sets


def kfoldCV(model, sets, **kwargs):
    """Takes train/validation sets generated above and a
    given model as input. The function iterates through each
    train/validation set and returns the average result of each fold."""
    classifier = model(**kwargs)
    accuracy = []
    for k_set in sets:
        classifier.fit(k_set[0], k_set[1])
        predictions = classifier.predict(k_set[2])
        accuracy.append(Classifier.evaluate_acc(predictions, k_set[3]))
    return np.array(accuracy).mean()
