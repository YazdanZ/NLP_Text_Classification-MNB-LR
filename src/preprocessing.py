from enum import Enum


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
        if dataset == Dataset.imdb_reviews:
            pass
        elif dataset == Dataset.twenty_news:
            pass
        else:
            raise IndexError("Dataset enumeration between 1 and 2")

    def clean(self):
        print("Cleaning dataset ...")