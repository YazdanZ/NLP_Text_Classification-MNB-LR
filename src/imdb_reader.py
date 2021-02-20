import glob
import os
import json


def get_subset(subset):
    if not os.path.exists("../data/imdb_" + subset + ".json"):

        corpus = []
        str_path = "../data/aclImdb/"
        list_types = [str_path + subset + "/neg", str_path + subset + "/pos"]
        for types in list_types:
            file_list = glob.glob(os.path.join(os.getcwd(), types, "*.txt"))

            try:
                for file_path in file_list:
                    with open(file_path, encoding="utf-8") as f_input:
                        corpus.append(f_input.read())
            except:
                print("Error while listing documents")

        dic_output = {
            "data": corpus,
            "target_names": ["negative", "positive"],
            # first half of the lift is negative while the second half is positive
            "target": int(len(corpus) / 2) * [0] + int(len(corpus) / 2) * [1],
        }
        with open("../data/imdb_" + subset + ".json", "w") as json_file:
            json.dump(dic_output, json_file)

    else:
        with open("../data/imdb_" + subset + ".json") as json_file:
            dic_output = json.load(json_file)
    return dic_output
