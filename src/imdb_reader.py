import glob
import os


def get_subset(subset):
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
    return dic_output
