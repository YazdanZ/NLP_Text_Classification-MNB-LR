import glob
import os


class dic_maker():

    def __init__(self):
        self.corpus = []
        self.str_path = "C:\\ETS\\COMP551- Reihane Rabbani\\Assignment2\\aclImdb_v1\\aclImdb\\train\\"
        self.list_types = [self.str_path + "neg", self.str_path + "pos"]

    def list_maker(self):
        for types in self.list_types:
            file_list = glob.glob(
                os.path.join(os.getcwd(), types,
                             "*.txt"))

            try:
                for file_path in file_list:
                    with open(file_path, encoding='utf-8') as f_input:
                        self.corpus.append(f_input.read())
            except:
                print("hey")

        dic_output = {
            'data': self.corpus,
            'target_names': ["negative", "positive"],
            'target': int(len(self.corpus) / 2) * [0] + int(len(self.corpus) / 2) * [1]
        }
        return dic_output
