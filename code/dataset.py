import os
from utils import tsv_to_dict_multiple, tsv_to_dict_unqiue

class DataProcessor:
    def __init__(self, data_root_dir):
        self.data_root_dir = data_root_dir
        self.passages = tsv_to_dict_unqiue(os.path.join(data_root_dir, "collection.tsv"))
        self.queries_train = tsv_to_dict_unqiue(os.path.join(data_root_dir, "queries.train.tsv"))
        self.queries_dev = tsv_to_dict_unqiue(os.path.join(data_root_dir, "queries.dev.small.tsv"))
        self.qrels_train = tsv_to_dict_multiple(os.path.join(data_root_dir, "qrels.train.tsv"), keys = [0, 2])
        self.qrels_dev = tsv_to_dict_multiple(os.path.join(data_root_dir, "qrels.dev.small.tsv"), keys = [0, 2])

    def get_data(self):
        return {
            "passages": self.passages,
            "queries_train": self.queries_train,
            "queries_dev": self.queries_dev,
            "qrels_train": self.qrels_train,
            "qrels_dev": self.qrels_dev
        }

    def print_samples(self, n = 5):
        print("Passages:")
        for i, (key, value) in enumerate(dict(list(self.passages.items())[:n])):
            print(f"{key}: {value}")

        print("Train Queries:")
        for i, (key, value) in enumerate(dict(list(self.queries_train.items())[:n])):
            print(f"{key}: {value}")

        print("Dev Queries:")
        for i, (key, value) in enumerate(dict(list(self.queries_dev.items())[:n])):
            print(f"{key}: {value}")

        print("Train Qrels:")
        for i, (key, value) in enumerate(dict(list(self.qrels_train.items())[:n])):
            print(f"{key}: {value}")

        print("Dev Qrels:")
        for i, (key, value) in enumerate(dict(list(self.qrels_dev.items())[:n])):
            print(f"{key}: {value}")