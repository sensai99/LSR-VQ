import csv

# Read the tsv file as a dictionary (each key has a single value)
def tsv_to_dict_unqiue(file_path, keys = [0, 1]):
    with open(file_path, mode = "r", encoding = "utf-8") as file:
        reader = csv.reader(file, delimiter= "\t")

        data = {}
        for row in reader:
            if row[keys[0]] in data:
                data[row[keys[0]]].append(row[keys[1]])
            else:
                data[row[keys[0]]] = [row[keys[1]]]

    return data


# Read the tsv file as a dictionary (each key has a multiple value)
def tsv_to_dict_multiple(file_path, keys = [0, 2]):
    with open(file_path, mode = "r", encoding = "utf-8") as file:
        reader = csv.reader(file, delimiter= "\t")

        data = {}
        for row in reader:
            if row[keys[0]] in data:
                data[row[keys[0]]].append(row[keys[1]])
            else:
                data[row[keys[0]]] = [row[keys[1]]]

    return data
