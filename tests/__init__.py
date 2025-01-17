import os


def load_all_test_files(test_data_dir):
    data_files = []
    json_files = []

    directory = os.fsencode(test_data_dir)

    lst_dir = sorted(os.listdir(directory))
    for file in lst_dir:
        filename = os.fsdecode(file)
        filepath = os.path.join(test_data_dir, filename)
        if filename.endswith(".json"):
            json_files.append(filepath)
        else:
            data_files.append(filepath)

    return list(zip(data_files, json_files))
