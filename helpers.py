import pandas as pd


def load_data(type):
    file_path = 'data/' + type + '.csv'
    data = pd.read_csv(file_path, header = 0, encoding='ISO-8859-1')
    print(data.head())
    return data

load_data('train')