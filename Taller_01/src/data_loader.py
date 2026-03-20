
import requests
import numpy as np

url = "https://zenodo.org/record/4269852/files/dermamnist.npz?download=1"
file_name = "./data/data.npz"
TAGS = {
    'train' : 'train'
    ,'test' :'test'
    ,'validation' : 'val'
}


def fetch_data():
    response = requests.get(url)
    with open(file_name, "wb") as file_ins:
        file_ins.write(response.content)


def load_data(fetch = True):
    if fetch:
        fetch_data()
    data = np.load(file_name)
    X = {label : data[f'{TAGS[label]}_images'] for label in TAGS}
    Y = {label : data[f'{TAGS[label]}_labels'] for label in TAGS}
    return X, Y



if __name__ == "__main__":
    fetch_data()
    



