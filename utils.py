import pickle

def read_pickle(f):
    with open(f, 'rb') as input_file:
        obj = pickle.load(input_file)
    return obj


def save_pickle(obj, f):
    with open(f, 'wb') as output_file:
        pickle.dump(obj, output_file)