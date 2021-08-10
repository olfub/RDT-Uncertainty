import math
import numpy as np
import arff

from sklearn.model_selection import train_test_split

import sklearn.datasets as sk_data

from random_tree import is_number


ALL = ["breast_cancer", "magic_telescope", "arff_diabetes", "arff_sonar", "arff_ionosphere", "arff_spambase",
       "transfusion", "pop_failures", "biodeg", "banknote_authentication", "scene", "Skin_NonSkin", "webdata",
       "MiniBooNE_PID", "vehicle_sens", "arff_tic-tac-toe", "arff_TrainingDataset", "arff_vote", "arff_elecNormNew",
       "arff_airlines", "mushroom"]

DATASET_TYPE_FEATURES = {"arff_elecNormNew": [0, 1, 0, 0, 0, 0, 0, 0],
                         "arff_airlines": [1, 0, 1, 1, 1, 0, 0]}


def load(identifier, verbose=True):
    """
    Load a dataset and return the feature array and the respective classes.
    :param identifier: identifier string for the dataset
    :param verbose: if True, print additional information to the console
    :return: tuple (x, y)
        WHERE
        x is the feature array of the dataset
        y is a numpy array of the labels of the dataset
    """
    # process easy-to-read identifier
    if identifier == "scene":
        identifier = "openml_312"
    elif identifier == "webdata":
        identifier = "openml_350"
    elif identifier == "vehicle_sens":
        identifier = "openml_357"

    # scikit datasets
    if identifier == "boston":
        x, y = sk_data.load_boston(return_X_y=True)
    elif identifier == "iris":
        x, y = sk_data.load_iris(return_X_y=True)
    elif identifier == "diabetes":
        x, y = sk_data.load_diabetes(return_X_y=True)
    elif identifier == "digits":
        x, y = sk_data.load_digits(return_X_y=True)
    elif identifier == "linnerud":
        x, y = sk_data.load_linnerud(return_X_y=True)
    elif identifier == "wine":
        x, y = sk_data.load_wine(return_X_y=True)
    elif identifier == "breast_cancer":
        x, y = sk_data.load_breast_cancer(return_X_y=True)

    # datasets from files
    elif identifier == "transfusion":
        dataset = np.loadtxt("data_sets/UCI/transfusion.data", dtype=float, delimiter=",", skiprows=1)
        x = dataset[:, :-1]
        y = dataset[:, -1]
    elif identifier == "Skin_NonSkin":
        dataset = np.loadtxt("data_sets/UCI/Skin_NonSkin.txt", dtype=float, delimiter="\t")
        x = dataset[:, :-1]
        y = dataset[:, -1]
    elif identifier == "pop_failures":
        with open("data_sets/UCI/pop_failures.dat", "r") as file:
            data = file.read()
        data_list = (data.split("\n")[1:])
        data_elements = [[float(value) for value in line.split()] for line in data_list]
        dataset = np.array(data_elements)
        x = dataset[:, :-1]
        x = x[:, 2:]
        y = dataset[:, -1]
    elif identifier == "MiniBooNE_PID":
        with open("data_sets/UCI/MiniBooNE_PID.txt", "r") as file:
            data = file.read()
        data_list = (data.split("\n")[1:])
        # last row is empty
        data_list = data_list[:-1]
        data_elements = [[float(value) for value in line.split()] for line in data_list]
        temp_dataset = np.array(data_elements)
        dataset = np.c_[temp_dataset, np.ones(len(data_elements))]
        for i in range(len(temp_dataset)):
            if i < 36499:
                dataset[i, -1] = 0
            else:
                dataset[i, -1] = 1
        x = dataset[:, :-1]
        y = dataset[:, -1]
    elif identifier == "biodeg":
        dataset = np.loadtxt("data_sets/UCI/biodeg.csv",
                             converters={41: lambda s: 0 if bytes.decode(s) == "NRB" else 1}, delimiter=";")
        x = dataset[:, :-1]
        y = dataset[:, -1]
    elif identifier == "banknote_authentication":
        dataset = np.loadtxt("data_sets/UCI/data_banknote_authentication.txt", dtype=float, delimiter=",")
        x = dataset[:, :-1]
        y = dataset[:, -1]
    elif identifier == "magic_telescope":
        dataset = np.loadtxt("data_sets/UCI/magic04.data",
                             converters={10: lambda s: 0 if bytes.decode(s) == "g" else 1}, delimiter=",")
        x = dataset[:, :-1]
        y = dataset[:, -1]
    elif identifier == "mushroom":
        mushroom = np.genfromtxt("data_sets/UCI/agaricus-lepiota.data", delimiter=",", dtype=str)
        x = mushroom[:, 1:]
        y = mushroom[:, 0]

    # arff datasets
    elif identifier.startswith("arff_"):
        if verbose:
            print("Note: arff datasets are only supported if there are no missing values in numeric feature columns")
        arff_name = identifier[5:]
        try:
            with open("data_sets/arff/" + arff_name + ".arff", "r") as file:
                dataset = arff.load(file)
        except OSError:
            raise FileNotFoundError(f"Arff file with the name {arff_name} does not exist")
        array_set = np.array(dataset["data"])
        x = array_set[:, :-1]
        if "?" in x:
            raise ValueError(f"There should not be a \"?\" sign in the loaded(!) arff dataset {arff_name}")
        x[x == None] = "?"  # do NOT replace == with "is"
        y = array_set[:, -1]

    # openml data
    elif identifier.startswith("openml_"):
        openml_id = int(identifier[7:])
        try:
            dataset = sk_data.fetch_openml(data_id=openml_id, as_frame=False)
        except ValueError:
            raise ValueError(f"No openml dataset found with the id {openml_id}")
        x = dataset.data
        y = dataset.target

    else:
        raise ValueError(f"Dataset with name {identifier} does not exist")
    return x, y


def load_train_test(identifier, test_size, random_state, verbose=True, test_min=0):
    """
    Load a dataset and return the feature array and labels of the train and test data, respectively.
    :param identifier: identifier string for the dataset
    :param test_size: size of the test set (percentage if < 1, else number of test instances)
    :param random_state: random state, can be used for deterministic behavior
    :param verbose: if True, print additional information to the console
    :param test_min: minimal number of instances which will be in the test set
    :return: tuple (x_train, y_train, x_test, y_test)
        WHERE
        x_train is the feature array of the training data
        y_train is a numpy array of the labels of the train data
        x_test is the feature array of the test data
        y_test is a numpy array of the labels of the test data
    """
    x, y = load(identifier, verbose=verbose)
    nr_instances = x.shape[0]
    test_size = math.ceil(nr_instances * test_size)
    if test_size >= test_min:
        # everything is good, continue as usual
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_min, random_state=random_state)
    return x_train, y_train, x_test, y_test


def load_train_val_test(identifier, val_size, test_size, random_state, verbose=True, test_min=0):
    """
    Load a dataset and return the feature array and labels of the train, validation, and test data, respectively.
    :param identifier: identifier string for the dataset
    :param val_size: size of the val set AFTER the test set is removed (please use a percentage in (0, 1))
    :param test_size: size of the test set (please use a percentage in (0, 1))
    :param random_state: random state, can be used for deterministic behavior
    :param verbose: if True, print additional information to the console
    :param test_min: minimal number of instances which will be in the test set
    :return: tuple (x_train, y_train, x_test, y_test)
        WHERE
        x_train is the feature array of the training data
        y_train is a numpy array of the labels of the train data
        x_val is the feature array of the validation data
        y_val is a numpy array of the labels of the validation data
        x_test is the feature array of the test data
        y_test is a numpy array of the labels of the test data
    """
    x, y = load(identifier, verbose=verbose)
    nr_instances = x.shape[0]
    test_size = math.ceil(nr_instances*test_size)
    if test_size >= test_min:
        # everything is good, continue as usual
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_size,
                                                                    random_state=random_state)
    else:
        # use test_min instead of the percentage
        x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_min, random_state=random_state)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_size,
                                                      random_state=random_state)
    return x_train, y_train, x_val, y_val, x_test, y_test


def load_data_for_tests(identifier):
    """
    Loads data sets which are manually defined to be used as tests for special cases (or regular cases), to test that
    the code works as intended.
    :param identifier: identifier string for the dataset
    :return: tuple (x_train, y_train, x_test, y_test)
        WHERE
        x_train is the feature array of the training data
        y_train is a numpy array of the labels of the train data
        x_test is the feature array of the test data
        y_test is a numpy array of the labels of the test data
    """
    if identifier == "nominal_splits":
        # contains nominal values in test set, which did not occur in the train set, also tests empty_leafs behavior
        x_train = np.array([["a", "f1"], ["a", "f2"], ["a", "f3"],
                            ["b", "f1"], ["b", "f2"], ["b", "f3"],
                            ["c", "f1"], ["c", "f2"], ["c", "f3"]])
        y_train = np.array([0, 0, 1, 0, 0, 0, 1, 1, 1])
        x_test = np.array([["b", "f1"], ["c", "f3"], ["a", "f4"], ["a", "f5"], ["d", "f3"]])
        y_test = np.array([0, 1, 0, 1])

    else:
        raise ValueError(f"Testing dataset with name {identifier} does not exist")

    return x_train, y_train, x_test, y_test


def dataset_properties(identifier):
    """
    Print several properties of the specified dataset.
    :param identifier: identifier string for the dataset
    :return: None
    """
    x, y = load(identifier, verbose=False)
    print(f"Dataset: {identifier}")
    print(f"Number instances: {x.shape[0]}")
    print(f"Feature size: {x.shape[1]}")
    _, counts = np.unique(y, return_counts=True)
    if counts[0] + counts[1] != x.shape[0]:
        raise RuntimeError("Unexpected error regarding the dataset size: Sum of appearance of both classes does not fit"
                           " overall number of instances.")
    ratio = counts[1] / x.shape[0]
    print(f"Dataset balance (rounded, 2 digits): {round(ratio, 2)}")
    has_nominal = False
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if not(is_number(x[i, j])):
                has_nominal = True
                break
        if has_nominal:  # no need to search any further
            break
    next_string = "yes" if has_nominal else "no"
    print(f"Has nominal features: {next_string}")
