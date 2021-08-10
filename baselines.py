import sys
import numpy as np
import scipy

from sklearn import tree, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

import datasets
from run_analysis import LPT
from random_tree import check_data, set_data_feature_information


def main(model):
    run(model)


def run(model):
    np.random.seed(0)
    lpt = LPT
    test_size = 0.5
    for data_ident in datasets.ALL:
        scores = np.zeros((len(lpt), 5, 2))
        x, y = datasets.load(data_ident, verbose=False)
        if type(x) == scipy.sparse.csr.csr_matrix:
            x = x.toarray()

        data_feature_types, _ = set_data_feature_information(x)
        if sum(data_feature_types) > 0:
            indices1 = [count for count, data_type in enumerate(data_feature_types) if data_type == 1]
            indices2 = [count for count, data_type in enumerate(data_feature_types) if data_type == 0]

            x_part1 = x[:, indices1]
            x_part2 = x[:, indices2]

            enc = OneHotEncoder(handle_unknown="error")
            enc.fit(x_part1)

            x_part1 = enc.transform(x_part1).toarray()

            if data_ident == "arff_airlines":
                x = scipy.sparse.csr_matrix(np.concatenate((x_part1, x_part2.astype("float64")), axis=1))
            else:
                x = np.concatenate((x_part1, x_part2), axis=1)

        for lpt_index, config in enumerate(lpt):
            min_samples_leaf = config[0]
            min_samples_split = config[1]
            for i in range(5):  # iterations
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=i)
                checked_x_train, checked_y_train, _, class_converter = check_data(x_train, y_train, suppress_info=True)
                checked_x_val, checked_y_val, _, _ = check_data(x_test, y_test, suppress_info=True,
                                                                class_converter=class_converter)
                if model == "RF":
                    clf = ensemble.RandomForestClassifier(n_estimators=100, min_samples_split=min_samples_split,
                                                          min_samples_leaf=min_samples_leaf, random_state=i)
                elif model == "DT":
                    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf, random_state=i)
                else:
                    raise ValueError("ERROR")
                clf = clf.fit(checked_x_train, checked_y_train)
                y_pred_class = clf.predict(checked_x_val)
                acc = accuracy_score(checked_y_val, y_pred_class)
                y_pred_score = clf.predict_proba(checked_x_val)
                auc = roc_auc_score(checked_y_val, y_pred_score[:, 1])
                scores[lpt_index, i, 0] = acc
                scores[lpt_index, i, 1] = auc
        if model == "RF":
            np.save(f"results/{data_ident}_rf_baseline.npy", scores)
        elif model == "DT":
            np.save(f"results/{data_ident}_dt_baseline.npy", scores)
        print(f"{data_ident} finished")


if __name__ == "__main__":
    num_arguments = len(sys.argv) - 1
    if num_arguments == 1:
        if sys.argv[1] in ["DT", "RF"]:
            main(sys.argv[1])
        else:
            raise Exception("Model needs to be \"DT\" or \"RF\"")
    else:
        raise Exception("Need to specify a model: \"DT\" or \"RF\"")
