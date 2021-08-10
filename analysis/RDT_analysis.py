import numpy as np

import datasets
from sklearn.model_selection import train_test_split

from tree_ensemble import RDT


def run_analysis(data_ident, test_size, empty_leafs_list, iterations, leaf_param_tuples, number_trees, dt_modes,
                 agg_modes):
    """ Runs the analysis comparing different prediction methods depending on leaf size (mostly determined by
    leaf_parameter_tuples) for the respective dataset."""
    x, y = datasets.load(data_ident, verbose=False)

    nr_methods = 0
    for modes in agg_modes.values():
        nr_methods += len(modes)
    results = np.zeros((iterations, len(leaf_param_tuples), len(empty_leafs_list), nr_methods, 2))

    for i in range(iterations):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=i)

        for count_param, (min_leaf, min_split) in enumerate(leaf_param_tuples):
            rdt_ensemble = RDT(number_trees=number_trees, min_samples_leaf=min_leaf,
                               min_samples_split=min_split, random_state=i*10)
            # set some features for datasats for which the automatic identification does not make as much sense
            if data_ident in datasets.DATASET_TYPE_FEATURES:
                rdt_ensemble.fit(x_train, y_train, data_feature_types=datasets.DATASET_TYPE_FEATURES[data_ident])
            else:
                rdt_ensemble.fit(x_train, y_train)
            for count_leafs, empty_leafs in enumerate(empty_leafs_list):
                method_results = rdt_ensemble.method_analysis(x_test, dt_modes, agg_modes, empty_leafs, y_test)
                results[i, count_param, count_leafs] = method_results
    np.save("results/" + data_ident + ".npy", results)
    # the following code creates a .info file which can be used to identify what the parameters for the .npy file were
    with open("results/" + data_ident + ".info", "w") as run_info:
        run_info.write(data_ident + "\n")
        run_info.write(str(iterations) + "\n")
        lpt_string = ""
        for param_pair in leaf_param_tuples:
            lpt_string += str(param_pair[0]) + "_" + str(param_pair[1]) + "---"
        lpt_string = lpt_string[:-3]
        run_info.write(lpt_string + "\n")
        run_info.write("---".join(str(e) for e in empty_leafs_list) + "\n")
        run_info.write("---".join(dt_modes))  # new line is in the loop, so that there is no new line at file end
        for mode in dt_modes:
            run_info.write("\n" + mode + ":" + "---".join(agg_modes[mode]))
