import numpy as np

from random_tree import Tree, check_data
from prediction.prediction_utility import check_methods, OutputType, prepare_empty_leafs_2, finish_empty_leafs_2
import prediction.aggregation as agg
import prediction.evidence_helper as ev_help

from prediction import prediction_methods

from sklearn.metrics import accuracy_score, roc_curve, auc


class RDT:
    """
    A class representing RDT (ensemble of randomly created decision trees).
    """

    def __init__(self, number_trees=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None):
        """
        Create a RDT classifier. A random seed can be specified to create a random state instance which allows for
        deterministic behavior.
        :param max_depth: maximum allowed depth of the tree
        :param min_samples_split: minimum number of instances to try to split
        :param min_samples_leaf: minimum number of instances in a leaf node
        :param random_state: random state, can be used for deterministic behavior
        """
        self.number_trees = number_trees
        self.trees = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_converter = None
        # set random_state
        if random_state is None:
            self.random_state = np.random.mtrand._rand
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        else:
            raise TypeError("random_state must be a numpy random state, an int or None")

    def fit(self, x, y, data_feature_types=None, verbose=False):
        """
        Fit the ensemble classifier.
        :param x: feature array of the dataset
        :param y: labels of the dataset
        :param data_feature_types: types of each feature (nominal or numeric)
        :param verbose: if True, print additional information to the console
        """
        # check, that the data has a valid structure and valid class types
        checked_x, checked_y, num_classes, self.class_converter = check_data(x, y, suppress_info=not verbose)

        # set the dataset probability for the evidence accumulation approach
        _, counts = np.unique(y, return_counts=True)
        if counts[0]+counts[1] != y.shape[0]:
            raise ValueError("There seems to be a problem with or around the np.unique function")
        ev_help.DATASET_PROBABILITY = counts[1]/(counts[0]+counts[1])

        self.trees = []
        for n in range(self.number_trees):
            tree = Tree(checked_x, checked_y, num_classes, data_feature_types)
            tree.build_tree(self.max_depth, self.min_samples_split, self.min_samples_leaf, self.random_state, verbose)
            self.trees.append(tree)

    def predict(self, x, dt_mode, agg_mode, empty_leafs=3, return_class=True, verbose=False, y=None):
        """
        Predict the classes corresponding to the set of instances x. If y is given, also return y in a formatted form
        as the first element of the results tuple (so that classes are always 0 and 1 and they can be compared with the
        prediction) and the prediction for x as the second. Otherwise, only return the predictions for x.
        :param x: feature array of the dataset for which to make predictions
        :param dt_mode: method to be applied on the single trees (leafs)
        :param agg_mode: method to be applied on top of all results of the dt_mode method
        :param empty_leafs: what should be done when encountering empty leafs (see notes in random_tree.py)
        :param return_class: if True, only return the class of the prediction, not the scores
        :param verbose: if True, print additional information to the console
        :param y: true classes, if not None, the "checked" version of y will be returned (see check_data)
        :return: the prediction if y is None, else a tuple of the "checked" y and the prediction
        """
        dt_output_type, agg_output_type = check_methods(dt_mode, agg_mode, verbose=verbose)

        y_true_formatted = None
        if y is not None:
            x, y_true_formatted, _, _ = check_data(x, y, suppress_info=not verbose)
        else:
            x = check_data(x, suppress_info=not verbose)
        tree_predictions = [tree.predict_value(x, empty_leafs, mode=dt_mode, verbose=verbose) for tree in self.trees]

        tree_pred_array = np.asarray(tree_predictions)

        # for some aggregations, the uncertainty values can be ignored
        if dt_mode == "plaus_values_4" and agg_mode in ["average", "maximum", "median", "voting", "gm_average",
                                                        "gm_maximum", "gm_median"]:
            tree_pred_array_plaus = tree_pred_array[:, :, 0] - tree_pred_array[:, :, 1]
            tpa_shape = tree_pred_array.shape
            tree_pred_array_plaus = tree_pred_array_plaus.reshape((tpa_shape[0], tpa_shape[1], 1))
            tree_pred_array = tree_pred_array_plaus
            dt_output_type = OutputType.S_POS_NEG

        aggregated_predictions = agg.aggregate_results(tree_pred_array, agg_mode, dt_output_type)

        if not return_class:
            if y is not None:
                return y_true_formatted, aggregated_predictions
            else:
                return aggregated_predictions
        if agg_output_type == OutputType.S_POS_NEG:
            aggregated_predictions[aggregated_predictions >= 0] = 1  # class 1 (order here is crucial)
            aggregated_predictions[aggregated_predictions < 0] = 0  # class 0
        elif agg_output_type == OutputType.S_0_1:
            aggregated_predictions[aggregated_predictions < 0.5] = 0  # class 0
            aggregated_predictions[aggregated_predictions >= 0.5] = 1  # class 1
        else:
            raise ValueError(f"\"{mode}\" with \"{agg_mode}\" does not result in class predictions")
        if y is not None:
            return y_true_formatted, aggregated_predictions
        else:
            return aggregated_predictions

    def method_analysis(self, new_x, dt_modes, agg_modes, empty_leafs, y_true, verbose=False):
        """Similar to the predict methods, but allows applying multiple prediction methods on the same trees. In this
        method, the trees are only traversed once, which would have to happen multiple times if the predict method was
        used multiple times. Some intermediate values can be resused, in general. This method still has to be called
        multiple times for new instances or different empty_leafs values. Returns both class scores and class
        predictions. """
        new_x, true_y, _, _ = check_data(new_x, y=y_true, suppress_info=not verbose,
                                         class_converter=self.class_converter)

        nr_methods = 0
        for modes in agg_modes.values():
            nr_methods += len(modes)

        results_array = np.zeros((nr_methods, 2))

        tree_leafs_orig = [tree.predict(new_x, empty_leafs, verbose=verbose) for tree in self.trees]

        tree_leafs = []
        leafs_per_instance = []
        if empty_leafs == 2:
            for tree_iter in tree_leafs_orig:
                tl, lpi = prepare_empty_leafs_2(tree_iter)
                tree_leafs.append(tl)
                leafs_per_instance.append(lpi)
        else:
            tree_leafs = tree_leafs_orig

        if tree_leafs[0].shape[1] != 2:
            raise ValueError("This implementation can only work with two classes. The implementation should be updated "
                             "or a different dataset or the simple class prediction method should be used.")

        row_count_results = 0
        for mode in dt_modes:
            tree_values = [prediction_methods.method_switcher(leafs, mode, np.sum(self.trees[0].root.classes))
                           for leafs in tree_leafs]

            if empty_leafs == 2:
                tree_values_new = []
                for tree_iter, lpi in zip(tree_values, leafs_per_instance):
                    temp = finish_empty_leafs_2(tree_iter, mode, new_x, lpi)
                    if len(temp.shape) == 1:
                        temp = temp.reshape((temp.shape[0], 1))
                    tree_values_new.append(temp)
                tree_values = tree_values_new

            tree_pred_array = np.asarray(tree_values)
            if len(tree_pred_array.shape) == 2:
                tree_pred_array = tree_pred_array.reshape((tree_pred_array.shape[0], tree_pred_array.shape[1], 1))

            tree_pred_array_plaus = None
            if mode == "plaus_values_4":
                tree_pred_array_plaus = tree_pred_array[:, :, 0] - tree_pred_array[:, :, 1]
                tpa_shape = tree_pred_array.shape
                tree_pred_array_plaus = tree_pred_array_plaus.reshape((tpa_shape[0], tpa_shape[1], 1))

            for agg_mode in agg_modes[mode]:
                dt_output_type, agg_output_type = check_methods(mode, agg_mode, verbose=verbose)

                # need to do some tricks to have plausibility be an input to "normal" aggregation as well as advanced on
                # the same run
                if mode == "plaus_values_4" and agg_mode in ["average", "maximum", "median", "voting", "gm_average",
                                                             "gm_maximum", "gm_median"]:
                    aggregation_input = tree_pred_array_plaus
                    dt_output_type_input = OutputType.S_POS_NEG
                else:
                    aggregation_input = tree_pred_array
                    dt_output_type_input = dt_output_type

                aggregated_predictions = agg.aggregate_results(aggregation_input, agg_mode, dt_output_type_input)

                aggregated_classes = np.zeros_like(aggregated_predictions)
                if agg_output_type == OutputType.S_POS_NEG:
                    aggregated_classes[aggregated_predictions >= 0] = 1  # class 1 (order here is crucial)
                    aggregated_classes[aggregated_predictions < 0] = 0  # class 0
                elif agg_output_type == OutputType.S_0_1:
                    aggregated_classes[aggregated_predictions < 0.5] = 0  # class 0
                    aggregated_classes[aggregated_predictions >= 0.5] = 1  # class 1
                else:
                    raise ValueError(f"\"{mode}\" with \"{agg_mode}\" does not result in class predictions")

                accuracy = accuracy_score(true_y, aggregated_classes)
                fpr, tpr, _ = roc_curve(true_y, aggregated_predictions)
                auc_score = auc(fpr, tpr)
                results_array[row_count_results] = np.array([accuracy, auc_score])
                row_count_results += 1

        return results_array

    def get_leafs(self, per_tree=False):
        """ Return all leafs for all trees in the ensemble. """
        leafs = []
        for tree in self.trees:
            if per_tree:
                leafs.append(tree.get_leafs())
            else:
                leafs.extend(tree.get_leafs())
        return leafs
