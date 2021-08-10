import numpy as np
from enum import Enum


class OutputType(Enum):
    INSTANCES = 0  # get a number of instances for each class (for each input instance)
    CLASS = 1  # get a class for each input instance
    S_0_1 = 2  # get a score for one class which is between 0 and 1 for each input instance
    S_POS_NEG = 3  # get a score for one class which is > 0 for the positive and > 0 for the negative class for
    # each input instance
    DIST = 4  # get a distribution (here: always represented by 101 values) for each input instance
    PLAUS = 5  # get the 4 plausibility related values for each input instance


DT_OUTPUT = {"instances": OutputType.INSTANCES, "class": OutputType.CLASS, "probability": OutputType.S_0_1,
             "linear": OutputType.S_POS_NEG, "beta_1_1": OutputType.S_0_1, "beta_prior_leaf": OutputType.S_0_1,
             "ucb": OutputType.S_POS_NEG, "plausibility": OutputType.S_POS_NEG, "fast_cb": OutputType.S_0_1,
             "raw_beta_dist": OutputType.DIST, "plaus_values_4": OutputType.PLAUS}
for i in range(18):
    cb_method = "cb" + str(i)
    if i < 9:
        DT_OUTPUT[cb_method] = OutputType.S_0_1
    else:
        DT_OUTPUT[cb_method] = OutputType.S_POS_NEG

AGG_OUTPUT = {("average", OutputType.INSTANCES): OutputType.INSTANCES,
              ("average", OutputType.CLASS): OutputType.CLASS,
              ("average", OutputType.S_0_1): OutputType.S_0_1,
              ("average", OutputType.S_POS_NEG): OutputType.S_POS_NEG,
              ("average", OutputType.PLAUS): OutputType.S_POS_NEG,
              ("maximum", OutputType.S_0_1): OutputType.S_0_1,
              ("maximum", OutputType.S_POS_NEG): OutputType.S_POS_NEG,
              ("maximum", OutputType.PLAUS): OutputType.S_POS_NEG,
              ("median", OutputType.S_0_1): OutputType.S_0_1,
              ("median", OutputType.S_POS_NEG): OutputType.S_POS_NEG,
              ("median", OutputType.PLAUS): OutputType.S_POS_NEG,
              ("voting", OutputType.S_0_1): OutputType.S_0_1,
              ("voting", OutputType.S_POS_NEG): OutputType.S_POS_NEG,
              ("voting", OutputType.PLAUS): OutputType.S_POS_NEG,
              ("gm_average", OutputType.S_0_1): OutputType.S_0_1,
              ("gm_average", OutputType.S_POS_NEG): OutputType.S_POS_NEG,
              ("gm_average", OutputType.PLAUS): OutputType.S_POS_NEG,
              ("gm_median", OutputType.S_0_1): OutputType.S_0_1,
              ("gm_median", OutputType.S_POS_NEG): OutputType.S_POS_NEG,
              ("gm_median", OutputType.PLAUS): OutputType.S_POS_NEG,
              ("gm_maximum", OutputType.S_0_1): OutputType.S_0_1,
              ("gm_maximum", OutputType.S_POS_NEG): OutputType.S_POS_NEG,
              ("gm_maximum", OutputType.PLAUS): OutputType.S_POS_NEG,
              ("pooling", OutputType.INSTANCES): OutputType.S_0_1,
              ("beta_agg_sum", OutputType.DIST): OutputType.S_0_1,
              ("beta_agg_max", OutputType.DIST): OutputType.S_0_1,
              ("dempster", OutputType.PLAUS): OutputType.S_POS_NEG,
              ("cautious", OutputType.PLAUS): OutputType.S_POS_NEG,
              ("epistemic_weight", OutputType.PLAUS): OutputType.S_POS_NEG,
              ("evidence", OutputType.INSTANCES): OutputType.S_POS_NEG
              }


def check_methods(dt_mode, agg_mode, verbose=True):
    """ Checks, whether the decision tree prediction and the aggregation method are compatible and returns an indicator
    for the output type of the single decision trees as well as the aggregation. """
    if dt_mode not in DT_OUTPUT.keys():
        raise ValueError(f"There is no decision tree prediction method with the name \"{dt_mode}\"")
    agg_modes = set(name for (name, output_type) in AGG_OUTPUT.keys())
    if agg_mode not in agg_modes:
        raise ValueError(f"There is no aggregation method with the name \"{agg_mode}\"")
    dt_output_type = DT_OUTPUT[dt_mode]
    if verbose:
        print(f"Each decision tree will return {dt_output_type.name.lower()} for each input instance")
    agg_input = (agg_mode, dt_output_type)
    if agg_input not in AGG_OUTPUT.keys():
        raise ValueError(
            f"The aggregation method {agg_mode} is not compatible with the decision tree prediction method {dt_mode}")
    agg_output_type = AGG_OUTPUT[agg_input]
    if verbose:
        print(f"The aggregation will return {agg_output_type.name.lower()} for each input instance")
    return dt_output_type, agg_output_type


def unroll_empty_leafs_2(classes):
    """ empty_leafs=2 makes for some complicated data structures. This code makes it so that at least the lists within
    the classes list are not further nested."""
    # unroll nested lists
    classes_unrolled = []
    for classes_for_instance in classes:
        if type(classes_for_instance) is np.ndarray:
            classes_unrolled.append(classes_for_instance)
        else:
            classes_unrolled.append(unroll_list(classes_for_instance))
    return classes_unrolled


def unroll_list(nested_list):
    """ Given a list which contains np arrays and/or further lists with np arrays, make it so that the returned list
    only contains np arrays (from those inner lists)."""
    to_return_list = []
    for elem in nested_list:
        if isinstance(elem, list):
            to_return_list = to_return_list + unroll_list(elem)
        else:
            to_return_list.append(elem)
    return to_return_list


def prepare_empty_leafs_2(classes):
    """ Transforms the data structured which combines lists and arrays into a numpy array data structure. Also returns
    leafs_per_instance which makes it possible to afterwards reconstruct which leaf data (instances) belonged to which
    prediction instance. """
    unrolled_classes = unroll_empty_leafs_2(classes)

    # transform classes into a numpy array, so that every leaf per instance (can be more than 1 per tree) has a row,
    # also save a list to be able to reassign those rows to the prediction instance
    leafs_per_instance = []
    for classes_for_instance in unrolled_classes:
        if type(classes_for_instance) is np.ndarray:
            leafs_per_instance.append(1)
        else:
            leafs_per_instance.append(len(classes_for_instance))
    new_classes = np.zeros((sum(leafs_per_instance), 2))
    current_row = 0
    for classes_for_instance in unrolled_classes:
        if type(classes_for_instance) is np.ndarray:
            new_classes[current_row] = classes_for_instance
            current_row += 1
        else:
            for leaf_values in classes_for_instance:
                new_classes[current_row] = leaf_values
                current_row += 1
    if current_row != new_classes.shape[0]:
        raise Exception("Something seems to be wrong with the empty_leafs=2 prediction results.")
    return new_classes, leafs_per_instance


def finish_empty_leafs_2(to_return, mode, new_x, leafs_per_instance):
    """ Because we now have predictions for each row as created in prepare_empty_leafs_2 that means that we can now
    have multiple predictions for a single instance. Here, these predictions are averaged so that we can then, for the
    aggregation over the different trees, have one single prediction for each instance (and tree) again. """
    if mode == "instances":
        to_return_2 = np.zeros((new_x.shape[0], 2))
    elif mode == "plaus_values_4":
        to_return_2 = np.zeros((new_x.shape[0], 4))
    elif mode == "raw_beta_dist":
        to_return_2 = np.zeros((new_x.shape[0], 101))
    else:
        to_return_2 = np.zeros((new_x.shape[0], 1))
    start = 0
    for count, instance_leafs in enumerate(leafs_per_instance):
        current_preds = to_return[start:start + instance_leafs]
        start += instance_leafs
        to_return_2[count] = np.average(current_preds, axis=0)
    to_return = to_return_2
    return to_return
