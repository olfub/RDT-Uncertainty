import numpy as np
from prediction.prediction_utility import OutputType
import prediction.extended_aggregation as e_agg
import prediction.evidence_helper as ev_help
import math


def aggregate_results(tree_pred_array, agg_mode, dt_output_type):
    """ Combine the results from the decision trees (scores or other information) to get final prediction scores. """
    if agg_mode in ["average", "maximum", "median", "voting"]:
        agg_pred = aggregate_simple(tree_pred_array, agg_mode, dt_output_type)
    elif agg_mode in ["gm_average", "gm_maximum", "gm_median"]:
        agg_pred = aggregate_gm(tree_pred_array, agg_mode, dt_output_type)
    elif agg_mode == "pooling":
        agg_pred = pooling(tree_pred_array, agg_mode, dt_output_type)
    elif agg_mode in ["beta_agg_sum", "beta_agg_max"]:
        agg_pred = beta_agg(tree_pred_array, agg_mode, dt_output_type)
    elif agg_mode == "dempster":
        agg_pred = dempster(tree_pred_array)
    elif agg_mode == "cautious":
        agg_pred = cautious(tree_pred_array)
    elif agg_mode == "epistemic_weight":
        agg_pred = epistemic_weight(tree_pred_array)
    elif agg_mode == "evidence":
        agg_pred = evidence(tree_pred_array)
    else:
        raise Exception(f"Mode {agg_mode} is not supported (yet)")
    return agg_pred


def aggregate_simple(tree_pred_array, agg_mode, dt_output_type):
    """ Calculate average, maximum, median, or voting. """
    agg_pred = None
    if agg_mode == "average":
        agg_pred = np.average(tree_pred_array, axis=0)
    elif agg_mode == "maximum":
        if dt_output_type == OutputType.S_0_1:
            temp = np.argmax(abs(tree_pred_array - 0.5), axis=0)
        elif dt_output_type == OutputType.S_POS_NEG:
            temp = np.argmax(abs(tree_pred_array), axis=0)
        else:
            raise Exception(f"Data Format {str(dt_output_type)} is not supported for the maximum aggregation (yet)")
        agg_pred = np.zeros((tree_pred_array.shape[1], 1))
        for i in range(tree_pred_array.shape[1]):
            agg_pred[i] = tree_pred_array[temp[i], i]
    elif agg_mode == "median":
        agg_pred = np.median(tree_pred_array, axis=0)
    elif agg_mode == "voting":  # unweighted
        if dt_output_type == OutputType.S_0_1:
            temp_pos = np.where(tree_pred_array > 0.5, 1.0, 0.0)
            temp_neg = np.where(tree_pred_array < 0.5, -1.0, 0.0)
            temp = temp_pos + temp_neg
            votes = np.sum(temp, axis=0)
        elif dt_output_type == OutputType.S_POS_NEG:
            temp_pos = np.where(tree_pred_array > 0, 1.0, 0.0)
            temp_neg = np.where(tree_pred_array < 0, -1.0, 0.0)
            temp = temp_pos + temp_neg
            votes = np.sum(temp, axis=0)
        else:
            raise Exception(f"Data Format {str(dt_output_type)} is not supported for the voting aggregation (yet)")
        if dt_output_type == OutputType.S_0_1:
            factor = 1 / (2 * tree_pred_array.shape[0])
            agg_pred = np.full_like(votes, 0.5, dtype=np.float64) + factor * votes
        elif dt_output_type == OutputType.S_POS_NEG:
            agg_pred = np.full_like(votes, 0, dtype=np.float64) + votes
    else:
        raise Exception(f"Mode {agg_mode} is not supported (yet)")
    return agg_pred


def aggregate_gm(tree_pred_array, agg_mode, dt_output_type):
    """ Applies Generalized Mixture Functions from "Combining multiple algorithms in classifier ensembles using
    generalized mixture functions" (Costa, 2018) """
    n = tree_pred_array.shape[0]

    # to avoid a division by zero later
    if n < 2:
        raise ValueError("There must be at least two classifier results to aggregate via a gm function.")

    # calculate the comparison function
    thetas = None
    if agg_mode == "gm_average":
        thetas = np.average(tree_pred_array, axis=0)
    elif agg_mode == "gm_median":
        thetas = np.median(tree_pred_array, axis=0)
    elif agg_mode == "gm_maximum":
        if dt_output_type == OutputType.S_0_1:
            temp = np.argmax(abs(tree_pred_array - 0.5), axis=0)
        elif dt_output_type == OutputType.S_POS_NEG:
            temp = np.argmax(abs(tree_pred_array), axis=0)
        else:
            raise Exception(f"Data Format {str(dt_output_type)} is not supported for the maximum aggregation (yet)")
        thetas = np.zeros((tree_pred_array.shape[1], 1))
        for i in range(tree_pred_array.shape[1]):
            thetas[i] = tree_pred_array[temp[i], i]

    # the divisor stays constant for every iteration and only needs to be calculated once
    orig_constants = np.zeros((tree_pred_array.shape[1], 1))
    for j in range(n):
        orig_constants += np.absolute(tree_pred_array[j] - thetas)

    # constant == 0 if and only if x_1 = ... = x_n (see formula), then all temp variables in the following loop we also
    # be 0, so we can divide those by 1 instead of 0 in order to not divide by 0 and correct later
    constants = np.where(orig_constants == 0, 1, orig_constants)
    # print(np.array_equal(constants, orig_constants))

    # loop over every classifier result
    results_sum = np.zeros((tree_pred_array.shape[1], 1))
    for i in range(n):
        x_i = tree_pred_array[i]
        temp = abs(x_i - thetas)
        temp *= x_i
        temp = temp / constants
        results_sum += (x_i - temp)
    result = (1 / (n - 1)) * results_sum

    constant_indices = np.where(orig_constants == 0)
    if len(constant_indices[0]) != 0:
        result[constant_indices] = tree_pred_array[0][constant_indices]
    return result


def pooling(tree_pred_array, *_):
    """ Implementation of the pooling aggregation. """
    results = np.zeros((tree_pred_array.shape[1], 1))
    pooled = np.sum(tree_pred_array, axis=0)
    for i in range(tree_pred_array.shape[1]):
        if (pooled[i, 0] + pooled[i, 1]) == 0:
            results[i] = 0.5
        else:
            results[i] = pooled[i, 1] / (pooled[i, 0] + pooled[i, 1])
    return results


def beta_agg(tree_pred_array, agg_mode, _):
    """ Implementation of the beta aggregation idea. """
    final_distribution = np.average(tree_pred_array, axis=0)
    if agg_mode == "beta_agg_max":
        length = final_distribution.shape[1]
        maxima = np.argmax(final_distribution, axis=1)
        return maxima/(length-1)
    elif agg_mode == "beta_agg_sum":
        # we assume that we have an uneven number of values, then, between pos_area and neg_area, we leave out the
        # middle one
        length = final_distribution.shape[1]
        neg_area = np.sum(final_distribution[:, :length//2], axis=1)
        pos_area = np.sum(final_distribution[:, (length//2)+1:], axis=1)
        results = pos_area / (pos_area+neg_area)
        return results
    else:
        print("Error, mode %s not defined (yet)" % str(agg_mode))


def dempster(tree_pred_array, normalized=False):
    """ Calls dempster's rule of combination. """
    results = np.zeros((tree_pred_array.shape[1], 1))
    for i in range(tree_pred_array.shape[1]):
        mass_functions = np.zeros_like(tree_pred_array[:, i, :])
        mass_functions[:, 3] = tree_pred_array[:, i, 2] + tree_pred_array[:, i, 3]
        mass_functions[:, 2] = tree_pred_array[:, i, 1]
        mass_functions[:, 1] = tree_pred_array[:, i, 0]
        final_masses = e_agg.dempster_n(mass_functions, normalized=normalized)[0]
        results[i] = final_masses[1] - final_masses[2]
    return results


def cautious(tree_pred_array, normalized=False):
    """ Calls the cautious rule. """
    results = np.zeros((tree_pred_array.shape[1], 1))
    for i in range(tree_pred_array.shape[1]):
        mass_functions = np.zeros_like(tree_pred_array[:, i, :])
        mass_functions[:, 3] = tree_pred_array[:, i, 2] + tree_pred_array[:, i, 3]
        mass_functions[:, 2] = tree_pred_array[:, i, 1]
        mass_functions[:, 1] = tree_pred_array[:, i, 0]
        final_masses = e_agg.cautious_n(mass_functions, normalized=normalized)[0]
        results[i] = final_masses[1] - final_masses[2]
    return results


def epistemic_weight(tree_pred_array):
    """ Implementation of our prediction methods which applies weighting corresponding the epistemic uncertainty. """
    # dim: n, i, 4
    results = np.zeros((tree_pred_array.shape[1], 1))
    number_clf = tree_pred_array.shape[0]
    for i in range(tree_pred_array.shape[1]):
        pred_sum = 0
        for j in range(number_clf):
            pred_sum += (1-tree_pred_array[j, i, 3])*(tree_pred_array[j, i, 0] - tree_pred_array[j, i, 1])
        results[i] = (pred_sum / number_clf)
    return results


def evidence(tree_pred_array):
    """ Implementation of the evidence accumulation prediction. """
    results = np.zeros((tree_pred_array.shape[1], 1))
    number_clf = tree_pred_array.shape[0]
    dataset_prob = ev_help.DATASET_PROBABILITY
    for i in range(tree_pred_array.shape[1]):
        sum_plus = 0
        sum_minus = 0
        for n in range(number_clf):
            neg = tree_pred_array[n, i, 0]
            pos = tree_pred_array[n, i, 1]
            # laplace correction
            neg += 0.1
            pos += 0.1
            p_y_given_x_plus = pos / (pos + neg)
            sum_plus += math.log(p_y_given_x_plus)
            p_y_given_x_minus = neg / (pos + neg)
            sum_minus += math.log(p_y_given_x_minus)
        positive_pred = sum_plus - (number_clf - 1) * dataset_prob
        negative_pred = sum_minus - (number_clf - 1) * (1 - dataset_prob)
        results[i] = positive_pred - negative_pred
    return results
