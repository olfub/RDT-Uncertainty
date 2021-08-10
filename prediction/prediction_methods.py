import numpy as np
from prediction import solve_plausibility
from prediction import confidence_bounds
import pickle


BETA_DIST_STEPS = 0.01


def method_switcher(classes, mode, all_instances):
    """ Given the leaf classes, apply the prediction method defined by the mode parameter. The all_instances parameter
    only matters for the ucb prediction."""
    if mode == "instances":
        to_return = classes
    elif mode == "probability":
        to_return = probability(classes)
    elif mode == "linear":
        to_return = linear_instances(classes)
    elif mode == "beta_1_1":
        to_return = beta_prediction(classes, np.ones_like(classes))
    elif mode == "ucb":
        to_return = ucb(classes, all_instances)
    elif mode == "plausibility":
        to_return = plausability(classes)
    elif mode.startswith("cb"):
        mode_number = int(mode[2:])
        if mode_number < 0 or mode_number > 17:
            raise ValueError("The confidence bound prediction method must be passed with an mode number between "
                             "(including) 0 and 17")
        to_return = hash_pred(classes, mode_number)
    elif mode == "fast_cb":
        to_return = only_best_cb(classes)
    elif mode == "raw_beta_dist":
        to_return = beta_dist(classes)
    elif mode == "plaus_values_4":
        to_return = plausibility_values(classes, translation=1)
    else:
        raise ValueError(f"Mode: \"{mode}\" not supported")
    return to_return


def linear_instances(leafs):
    """
    Returns the difference between the positive and the negative instances, so that returned values are >1 for
    pos > neg, <1 for neg > pos and =0 otherwise.
    """
    return leafs[:, 1] - leafs[:, 0]


def probability(leafs):
    """
    Returns the probability.
    """
    pos_ins = leafs[:, 1]
    neg_ins = leafs[:, 0]
    prob = np.zeros(pos_ins.shape)
    for i in range(len(pos_ins)):
        p = pos_ins[i]
        n = neg_ins[i]
        if p == n == 0:
            prob[i] = 0.5
        else:
            prob[i] = p/(p+n)
    return prob


def beta_prediction(leafs, priors):
    """
    Priors needs to have the same shape as leafs, because the priors for each leaf can be different. Predicts using beta
    prediction.
    """
    # assumes, either leafs or prior per instance is not 0
    pos_ins = leafs[:, 1]
    neg_ins = leafs[:, 0]
    pos_prior = priors[:, 1]
    neg_prior = priors[:, 0]
    beta_prob = (pos_ins+pos_prior)/(pos_ins+pos_prior+neg_ins+neg_prior)
    return beta_prob


def plausability(leafs):
    """
    Returns the plausibility.
    """
    pos_ins = leafs[:, 1]
    neg_ins = leafs[:, 0]
    plaus = np.zeros(pos_ins.shape)
    for i in range(len(pos_ins)):
        p = pos_ins[i]
        n = neg_ins[i]
        p_plus, p_minus = solve_plausibility.calculate_plausibilities(p, n)
        plaus[i] = p_plus - p_minus
    return plaus


def ucb(leafs, all_instances):
    """
    Return the ucb inspired formula value.
    """
    pos_ins = leafs[:, 1]
    neg_ins = leafs[:, 0]
    ucb_res = np.zeros(pos_ins.shape)
    for i in range(len(pos_ins)):
        p = pos_ins[i]
        n = neg_ins[i]
        if p == n == 0:
            ucb_res[i] = 0
        else:
            ucb_res[i] = confidence_bounds.ucb(p, n, all_instances)
    return ucb_res


def hash_pred(leafs, method_index, verbose=False):
    """ Calculate the confidence bounds methods. This function is designed in a way that it always calculates all 18
    variants because the purpose here is analysis and comparison. If you want efficient methods for single confidence
    bounds predictions, you have to either refer to another method (if possible) or write a new one. But while always
    all methods are calculated if they do not exist within the pickle file, only the one methods result is returned
    which corresponds to the method index."""
    pos_ins = leafs[:, 1]
    neg_ins = leafs[:, 0]
    results = np.zeros_like(pos_ins, dtype=np.float64)
    path = "prediction/cb_hashs/"
    open_string = path + "cb_pred_" + str(method_index) + ".pkl"
    try:
        f = open(open_string, "rb")
        value_dict = pickle.load(f)
        f.close()
    except FileNotFoundError:
        for i in range(18):
            open_string_temp = path + "cb_pred_" + str(i) + ".pkl"
            f_temp = open(open_string_temp, "wb")
            pickle.dump({}, f_temp)
            f_temp.close()
        value_dict = {}
    for i in range(len(pos_ins)):
        pos = pos_ins[i]
        neg = neg_ins[i]
        value_tuple = (pos, neg)
        if value_tuple in value_dict:
            pass
        else:
            bin_tuples = confidence_bounds.binomial_values(pos, neg)
            beta_bin_tuples = confidence_bounds.beta_binomial_values(pos + neg, pos + 1, neg + 1)
            beta_tuples = confidence_bounds.cb_beta_values(pos + 1, neg + 1)
            temp_tuple = bin_tuples + beta_bin_tuples + beta_tuples
            res1_tuple = ()
            res2_tuple = ()
            for temp in temp_tuple:
                prob = pos / (pos + neg)
                res1 = 0.5
                if pos == 0 and neg == 0:
                    res1 = 0.5
                elif neg > pos:
                    res1 -= (1 - temp) * (0.5 - prob)
                else:
                    res1 += (1 - temp) * (prob - 0.5)
                res1_tuple += (res1,)
                res2 = 1 - temp
                if neg > pos:
                    res2 = res2 * -1
                elif pos == neg:
                    # manually set to 0 for pos == neg
                    res2 = 0
                res2_tuple += (res2,)
            res_tuple = res1_tuple + res2_tuple
            for j in range(len(res_tuple)):
                if j == method_index:
                    value_dict[value_tuple] = res_tuple[j]
                else:
                    open_string_temp = path + "cb_pred_" + str(j) + ".pkl"
                    f_temp = open(open_string_temp, "rb")
                    dict_temp = pickle.load(f_temp)
                    dict_temp[value_tuple] = res_tuple[j]
                    f_temp.close()
                    f_temp = open(open_string_temp, "wb")
                    pickle.dump(dict_temp, f_temp)
                    f_temp.close()
            if verbose:
                print("New value calculated and added")
        results[i] = value_dict[value_tuple]
    f = open(open_string, "wb")
    pickle.dump(value_dict, f)
    f.close()
    return results


def only_best_cb(leafs):
    """ Calculates what we found to be the best of our confidence bounds methods (beta binomial distribution, normalized
    height, score calculated as shown in this function). This can be done relatively quickly and without needing to
    store results because not the whole distribution needs to be calculated."""
    pos_ins = leafs[:, 1]
    neg_ins = leafs[:, 0]
    results = np.zeros_like(pos_ins, dtype=np.float64)
    for i in range(len(pos_ins)):
        pos = pos_ins[i]
        neg = neg_ins[i]
        if pos == neg == 0:
            results[i] = 0.5
        else:
            beta_bin_n_height = confidence_bounds.fast_beta_binomial(pos+neg, pos+1, neg+1)
            prob = pos / (pos+neg)
            res = 0.5
            if neg > pos:
                res -= (1-beta_bin_n_height) * (0.5-prob)
            else:
                res += (1-beta_bin_n_height) * (prob-0.5)
            results[i] = res
    return results


def beta_dist(leafs):
    """ Calculate a beta distribution for each leaf. """
    pos_ins = leafs[:, 1]
    neg_ins = leafs[:, 0]
    steps = BETA_DIST_STEPS
    results = np.zeros((pos_ins.shape[0], int(1/steps) + 1))
    for i in range(len(pos_ins)):
        pos = pos_ins[i]
        neg = neg_ins[i]
        results[i] = confidence_bounds.beta_distribution(pos+1, neg+1, steps=steps)
    return results


def plausibility_values(leafs, translation=0):
    """
    Returns all four plausibility values, p_plus, p_minus, u_a and u_e. The translation parameter determines how those
    values are transformed into mass functions.
    """
    pos_ins = leafs[:, 1]
    neg_ins = leafs[:, 0]
    plaus_values = np.zeros((pos_ins.shape[0], 4))
    for i in range(len(pos_ins)):
        p = pos_ins[i]
        n = neg_ins[i]
        _, _, u_e, u_a, _, p_plus, p_minus = solve_plausibility.calculate_plausibility_values(p, n)
        if translation == 0:
            plaus_values[i] = np.array([0, p_plus, p_minus, u_e+u_a])
        if translation == 1:
            plaus_values[i] = np.array([p_plus, p_minus, u_a, u_e])
    return plaus_values
