import scipy.special as scisp
from scipy.special import gammaln
from scipy.stats import binom, beta
import matplotlib.pyplot as plt
import numpy as np
import math

"""
This file contains methods for the binomial, beta and beta-binomial distribution as well as a new method inspired by the
Monte Carlo tree search UCB approach.
"""


def binomial(pos, neg, graph=False, return_area=True):
    """ Calculates the overlapping area of the positive and negative binomial curve by taking the minimum of both
    curves. This works and returns a value within [0,1] because the curves are discrete. Also shows the graph if
    graph=True. By default, returns that area but if return_area=False, the height is returned (height being the
    intersection between both curves, beware that this value is not normalized, so not always in [0,1]). """
    # idea of area calculation: take the non continuous binomial graph and
    # use the sum of the minimum, a smaller sum is better
    pos_prob = pos / (pos + neg)
    neg_prob = neg / (pos + neg)
    x = np.arange(0, pos + neg + 1)
    pos_graph = binom.pmf(x, pos + neg, pos_prob)
    neg_graph = binom.pmf(x, pos + neg, neg_prob)
    if graph:
        plt.plot(x, pos_graph, "g", label="positive binom")
        plt.plot(x, neg_graph, "r", label="negative binom")
        plt.legend(loc="best", frameon=False)
        plt.show()
    if return_area:
        min_graph = np.minimum(pos_graph, neg_graph)
        area = np.sum(min_graph)
        return area
    i_point = (pos + neg) / 2.0
    if i_point.is_integer():
        # use the middle point if possible
        height = pos_graph[int(i_point)]
    else:
        # otherwise, use the average of both middle points
        lower = pos_graph[int(i_point - 0.5)]
        upper = pos_graph[int(i_point + 0.5)]
        height = (lower + upper) / 2.0
    max_value = max(pos_graph)
    return float(height) / float(max_value)


def binomial_values(pos, neg):
    """ Same as binomial but return area, height and normalized height values. """
    # idea of area calculation: take the non continuous binomial graph and
    # use the sum of the minimum, a smaller sum is better
    pos_prob = pos / (pos + neg)
    neg_prob = neg / (pos + neg)
    x = np.arange(0, pos + neg + 1)
    pos_graph = binom.pmf(x, pos + neg, pos_prob)
    neg_graph = binom.pmf(x, pos + neg, neg_prob)
    min_graph = np.minimum(pos_graph, neg_graph)
    area = np.sum(min_graph)
    i_point = (pos + neg) / 2.0
    if i_point.is_integer():
        # use the middle point if possible
        height = pos_graph[int(i_point)]
    else:
        # otherwise, use the average of both middle points
        lower = pos_graph[int(i_point - 0.5)]
        upper = pos_graph[int(i_point + 0.5)]
        height = (lower + upper) / 2.0
    max_value = max(pos_graph)
    norm_height = float(height) / float(max_value)
    return area, height, norm_height


def cb_beta(a, b, graph=False, return_area=True, steps=0.001):
    """ Calculates the overlapping area of the (continuous) positive and negative beta curve by taking the minimum of
    both curves. This works and returns a value within [0,1] because the curves are discrete. Also shows the graph if
    graph=True. By default, returns that area but if return_area=False, the height is return (height being the
    intersection between both curves, beware that this value is not normalized, so not always in [0,1]).
    Keep in mind, that this function works numerically approximated (since i can not compute a "complete" continuous
    function). Therefore the returned values are also approximated and the are might succeed 1 by a bit. """
    x = np.arange(0, 1 + steps, steps)
    pos_graph = beta.pdf(x, a, b)
    neg_graph = beta.pdf(x, b, a)
    if graph:
        plt.plot(x, pos_graph, "g", label="positive beta")
        plt.plot(x, neg_graph, "r", label="negative beta")
        plt.legend(loc="best", frameon=False)
        plt.show()
    if return_area:
        min_graph = np.minimum(pos_graph, neg_graph)
        area = np.trapz(min_graph, x)
        return area
    # use the middle point or the point closest to it
    middle_point = round(1.0 / (steps * 2))
    height = pos_graph[middle_point]
    max_value = max(pos_graph)
    return float(height) / float(max_value)


def beta_distribution(a, b, steps=0.001):
    """ Calculates a beta distribution. """
    x = np.arange(0, 1 + steps, steps)
    graph = beta.pdf(x, a, b)
    return graph


def cb_beta_values(a, b, steps=0.001):
    """ Same as cb_beta but return area, height and normalized height values. """
    x = np.arange(0, 1 + steps, steps)
    pos_graph = beta.pdf(x, a, b)
    neg_graph = beta.pdf(x, b, a)
    min_graph = np.minimum(pos_graph, neg_graph)
    area = np.trapz(min_graph, x)
    # use the middle point or the point closest to it
    middle_point = round(1.0 / (steps * 2))
    height = pos_graph[middle_point]
    max_value = max(pos_graph)
    norm_height = float(height) / float(max_value)
    return area, height, norm_height


def beta_binomial(n, a, b, graph=False, return_area=True):
    """ Calculates the overlapping area of the positive and negative beta-binomial curve by taking the minimum of both
    curves. This works and returns a value within [0,1] because the curves are discrete. Also shows the graph if
    graph=True. By default, returns that area but if return_area=False, the height is return (height being the
    intersection between both curves, beware that this value is not normalized, so not always in [0,1]). """
    x = np.arange(0, n + 1)
    pos_graph = np.zeros_like(x, dtype=float)
    neg_graph = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        ln_answer = gammaln(n + 1) + gammaln(x[i] + a) + gammaln(n - x[i] + b) + gammaln(a + b) - \
                    (gammaln(x[i] + 1) + gammaln(n - x[i] + 1) + gammaln(a) + gammaln(b) + gammaln(n + a + b))
        pos_graph[i] = math.exp(ln_answer)
        ln_answer = gammaln(n + 1) + gammaln(x[i] + b) + gammaln(n - x[i] + a) + gammaln(b + a) - \
                    (gammaln(x[i] + 1) + gammaln(n - x[i] + 1) + gammaln(b) + gammaln(a) + gammaln(n + b + a))
        neg_graph[i] = math.exp(ln_answer)
    if graph:
        plt.plot(x, pos_graph, "g", label="positive binom")
        plt.plot(x, neg_graph, "r", label="negative binom")
        plt.legend(loc="best", frameon=False)
        plt.show()
    if return_area:
        min_graph = np.minimum(pos_graph, neg_graph)
        area = np.sum(min_graph)
        return area
    i_point = n / 2.0
    if i_point.is_integer():
        # use the middle point if possible
        height = pos_graph[int(i_point)]
    else:
        # otherwise, use the average of both middle points
        lower = pos_graph[int(i_point - 0.5)]
        upper = pos_graph[int(i_point + 0.5)]
        height = (lower + upper) / 2.0
    max_value = max(pos_graph)
    if max_value == 0:
        return height
    return float(height) / float(max_value)


def beta_binomial_values(n, a, b):
    """ Same as beta_binomial but return area, height and normalized height values. """
    x = np.arange(0, n + 1)
    pos_graph = np.zeros_like(x, dtype=float)
    neg_graph = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        ln_answer = gammaln(n + 1) + gammaln(x[i] + a) + gammaln(n - x[i] + b) + gammaln(a + b) - \
                    (gammaln(x[i] + 1) + gammaln(n - x[i] + 1) + gammaln(a) + gammaln(b) + gammaln(n + a + b))
        pos_graph[i] = math.exp(ln_answer)
        ln_answer = gammaln(n + 1) + gammaln(x[i] + b) + gammaln(n - x[i] + a) + gammaln(b + a) - \
                    (gammaln(x[i] + 1) + gammaln(n - x[i] + 1) + gammaln(b) + gammaln(a) + gammaln(n + b + a))
        neg_graph[i] = math.exp(ln_answer)
    min_graph = np.minimum(pos_graph, neg_graph)
    area = np.sum(min_graph)
    i_point = n / 2.0
    if i_point.is_integer():
        # use the middle point if possible
        height = pos_graph[int(i_point)]
    else:
        # otherwise, use the average of both middle points
        lower = pos_graph[int(i_point - 0.5)]
        upper = pos_graph[int(i_point + 0.5)]
        height = (lower + upper) / 2.0
    max_value = max(pos_graph)
    if max_value == 0:
        norm_height = height
    else:
        norm_height = float(height) / float(max_value)
    return area, height, norm_height


def fast_beta_binomial(n, a, b):
    """ Same as beta_binomial_values but only return the normalized height (faster). """
    i_point = n / 2.0
    if i_point.is_integer():
        # use the middle point if possible
        i = int(i_point)
        height = gammaln(n + 1) + gammaln(i + a) + gammaln(n - i + b) + gammaln(a + b) - \
                 (gammaln(i + 1) + gammaln(n - i + 1) + gammaln(a) + gammaln(b) + gammaln(n + a + b))
        height = math.exp(height)
    else:
        # otherwise, use the average of both middle points
        i_lower = int(i_point - 0.5)
        i_upper = int(i_point + 0.5)
        height_lower = gammaln(n + 1) + gammaln(i_lower + a) + gammaln(n - i_lower + b) + gammaln(a + b) - \
                       (gammaln(i_lower + 1) + gammaln(n - i_lower + 1) + gammaln(a) + gammaln(b) + gammaln(n + a + b))
        height_lower = math.exp(height_lower)
        height_upper = gammaln(n + 1) + gammaln(i_upper + a) + gammaln(n - i_upper + b) + gammaln(a + b) - \
                       (gammaln(i_upper + 1) + gammaln(n - i_upper + 1) + gammaln(a) + gammaln(b) + gammaln(n + a + b))
        height_upper = math.exp(height_upper)
        height = (height_lower + height_upper) / 2.0
    prob = (a - 1) / (a + b - 2)  # used for max
    prob_point = prob * n
    if prob_point.is_integer():
        max_point = int(prob_point)
        max_value = gammaln(n + 1) + gammaln(max_point + a) + gammaln(n - max_point + b) + gammaln(a + b) - \
                    (gammaln(max_point + 1) + gammaln(n - max_point + 1) + gammaln(a) + gammaln(b) + gammaln(n + a + b))
        max_value = math.exp(max_value)
    else:
        max_lower = int(prob_point)
        max_upper = int(prob_point) + 1
        max_value_1 = gammaln(n + 1) + gammaln(max_lower + a) + gammaln(n - max_lower + b) + gammaln(a + b) - \
                      (gammaln(max_lower + 1) + gammaln(n - max_lower + 1) + gammaln(a) + gammaln(b) + gammaln(
                          n + a + b))
        max_value_1 = math.exp(max_value_1)
        max_value_2 = gammaln(n + 1) + gammaln(max_upper + a) + gammaln(n - max_upper + b) + gammaln(a + b) - \
                      (gammaln(max_upper + 1) + gammaln(n - max_upper + 1) + gammaln(a) + gammaln(b) + gammaln(
                          n + a + b))
        max_value_2 = math.exp(max_value_2)
        max_value = max(max_value_1, max_value_2)
    if max_value == 0:
        norm_height = height
    else:
        norm_height = float(height) / float(max_value)
    return norm_height


def ucb(pos, neg, all_instances):
    """ Uses the UCB formula used in Monte Carlo tree search. Number of wins -> number of positive instances. Number of
    games -> number of positive and negative instances. Total games -> number of instances used for training
    (all_instances parameter). Returns the alpha value for which the UCB formula would result in 0.5. """
    # exploit: probability
    number_instances = pos + neg
    prob = float(pos) / float(number_instances)
    exploit = prob
    # explore = square_root(log(#allInstancesofTraining)/#allhere))
    log_all = math.log(float(all_instances))
    temp = log_all / float(number_instances)
    explore = math.sqrt(temp)
    exploit_pos = exploit
    exploit_neg = 1 - exploit
    alpha = (exploit_pos - exploit_neg) / (2 * explore)
    return alpha
