""" Just some functions of the aggregation file put into a new file to keep that file more clear."""
import numpy as np


def dempster_n(clf_results, normalized=False):
    """
    shape of clf_results: [n][4], n: number of classifiers, 4: mass of "empty", "negative", "positive" and
    "positive or negative".
    Therefore (for simplicity), this method only applies dempster's rule for sets of that shape (so only for frame of
    discernment omega = {+,-}). If n == 1, no normalization will be done and the input will be returned without change.
    Otherwise, normalized==True will apply the normalized dempster's rule of combination instead of the unnormalized
    one.
    """
    # Start changing values for numerical stability
    for i in range(clf_results.shape[0]):
        epsilon = 0.00001
        if clf_results[i, 1] > 1-epsilon:
            clf_results[i, 1] = 1-epsilon
            clf_results[i, 3] = epsilon
        if clf_results[i, 2] > 1-epsilon:
            clf_results[i, 2] = 1-epsilon
            clf_results[i, 3] = epsilon
    # Finish changing values for numerical stability

    # n == 1, return
    if clf_results.shape[0] == 1:
        return clf_results
    number_results = clf_results.shape[0]//2
    if clf_results.shape[0] % 2 == 1:
        number_results += 1
    temp_results = np.zeros((number_results, clf_results.shape[1]))
    for i in range(clf_results.shape[0]//2):
        temp_results[i] = dempster_two(clf_results[2*i], clf_results[2*i+1], normalized)
    if clf_results.shape[0] % 2 == 1:
        temp_results[-1] = clf_results[-1]
    return dempster_n(temp_results, normalized)


def dempster_two(res_1, res_2, normalized=False):
    """
    Calculated dempster's rule of combination of two mass functions res_1 and res_2. If normalized==True, use normalized
    dempster's rule of combination, otherwise use the unnormalized one.
    """

    m_zero = res_1[0]*(res_2[0]+res_2[1]+res_2[2]+res_2[3])
    m_zero += res_1[1]*(res_2[0]+res_2[2])
    m_zero += res_1[2]*(res_2[0]+res_2[1])
    m_zero += res_1[3]*res_2[0]
    norm_factor = 0
    if normalized:
        norm_factor = 1/(1-m_zero)
        m_zero = 0
    m_plus = res_1[1]*(res_2[1]+res_2[3]) + res_1[3]*res_2[1]
    m_minus = res_1[2]*(res_2[2]+res_2[3]) + res_1[3]*res_2[2]
    m_plus_minus = res_1[3]*res_2[3]
    result = np.array([m_zero, m_plus, m_minus, m_plus_minus])
    if normalized:
        result *= norm_factor
    return result


def cautious_n(clf_results, normalized=False):
    """
    shape of clf_results: [n][4], n: number of classifiers, 4: mass of "empty", "negative", "positive" and
    "positive or negative"
    Therefore (for simplicity), this method only applies cautious rule for sets of that shape (so only for frame of
    discernment omega = {+,-}). For n==1 no calculation will be done. If normalized==True, the final result (not the
    intermediate results) will be normalized.
    """
    # Start changing values for numerical stability
    for i in range(clf_results.shape[0]):
        epsilon = 0.00001
        if clf_results[i, 1] > 1-epsilon:
            clf_results[i, 1] = 1-epsilon
            clf_results[i, 3] = epsilon
        if clf_results[i, 2] > 1-epsilon:
            clf_results[i, 2] = 1-epsilon
            clf_results[i, 3] = epsilon
    # Finish changing values for numerical stability

    if clf_results.shape[0] == 1:
        if normalized:
            norm_factor = 1/(1-clf_results[0, 0])
            clf_results[0, 0] = 0
            clf_results = clf_results * norm_factor
        return clf_results
    number_results = clf_results.shape[0]//2
    if clf_results.shape[0] % 2 == 1:
        number_results += 1
    temp_results = np.zeros((number_results, clf_results.shape[1]))
    for i in range(clf_results.shape[0]//2):
        temp_results[i] = cautious_two(np.stack((clf_results[2*i], clf_results[2*i+1])))
    if clf_results.shape[0] % 2 == 1:
        temp_results[-1] = clf_results[-1]
    return cautious_n(temp_results, normalized=normalized)


def cautious_two(clf_results):
    """
    Calculate the cautious rule of two mass functions res_1 and res_2.
    """

    # 1. compute the commonality functions
    coms = np.zeros_like(clf_results)
    for i in range(clf_results.shape[0]):
        q_zero = np.sum(clf_results[i])
        q_plus = clf_results[i, 1] + clf_results[i, 3]
        q_minus = clf_results[i, 2] + clf_results[i, 3]
        q_plus_minus = clf_results[i, 3]
        coms[i] = np.array([q_zero, q_plus, q_minus, q_plus_minus])
    # print(coms)
    # 2. compute the weight functions
    ws = np.zeros((clf_results.shape[0], 3))
    for i in range(coms.shape[0]):
        w_zero = (1/coms[i, 0]) * coms[i, 1] * coms[i, 2] * (1/coms[i, 3])
        w_plus = (1/coms[i, 1]) * coms[i, 3]
        w_minus = (1/coms[i, 2]) * coms[i, 3]
        ws[i] = np.array([w_zero, w_plus, w_minus])
    # print(ws)
    # 3. combine weight function (if i understand it correctly, this step is the t-norm)
    w_1_2 = np.minimum(ws[0], ws[1])  # w_1_2 = ws[0]*ws[1] for dempster's rule
    # print(w_1_2)
    # 4. extract simple mass function
    mass_zero = np.array([1-w_1_2[0], 0, 0, w_1_2[0]])
    mass_plus = np.array([0, 1-w_1_2[1], 0, w_1_2[1]])
    mass_minus = np.array([0, 0, 1-w_1_2[2], w_1_2[2]])
    # 5. apply unnormalized dempster's rule to all generated mass functions
    all_masses = np.array([mass_zero, mass_plus, mass_minus])
    # print(all_masses)
    result = dempster_n(all_masses)
    return result
