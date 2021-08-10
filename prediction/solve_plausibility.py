from scipy.optimize import bisect

"""
This file calculated the plausibility as described in "Aleatoric and Epistemic Uncertainty with Random Forests". We use
scipy.optimize.bisect to find the correct alpha value (theta in the paper). Here, this is done by looking at both terms
within the scope of the min function for the pi calculation. We know that if we look at the intersection of those two
terms within a specific interval (between the peak of both terms), this will result in the optimal alpha value.

Some methods contain verbose parameter which are false by default and it is encouraged to leave it at that. This
parameter is only useful if you really want to understand the calculation of single value pairs for p and n.

Works for until (p=537, n=537) or (p=331, n=996) or similar sized values. It does not work for bigger values because of
(p/(p+n))**p * (n/(p+n))**n which will be 0 (because the actual solution would be too small). In order to deal with this
problem, many methods reduce the input values proportionally so that this does not happen.
"""


def calc_alpha(x, *args):
    """
    Function, which calculates both values within the minimum of the pi calculation and subtracts them so that the
    result of this function would be 0 at their intersection. This intersection corresponds to the point with the
    optimal alpha value.
    """
    p, n, plus = args
    if plus:
        return ((x**p*(1-x)**n)/((p/(p+n))**p*(n/(p+n))**n)) - 2*x + 1
    else:
        return ((x**p*(1-x)**n)/((p/(p+n))**p*(n/(p+n))**n)) + 2*x - 1


def check_result(p, n, alpha, plus, verbose=False):
    """
    Check if the result of left_term and right_term are actually the same. If they are not, the results do not
    necessarily need to be incorrect because rounding errors might be the reason. Terms are printed to compare the
    values and see, whether they are approximately equal. This function was mainly used to test the implementation, now
    it us not used because slight numerical differences result in a wrong check.
    """
    p = float(p)
    n = float(n)
    p_part = (p/(p+n))**p
    n_part = (n/(p+n))**n
    c = p_part*n_part
    if verbose:
        print("Using the calculated x (alpha) intersection value, both original terms should result in the same "
              "function value.")
        print()
    numerator = (alpha**p)*((1-alpha)**n)
    left_term = numerator / c
    if verbose:
        print("Result of the left_term (x**p)*((1-x)**n)/c:")
        print(left_term)
        print()
    if plus:
        right_term = 2*alpha-1
    else:
        right_term = 1-2*alpha
    if verbose:
        print("Result of the right_term 2*x-1:")
        print(right_term)
        print()
    if not(left_term == right_term):
        print("left_term and right_term are not exactly equal, but the difference might only be a small numerical "
              "error, so here are both results:")
        print((left_term, right_term))
        print("It should be visible, whether the results are almost equal or entirely different (which would mean the "
              "calculation is wrong)")
        print()


def calc_pi(p, n, plus, check_res=False, verbose=False):
    """
    Calculate the pi_plus or pi_minus value in analytical way.
    """
    if p + n == 0:
        return 1
    if plus:
        a = float(p)/float(p+n)
        b = 1
    else:
        a = 0
        b = float(p)/float(p+n)
    # You can show graphically, that the optimal alpha value is within a and b
    # Therefore, look at both terms within the minimum of the pi calculation
    alpha = bisect(calc_alpha, a, b, args=(p, n, plus))
    if verbose:
        print("Numerically calculated alpha value:")
        print(alpha)
        print()
    if check_res:
        check_result(p, n, alpha, plus)
    if verbose:
        print("Using the result of the easier to calculate term 2*x-1 as the final result:")
        print(2*alpha-1)
    if plus:
        return 2*alpha-1
    else:
        return 1-2*alpha


def epistemic_unc(pi_plus, pi_minus):
    """
    Calculates the epistemic uncertainty.
    """
    return min(pi_plus, pi_minus)


def aleatoric_unc(pi_plus, pi_minus):
    """
    Calculates the aleatoric uncertainty.
    """
    return 1-max(pi_plus, pi_minus)


def uncertainty(e_u, a_u):
    """
    Calculates the summed up uncertainty.
    """
    return e_u + a_u


def calc_p_plus(pi_plus, pi_minus, u):
    """
    Calculates the p-plus value. Can be seen as the overall plausibility of the positive class.
    """
    if pi_plus > pi_minus:
        return 1 - u
    elif pi_plus == pi_minus:
        return (1-u)/2
    else:
        return 0


def calc_p_minus(p_plus, u):
    """
    Calculates the p-plus value. Can be seen as the overall plausibility of the negative class.
    """
    return 1 - (p_plus + u)


def test_result(p_plus, p_minus, e_u, a_u):
    """
    Checks, whether the assumption that the sum of plausibilities and uncertainty equals one, holds.
    """
    return p_plus + p_minus + e_u + a_u == 1.0


def calculate_plausibility_values(p, n):
    """
    Calculate and return any relevant results and intermediate result of this approach.
    """
    if max(p, n) > 537:
        temp = max(p, n)
        p = float(round(p/(temp/537)))
        n = float(round(n/(temp/537)))
    pi_plus = calc_pi(p, n, True)
    pi_minus = calc_pi(p, n, False)
    e_u = epistemic_unc(pi_plus, pi_minus)
    a_u = aleatoric_unc(pi_plus, pi_minus)
    u = uncertainty(e_u, a_u)
    p_plus = calc_p_plus(pi_plus, pi_minus, u)
    p_minus = calc_p_minus(p_plus, u)
    if not(test_result(p_plus, p_minus, e_u, a_u)):
        print("Something went wrong, check implementation")
        print(p, n)
    return pi_plus, pi_minus, e_u, a_u, u, p_plus, p_minus


def calculate_plausibilities(p, n):
    """
    Return the plus and minus probabilities.
    """
    if max(p, n) > 537:
        temp = max(p, n)
        p = float(round(p/(temp/537)))
        n = float(round(n/(temp/537)))
    pi_plus = calc_pi(p, n, True)
    pi_minus = calc_pi(p, n, False)
    e_u = epistemic_unc(pi_plus, pi_minus)
    a_u = aleatoric_unc(pi_plus, pi_minus)
    u = uncertainty(e_u, a_u)
    p_plus = calc_p_plus(pi_plus, pi_minus, u)
    p_minus = calc_p_minus(p_plus, u)
    if not(test_result(p_plus, p_minus, e_u, a_u)):
        print("Something went wrong, check implementation")
        print(p, n)
    return p_plus, p_minus


def calculate_plausibility(p, n, plus):
    """
    Return the plus or minus probability.
    """
    if max(p, n) > 537:
        temp = max(p, n)
        p = float(round(p/(temp/537)))
        n = float(round(n/(temp/537)))
    pi_plus = calc_pi(p, n, True)
    pi_minus = calc_pi(p, n, False)
    e_u = epistemic_unc(pi_plus, pi_minus)
    a_u = aleatoric_unc(pi_plus, pi_minus)
    u = uncertainty(e_u, a_u)
    p_plus = calc_p_plus(pi_plus, pi_minus, u)
    p_minus = calc_p_minus(p_plus, u)
    if not(test_result(p_plus, p_minus, e_u, a_u)):
        print("Something went wrong, check implementation")
        print(p, n)
    if plus:
        return p_plus
    else:
        return p_minus
