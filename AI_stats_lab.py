import numpy as np


# -------------------------------------------------
# Question 1: Continuous pair on the unit square
# -------------------------------------------------

def joint_cdf_unit_square(x, y):
    """
    Joint CDF for uniform distribution on unit square
    """

    if x <= 0 or y <= 0:
        return 0
    elif 0 < x < 1 and 0 < y < 1:
        return x * y
    elif 0 < x < 1 and y >= 1:
        return x
    elif x >= 1 and 0 < y < 1:
        return y
    else:  # x >= 1 and y >= 1
        return 1


def rectangle_probability(x1, x2, y1, y2):
    """
    P(x1 < X <= x2, y1 < Y <= y2)
    Using CDF formula
    """

    return (
        joint_cdf_unit_square(x2, y2)
        - joint_cdf_unit_square(x1, y2)
        - joint_cdf_unit_square(x2, y1)
        + joint_cdf_unit_square(x1, y1)
    )


def marginal_fx_unit_square(x):
    """
    Marginal PDF of X
    """

    if 0 < x < 1:
        return 1
    else:
        return 0


def marginal_fy_unit_square(y):
    """
    Marginal PDF of Y
    """

    if 0 < y < 1:
        return 1
    else:
        return 0


# -------------------------------------------------
# Question 2: Joint PMF, marginals, independence
# -------------------------------------------------

def joint_pmf_heads(x, y):
    """
    Joint PMF table
    """

    if x == 0 and y == 0:
        return 1/4
    elif x == 0 and y == 1:
        return 1/4
    elif x == 0 and y == 2:
        return 0
    elif x == 1 and y == 0:
        return 0
    elif x == 1 and y == 1:
        return 1/4
    elif x == 1 and y == 2:
        return 1/4
    else:
        return 0


def marginal_px_heads(x):
    """
    P_X(x) = sum over y
    """

    return (
        joint_pmf_heads(x, 0)
        + joint_pmf_heads(x, 1)
        + joint_pmf_heads(x, 2)
    )


def marginal_py_heads(y):
    """
    P_Y(y) = sum over x
    """

    return (
        joint_pmf_heads(0, y)
        + joint_pmf_heads(1, y)
    )


def check_independence_heads():
    """
    Check if X and Y are independent
    """

    for x in [0, 1]:
        for y in [0, 1, 2]:
            joint = joint_pmf_heads(x, y)
            px = marginal_px_heads(x)
            py = marginal_py_heads(y)

            if not np.isclose(joint, px * py):
                return False

    return True
