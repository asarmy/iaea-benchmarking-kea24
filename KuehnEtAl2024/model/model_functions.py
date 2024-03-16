""" """

# Python imports
from scipy import stats
import numpy as np
import pandas as pd


# Model constants
MAG_BREAK, DELTA = 7.0, 0.1


# Check inputs
def check_numeric_type(arg):
    """ Check for single values because core model functions are not vectorized yet."""
    if not isinstance(arg, (int, float)):
        raise TypeError("Argument must be an int or float; in other words, only one value is allowed.")


# Model formulas
def func_mode(coefficients, magnitude):
    """
    Calculate magnitude scaling in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    Returns
    -------
    fm : np.array
        Mode in transformed units.
    """
    
    check_numeric_type(magnitude)
    
    fm = (
        coefficients["c1"]
        + coefficients["c2"] * (magnitude - MAG_BREAK)
        + (coefficients["c3"] - coefficients["c2"])
        * DELTA
        * np.log(1 + np.exp((magnitude - MAG_BREAK) / DELTA))
    )
    return fm


def func_mu(coefficients, magnitude, location):
    """
    Calculate mean prediction in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    location : np.array
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    mu : float
        Mean prediction in transformed units.
    """
    
    check_numeric_type(magnitude)
    check_numeric_type(location)

    fm = func_mode(coefficients, magnitude=magnitude)

    alpha = coefficients["alpha"]
    beta = coefficients["beta"]
    gamma = coefficients["gamma"]

    a = fm - gamma * np.power(alpha / (alpha + beta), alpha) * np.power(
        beta / (alpha + beta), beta
    )

    mu = a + gamma * np.power(location, alpha) * np.power(1 - location, beta)
    return np.asarray(mu)


def func_sd_mode_bilinear(coefficients, magnitude):
    """
    Calculate standard deviation of the mode in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    Returns
    -------
    sd: np.array
        Standard deviation of the mode in transformed units.

    Notes
    ------
    Bilinear standard deviation model is only used for strike-slip faulting.
    """
    
    check_numeric_type(magnitude)

    sd = (
        coefficients["s_m,s1"]
        + coefficients["s_m,s2"] * (magnitude - coefficients["s_m,s3"])
        - coefficients["s_m,s2"]
        * DELTA
        * np.log(1 + np.exp((magnitude - coefficients["s_m,s3"]) / DELTA))
    )
    return np.asarray(sd)


def func_sd_mode_sigmoid(coefficients, magnitude):
    """
    Calculate standard deviation of the mode in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    Returns
    -------
    sd: np.array
        Standard deviation of the mode in transformed units.

    Notes
    ------
    Sigmoidal standard deviation model is only used for normal faulting.
    """
    
    check_numeric_type(magnitude)

    sd = coefficients["s_m,n1"] - coefficients["s_m,n2"] / (
        1 + np.exp(-1 * coefficients["s_m,n3"] * (magnitude - MAG_BREAK))
    )
    return np.asarray(sd)


def func_sd_u(coefficients, location):
    """
    Calculate standard deviation of the location in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    sd : np.array
        Standard deviation of the location in transformed units.

    Notes
    ------
    Used only for strike-slip and reverse faulting.
    """
    
    check_numeric_type(location)

    # Column name2 for stdv coefficients "s_" varies for style of faulting, fix that here
    if isinstance(coefficients, pd.DataFrame):
        s_1 = coefficients["s_s1"] if "s_s1" in coefficients.columns else coefficients["s_r1"]
        s_2 = coefficients["s_s2"] if "s_s2" in coefficients.columns else coefficients["s_r2"]
    elif isinstance(coefficients, np.recarray):
        s_1 = coefficients["s_s1"] if "s_s1" in coefficients.dtype.names else coefficients["s_r1"]
        s_2 = coefficients["s_s2"] if "s_s2" in coefficients.dtype.names else coefficients["s_r2"]
    else:
        raise TypeError(
            "Function argument for model coefficients must be pandas DataFrame or numpy recarray."
        )

    alpha = coefficients["alpha"]
    beta = coefficients["beta"]

    sd = s_1 + s_2 * np.power(location - alpha / (alpha + beta), 2)
    return np.asarray(sd)


def func_ss(coefficients, magnitude, location):
    """
    Calculate mean prediction and standard deviations (both in transformed units) for strike-slip
    faulting.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[np.array, np.array]
        mu : Mean prediction in transformed units.
        sd_total : Total standard deviation in transformed units.
    """

    # Calculate mean prediction
    mu = func_mu(coefficients, magnitude, location)

    # Calculate standard deviations
    sd_mode = func_sd_mode_bilinear(coefficients, magnitude)
    sd_u = func_sd_u(coefficients, location)
    sd_total = np.sqrt(np.power(sd_mode, 2) + np.power(sd_u, 2))

    return mu, sd_total


def func_nm(coefficients, magnitude, location):
    """
    Calculate mean prediction and standard deviations (both in transformed units) for normal
    faulting.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[np.array, np.array]
        mu : Mean prediction in transformed units.
        sd_total : Total standard deviation in transformed units.
    """

    # Calculate mean prediction
    mu = func_mu(coefficients, magnitude, location)

    # Calculate standard deviations
    sd_mode = func_sd_mode_sigmoid(coefficients, magnitude)
    sd_u = np.asarray(coefficients["sigma"])
    sd_total = np.sqrt(np.power(sd_mode, 2) + np.power(sd_u, 2))

    return mu, sd_total


def func_rv(coefficients, magnitude, location):
    """
    Calculate mean prediction and standard deviations (both in transformed units) for reverse
    faulting.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[np.array, np.array]
        mu : Mean prediction in transformed units.
        sd_total : Total standard deviation in transformed units.
    """

    # Calculate mean prediction
    mu = func_mu(coefficients, magnitude, location)

    # Calculate standard deviations
    sd_mode = np.asarray(coefficients["s_m,r"])
    sd_u = func_sd_u(coefficients, location)
    sd_total = np.sqrt(np.power(sd_mode, 2) + np.power(sd_u, 2))

    return mu, sd_total
