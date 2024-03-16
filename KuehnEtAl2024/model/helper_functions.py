# Python imports
import numpy as np
import pandas as pd

# Import package modules
import model.model_functions as model

def calc_distrib_params(
    *,
    magnitude: float,
    location: float,
    style: str,
    posterior: dict,
    mean_model: bool = True,
):
    """
    Calculate median and sigma values for KEA22 on magnitude, rupture location, and style.
    Note all returns are in natural log units.
    Note returned values are asymmetrical (i.e., not folded).

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude.
    location : float
        Normalized location along rupture length, range [0, 1.0].
    style : str
        Style of faulting, case insensitive.
        Valid options are "strike-slip", "reverse", or "normal".
    posterior : dict
        A dictionary containing dataframes of the loaded model parameters for each style of faulting.
        For example, `posterior["strike-slip"]["mean"]` or `posterior["reverse"]["full"]`
    mean_model : bool, optional
        If True, use mean coefficients and adjustments.
        If False, use full (n=1000) coefficients and adjustments.
        Default True.

    Returns
    -------
    Tuple[np.array, np.array, np.array]
        mu : Mean prediction in transformed units.
        sd_total : Total standard deviation in transformed units.
        bc_lambda : "lambda" transformation parameter in Box-Cox transformation.
    """

    # Get appropriate coefficients 
    flag = "mean" if mean_model else "full"
    style = style.lower()
    if style not in ["strike-slip", "reverse", "normal"]:
        raise ValueError(f"Invalid style {style} was provided.")
    coefficients = posterior[style][flag]
        
    # Get appropriate model
    model_map = {
    "strike-slip": model.func_ss,
    "reverse": model.func_rv,
    "normal": model.func_nm,
    }
    params_function = model_map.get(style)
    
    # Compute distribution parameters
    mu, sigma = params_function(coefficients, magnitude, location)
    bc_lambda = np.asarray(coefficients["lambda"])
    
    # Return distribution and transformation parameters
    return mu, sigma, bc_lambda
