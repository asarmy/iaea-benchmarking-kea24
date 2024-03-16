# Import python libraries
import numpy as np
import pandas as pd

# Import configurations
from model_config import *

# Import package functions
from model.import_data import load_posterior
from model.helper_functions import calc_distrib_params

# Load posterior distributions
POSTERIOR = load_posterior()

def calc_params(row: pd.Series, style: str, mean_model_flag: bool) -> tuple:
    """
    Calculate distribution parameters based on row values.

    Parameters
    ----------
    row : pd.Series
        A pandas Series representing a row of a DataFrame.
    style : str
        Style of faulting.
    mean_model_flag : bool
        Flag indicating whether to use the mean model (True) or full model with
        1000 runs (False).

    Returns
    -------
    tuple
        A tuple of the calculated mean, total sigma, and lambda parameter. 
        Not that the mean and sigma are in Box-Cox transform units.

    """

    # Define function variable and apply function to each row of dataframe
    f = calc_distrib_params
    result = f(
        magnitude=row["magnitude"],
    location=row["u_star"],
    style=style,
    posterior=POSTERIOR,
    mean_model=mean_model_flag,
    )
    
    return tuple(np.squeeze(arr) for arr in result[:3])
    

def calc_model_predictions(
    dataframe: pd.DataFrame, style: str, mean_model_flag: bool
) -> pd.DataFrame:
    """
    Calculate model predictions for the input DataFrame. The column names are specific
    to this project and assumed to be correct.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The input DataFrame containing the data.
    style : str
        Style of faulting.
    mean_model_flag : bool
        Flag indicating whether to use the mean model (True) or full model with
        1000 runs (False).

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the input data and model predictions.

    """

    # Calculuate mu, sigma (in transformed units)for each row and number of model runs
    dataframe = dataframe.copy()
    mu, sig, bc_param = "mu", "sigma", "lambda"
    dataframe[[mu, sig, bc_param]] = dataframe.apply(
        lambda row: pd.Series(calc_params(row, style, mean_model_flag)), axis=1
    )

    # Additional processing based on number of model runs and weights
    if mean_model_flag:
        dataframe["MODEL_ID"] = 1
        dataframe["fdm_wt"] = 1
    else:
        # Enumerate MODEL_IDs
        dataframe["MODEL_ID"] = dataframe[mu].apply(
            lambda x: list(range(1, len(x) + 1))
        )
        # Calcualte weights for each MODEL_ID
        n_runs = dataframe[mu].apply(lambda x: len(x))
        dataframe["fdm_wt"] = pd.Series([[1 / i] * i for i in n_runs])
        dataframe = dataframe.explode(
            [mu, sig, bc_param, "MODEL_ID", "fdm_wt"], ignore_index=True
        )

    return dataframe
