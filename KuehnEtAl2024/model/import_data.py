# Python imports
from pathlib import Path
import pandas as pd

def load_posterior():
    """
    Load model parameters.

    Parameters
    ----------
    None

    Returns
    -------
    dict
        A dictionary containing dataframes of the loaded model parameters for each style of 
        faulting. The full set of parameters and the mean parameters are provided.
        For example:
        {
            "strike-slip": {
                "mean": pandas.DataFrame,
                "full": pandas.DataFrame
            },
            "reverse": {
                "mean": pandas.DataFrame,
                "full": pandas.DataFrame
            },
            "normal": {
                "mean": pandas.DataFrame,
                "full": pandas.DataFrame
            }
        }
    
    Examples
    -------
    >>> posterior = load_posterior()
    >>> print(posterior["strike-slip"]["mean"])
    >>> print(posterior["normal"]["full"])
    """
    
    # Filepath for model coefficients
    dir_data = Path(__file__).parents[1] / "data"
    
    # Filenames for model coefficients
    filenames = {
        "strike-slip": "coefficients_posterior_SS_powtr.csv",
        "reverse": "coefficients_posterior_REV_powtr.csv",
        "normal": "coefficients_posterior_NM_powtr.csv",
    }

    def load(fname):
        samples = pd.read_csv(dir_data / fname).rename(columns={"Unnamed: 0": "model_number"})

        mean = samples.mean(axis=0).to_frame().transpose()
        mean.loc[0, "model_number"] = -1  # Define model id as -1 for mean coeffs
        mean["model_number"] = mean["model_number"].astype(int)
        return {"mean": mean, "full": samples}

    return {key: load(fname) for key, fname in filenames.items()}
