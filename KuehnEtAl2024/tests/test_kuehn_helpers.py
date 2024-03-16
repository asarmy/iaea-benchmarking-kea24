# Python imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Model imports ("hack" for relative imports)
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.import_data import load_posterior
import model.helper_functions as helpers

# Test setup
RTOL = 1e-2
EXPECTED = Path(__file__).parent / "expected" / "kea_function_results.csv"

def pytest_generate_tests(metafunc):
    if "test_data" in metafunc.fixturenames:
        # Load the CSV file, determine number of rows
        data = pd.read_csv(EXPECTED)
        num_rows = len(data)

        # Generate a list of indices for parameterizing tests
        metafunc.parametrize("test_data", range(num_rows), indirect=True)


@pytest.fixture(scope="session")
def coefficients():
    return load_posterior()


@pytest.fixture(scope="module")
def expected_data():
    return pd.read_csv(EXPECTED)


@pytest.fixture(scope="function")
def test_data(expected_data, request):
    idx = request.param
    row = expected_data.iloc[idx]
    return idx, row


def test_calc_distrib_params(coefficients, test_data, request):
    idx, row = test_data

    expected = (row["mu"], row["sd_tot"], row["lambda"])

    computed_arr = helpers.calc_distrib_params(
        magnitude=row["mag"],
        location=row["u_star"],
        style=row["style"],
        posterior=coefficients,
        mean_model=True,
    )
    
    computed = (computed_arr[0][0], computed_arr[1][0], computed_arr[2][0])

    err_msg = f"""
    ***********************
    Test Function: {request.node.name}
    Testing Fails on Row {idx+2}:
    Expected {expected}, got {computed}
    ***********************
    """
    np.testing.assert_allclose(computed, expected, rtol=RTOL, err_msg=err_msg)
