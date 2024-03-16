# Python imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Model imports ("hack" for relative imports)
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.import_data import load_posterior
import model.model_functions as model

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


def test_func_mode(coefficients, test_data, request):
    idx, row = test_data
    coeffs = coefficients[row["style"]]["mean"]

    expected = row["mode"]
    computed = model.func_mode(coeffs, row["mag"])

    err_msg = f"""
    ***********************
    Test Function: {request.node.name}
    Testing Fails on Row {idx+2}:
    Expected {expected}, got {computed}
    ***********************
    """
    np.testing.assert_allclose(computed, expected, rtol=RTOL, err_msg=err_msg)


def test_func_mu(coefficients, test_data, request):
    idx, row = test_data
    coeffs = coefficients[row["style"]]["mean"]

    expected = row["mu"]
    computed = model.func_mu(coeffs, row["mag"], row["u_star"])

    err_msg = f"""
    ***********************
    Test Function: {request.node.name}
    Testing Fails on Row {idx+2}:
    Expected {expected}, got {computed}
    ***********************
    """
    np.testing.assert_allclose(computed, expected, rtol=RTOL, err_msg=err_msg)


def test_func_sd_mode(coefficients, test_data, request):
    idx, row = test_data
    coeffs = coefficients[row["style"]]["mean"]

    expected = row["sd_mode"]

    # Map applicable function or data column based on style
    sof = row["style"]
    if sof == "strike-slip":
        computed = model.func_sd_mode_bilinear(coeffs, row["mag"])
    elif sof == "reverse":
        computed = coeffs["s_m,r"]
    elif sof == "normal":
        computed = model.func_sd_mode_sigmoid(coeffs, row["mag"])
    else:
        raise ValueError(f"Unrecognized style: {sof}")

    err_msg = f"""
    ***********************
    Test Function: {request.node.name}
    Testing Fails on Row {idx+2}:
    Expected {expected}, got {computed}
    ***********************
    """
    np.testing.assert_allclose(computed, expected, rtol=RTOL, err_msg=err_msg)


def test_func_sd_u(coefficients, test_data, request):
    idx, row = test_data
    coeffs = coefficients[row["style"]]["mean"]

    expected = row["sd_u"]

    # Map applicable function or data column based on style
    sof = row["style"]
    if sof in ["strike-slip", "reverse"]:
        computed = model.func_sd_u(coeffs, row["u_star"])
    elif sof == "normal":
        computed = coeffs["sigma"]
    else:
        raise ValueError(f"Unrecognized style: {sof}")

    err_msg = f"""
    ***********************
    Test Function: {request.node.name}
    Testing Fails on Row {idx+2}:
    Expected {expected}, got {computed}
    ***********************
    """
    np.testing.assert_allclose(computed, expected, rtol=RTOL, err_msg=err_msg)


def test_func_x(coefficients, test_data, request):
    idx, row = test_data
    coeffs = coefficients[row["style"]]["mean"]

    expected = (row["mu"], row["sd_tot"])

    # Map applicable function or data column based on style
    sof = row["style"]
    if sof == "strike-slip":
        computed_arr = model.func_ss(coeffs, row["mag"], row["u_star"])
    elif sof == "reverse":
        computed_arr = model.func_rv(coeffs, row["mag"], row["u_star"])
    elif sof == "normal":
        computed_arr = model.func_nm(coeffs, row["mag"], row["u_star"])
    else:
        raise ValueError(f"Unrecognized style: {sof}")

    computed = (computed_arr[0][0], computed_arr[1][0])

    err_msg = f"""
    ***********************
    Test Function: {request.node.name}
    Testing Fails on Row {idx+2}:
    Expected {expected}, got {computed}
    ***********************
    """
    np.testing.assert_allclose(computed, expected, rtol=RTOL, err_msg=err_msg)
