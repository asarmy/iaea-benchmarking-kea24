# Python imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Model imports ("hack" for relative imports)
sys.path.append(str(Path(__file__).resolve().parents[1]))
from model.import_data import load_posterior


def test_load_posterior():
    posterior = load_posterior()

    # Check if the return value is a dictionary
    assert isinstance(posterior, dict)

    # Check error is raised
    with pytest.raises(AssertionError):
        assert isinstance(posterior, int)

    # Check if all required faulting styles are present
    assert "strike-slip" in posterior
    assert "reverse" in posterior
    assert "normal" in posterior

    # Check the "mean" and "full" options are
    for style in posterior.values():
        assert "mean" in style
        assert "full" in style

    for style in posterior.values():
        assert isinstance(style["mean"], pd.DataFrame)
        assert isinstance(style["full"], pd.DataFrame)
