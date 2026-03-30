"""Regression test: matrix_X path vs matrix_XX path.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Regression test to verify that the new mode (passing precomputed matrix_XX) returns
the same quantization results as the old mode (passing matrix_X directly).

Test Contents
-------------
test_quantize_matrix_xx_vs_matrix_x:
    Run the old mode (matrix_X) and the new mode (matrix_XX) with the same data
    and parameters, verifying the following:

    - integers_z (integer tensor): exact match via torch.equal
    - scales (float64 tensor): approximate match via torch.allclose (atol=1e-12)
    - zero_point (integer tensor): exact match via torch.equal
    - mse (scalar): approximate match via math.isclose (rel_tol=1e-10)

    Skipped if CUDA is unavailable or input data is not present.
"""

import math

import pytest
import torch

from .regression_quantize_matrix_xx_helper import (
    DEFAULT_DATA_PATH,
    run_quantize_with_matrix_x,
    run_quantize_with_matrix_xx,
)

_SKIP_NO_CUDA = not torch.cuda.is_available()
_SKIP_NO_DATA = not DEFAULT_DATA_PATH.exists()


@pytest.mark.skipif(_SKIP_NO_CUDA, reason="CUDA is not available")
@pytest.mark.skipif(_SKIP_NO_DATA, reason=f"Data file not found: {DEFAULT_DATA_PATH}")
def test_quantize_matrix_xx_vs_matrix_x():
    """Test that quantize with matrix_XX produces the same result as with matrix_X."""

    result_x = run_quantize_with_matrix_x(DEFAULT_DATA_PATH, device_id=0)
    result_xx = run_quantize_with_matrix_xx(DEFAULT_DATA_PATH, device_id=0)

    # integers_z (int tensor): exact match
    assert torch.equal(
        result_xx["integers_z"], result_x["integers_z"]
    ), "integers_z mismatch between matrix_X and matrix_XX paths"

    # scales (float64 tensor): near-exact match
    assert torch.allclose(result_xx["scales"], result_x["scales"], atol=1e-12), (
        f"scales mismatch: max diff = " f"{(result_xx['scales'] - result_x['scales']).abs().max()}"
    )

    # zero_point: exact match (integer type)
    assert torch.equal(
        result_xx["zero_point"], result_x["zero_point"]
    ), "zero_point mismatch between matrix_X and matrix_XX paths"

    # MSE (scalar): relative tolerance
    assert math.isclose(
        result_xx["mse"], result_x["mse"], rel_tol=1e-10
    ), f"MSE mismatch: matrix_X={result_x['mse']}, matrix_XX={result_xx['mse']}"
