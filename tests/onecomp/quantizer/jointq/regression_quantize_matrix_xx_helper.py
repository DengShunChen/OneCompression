"""Regression test helper for quantize: matrix_X path vs matrix_XX path.

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

Helper module to verify that the new mode (passing precomputed matrix_XX) returns
the same results as the old mode (passing matrix_X directly).

Functions
---------
run_quantize_with_matrix_x(data_path, device_id=0)
    Old mode: Pass matrix_X directly to quantize for quantization.

run_quantize_with_matrix_xx(data_path, device_id=0)
    New mode: Precompute X^T X via compute_matrix_XX, then
    run quantize(matrix_XX=..., dim_n=...) for quantization.

Both functions return the same format (scales, zero_point, integers_z, mse).
"""

from pathlib import Path

import torch

from onecomp.quantizer.jointq.core import compute_matrix_XX, quantize

DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_DATA_PATH = DATA_DIR / "model_layers_0_self_attn_k_proj.pth"


def _load_data(data_path):
    """Load weight and activation data from a .pth file.

    Returns
    -------
    matrix_W : torch.Tensor
        Weight matrix, shape (p, m), dtype float64, on CPU.
    matrix_X : torch.Tensor
        Input matrix, shape (n, m), dtype float64, on CPU.
    """
    weight_and_activation = torch.load(data_path, weights_only=True)

    matrix_W = weight_and_activation["W"].to("cpu").to(torch.float64)
    matrix_X = weight_and_activation["X"].to("cpu").to(torch.float64)

    if matrix_X.ndim == 3:
        matrix_X = matrix_X.reshape(-1, matrix_X.shape[-1])
    elif matrix_X.ndim != 2:
        raise ValueError(f"Unsupported matrix_X shape: {matrix_X.shape}")

    return matrix_W, matrix_X


def _compute_mse(matrix_W, matrix_X, solution, device):
    """Compute MSE: mean(||WX^T - hat_W X^T||^2)."""
    matrix_W_hat = solution.get_dequantized_weight_matrix()
    matrix_W_gpu = matrix_W.to(device)
    matrix_X_gpu = matrix_X.to(device)
    return float(torch.mean((matrix_W_gpu @ matrix_X_gpu.T - matrix_W_hat @ matrix_X_gpu.T) ** 2))


def _pack_result(solution, mse):
    """Pack a Solution object and MSE into a result dict."""
    return {
        "scales": solution.scales.cpu(),
        "zero_point": solution.zero_point.cpu(),
        "integers_z": solution.integers_z.cpu(),
        "mse": mse,
    }


def run_quantize_with_matrix_x(data_path, device_id=0):
    """Old mode: Pass matrix_X directly for quantization.

    Parameters
    ----------
    data_path : str or Path
        Path to the .pth file containing weight and activation data.
    device_id : int
        GPU device ID.

    Returns
    -------
    dict
        Dictionary with keys: scales, zero_point, integers_z (Tensors on CPU),
        and mse (float).
    """
    matrix_W, matrix_X = _load_data(data_path)
    device = torch.device(device_id)

    solution = quantize(
        matrix_W=matrix_W,
        matrix_X=matrix_X,
        bits=4,
        symmetric=False,
        group_size=128,
        batch_size=2048,
        device=device,
        log_level=2,
    )

    mse = _compute_mse(matrix_W, matrix_X, solution, device)
    return _pack_result(solution, mse)


def run_quantize_with_matrix_xx(data_path, device_id=0):
    """New mode: Precompute X^T X via compute_matrix_XX and run quantization.

    Parameters
    ----------
    data_path : str or Path
        Path to the .pth file containing weight and activation data.
    device_id : int
        GPU device ID.

    Returns
    -------
    dict
        Dictionary with keys: scales, zero_point, integers_z (Tensors on CPU),
        and mse (float).
    """
    matrix_W, matrix_X = _load_data(data_path)
    device = torch.device(device_id)

    # Precompute matrix_XX
    dim_n = matrix_X.shape[0]
    matrix_XX = compute_matrix_XX(matrix_X, device)

    solution = quantize(
        matrix_W=matrix_W,
        matrix_XX=matrix_XX,
        dim_n=dim_n,
        bits=4,
        symmetric=False,
        group_size=128,
        batch_size=2048,
        device=device,
        log_level=2,
    )

    mse = _compute_mse(matrix_W, matrix_X, solution, device)
    return _pack_result(solution, mse)
