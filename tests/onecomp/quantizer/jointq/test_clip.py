"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import torch

from onecomp.quantizer.jointq.core.clip import clip


def test_clip_symmetric():
    """Test the clip function."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix_W = torch.tensor(
        [
            [1.2, 0.85, -0.9, -0.58, 0.38, 0.18],
            [10.1, -11.8, 12.0, 3.5, -2.5, -1.5],
        ],
        dtype=torch.float32,
    ).to(device)

    scales, assignment, zero_point = clip(
        matrix_W,
        group_size=3,
        symmetric=True,
        lower_bound=-8,
        upper_bound=7,
    )
    print(scales)
    print(assignment)
    print(zero_point)

    assert scales.shape == (2, 2)
    assert assignment.shape == (2, 2, 3)
    assert zero_point.shape == (2, 2)
    assert abs(float(scales[0, 0]) - 1.2 / 7) < 1e-6
    assert abs(float(scales[0, 1]) - 0.58 / 7) < 1e-6
    assert abs(float(scales[1, 0]) - 12.0 / 7) < 1e-6
    assert abs(float(scales[1, 1]) - 3.5 / 7) < 1e-6
    assert abs(int(zero_point[0, 0]) - 0) < 1e-6
    assert abs(int(zero_point[0, 1]) - 0) < 1e-6
    assert abs(int(zero_point[1, 0]) - 0) < 1e-6
    assert abs(int(zero_point[1, 1]) - 0) < 1e-6

    # w_dequantized[i, t, j] = scales[i, t] * assignment[i, t, j]
    w_dequantized = scales.unsqueeze(2) * assignment
    print(w_dequantized)


def test_clip_asymmetric():
    """Test the clip function."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    matrix_W = torch.tensor(
        [
            [1.2, 0.85, -0.9, -0.58, 0.38, 0.18],
            [10.1, -11.8, 12.0, 3.5, -2.5, -1.5],
        ],
        dtype=torch.float32,
    ).to(device)

    scales, assignment, zero_point = clip(
        matrix_W,
        group_size=3,
        symmetric=False,
        lower_bound=0,
        upper_bound=15,
    )
    print(scales)
    print(assignment)
    print(zero_point)

    assert scales.shape == (2, 2)
    assert assignment.shape == (2, 2, 3)
    assert zero_point.shape == (2, 2)
    assert abs(float(scales[0, 0]) - (1.2 + 0.9) / 15) < 1e-6
    assert abs(float(scales[0, 1]) - (0.38 + 0.58) / 15) < 1e-6
    assert abs(float(scales[1, 0]) - (12.0 + 11.8) / 15) < 1e-6
    assert abs(float(scales[1, 1]) - (3.5 + 2.5) / 15) < 1e-6
    assert abs(int(zero_point[0, 0]) - round(0.9 / ((1.2 + 0.9) / 15))) < 1e-6
    assert abs(int(zero_point[0, 1]) - round(0.58 / ((0.38 + 0.58) / 15))) < 1e-6
    assert abs(int(zero_point[1, 0]) - round(11.8 / ((12.0 + 11.8) / 15))) < 1e-6
    assert abs(int(zero_point[1, 1]) - round(2.5 / ((3.5 + 2.5) / 15))) < 1e-6

    # w_dequantized[i, t, j] = scales[i, t] * (assignment[i, t, j] - zero_point[i, t])
    w_dequantized = scales.unsqueeze(2) * (assignment - zero_point.unsqueeze(2))
    print(w_dequantized)

    # (2, 2, 3) -> (2, 6)
    w_dequantized = w_dequantized.view(2, 6)
    print(w_dequantized)

    # Generate random data
    torch.manual_seed(0)
    matrix_X = torch.rand(10, 6).to(torch.float32).to(device)

    print(torch.sum((matrix_W @ matrix_X.T - w_dequantized @ matrix_X.T) ** 2))
