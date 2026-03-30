"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

from unittest import TestCase

import torch

from onecomp.quantizer.jointq.core.solution import Solution


class TestSolution(TestCase):
    """Test the Solution class."""

    def test_init(self):
        """Test the initialization of the Solution class."""

        scales = torch.tensor([[0.14, 0.064], [1.5867, 0.4]])
        assignment = torch.tensor([[[15, 12, 0], [0, 15, 12]], [[13, 0, 15], [15, 0, 2]]])
        zero_point = torch.tensor([[6, 9], [7, 6]])

        w_dequantized = scales.unsqueeze(2) * (assignment - zero_point.unsqueeze(2))
        print(w_dequantized)

        solution = Solution(scales, assignment, zero_point)
        self.assertTrue(torch.allclose(solution.scales, scales.T))
        self.assertTrue(torch.equal(solution.zero_point, zero_point.T))
        self.assertTrue(
            torch.equal(
                solution.integers_z,
                torch.tensor([[[9, 6, -6], [6, -7, 8]], [[-9, 6, 3], [9, -6, -4]]]),
            )
        )
