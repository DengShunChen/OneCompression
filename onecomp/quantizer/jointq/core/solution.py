"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import torch


# pylint: disable=too-many-arguments,too-many-positional-arguments
class Solution:
    """Solution class."""

    def __init__(self, scales=None, assignment=None, zero_point=None):
        """Initialize the solution.

        Key design points:
        - Asymmetric quantization has the form w ≈ s (a - b), but since
          the zero point is not modified, we store z = a - b.
          This allows handling both asymmetric and symmetric quantization with the same code.
        - Scale coefficient storage:
          - scales[t, i]: scale coefficient for the t-th group of the i-th row of the weight matrix
          - shape: (num_groups, p)
          - Note: stored as the transpose of input
        - Zero point storage:
          - zero_point[t, i]: zero point for the t-th group of the i-th row of the weight matrix
          - shape: (num_groups, p)
          - Note: stored as the transpose of input
        - Integer assignment storage:
          - assignment[t, i, j]: j-th integer assignment for the t-th group of the i-th row
          - shape: (num_groups, p, group_size)
          - Note: stored as the transpose of input

        Parameters
        ----------
        scales: torch.Tensor
            The scales of the quantization, shape (p, num_groups).
            Stored as transpose.
        assignment: torch.Tensor
            The assignment of the quantization, shape (p, num_groups, group_size).
            Zero point is subtracted and then stored as transpose.
        zero_point: torch.Tensor
            The zero point of the quantization, shape (p, num_groups).
            Stored as transpose.

        """
        if scales is not None:
            self.scales = scales.T.contiguous()
        else:
            self.scales = None
        if zero_point is not None:
            self.zero_point = zero_point.T.contiguous()
        else:
            self.zero_point = None
        if assignment is not None:
            self.integers_z = (assignment - zero_point.unsqueeze(2)).transpose(0, 1).contiguous()
        else:
            self.integers_z = None

        self.squared_error = None
        self.squared_errors = None
        self.mean_squared_error = None
        self.mean_squared_errors = None

    def __repr__(self):
        """Return the representation of the solution."""
        return (
            f"scales: {self.scales}\n"
            f"zero_point: {self.zero_point}\n"
            f"integers_z: {self.integers_z}\n"
            f"squared_error: {self.squared_error}\n"
            f"mean_squared_error: {self.mean_squared_error}"
        )

    def clean(self):
        """Clean up the solution to free memory."""

        if self.mean_squared_errors is not None:
            del self.mean_squared_errors
            self.mean_squared_errors = None

        if self.squared_errors is not None:
            del self.squared_errors
            self.squared_errors = None

    def compute_objective_value(
        self, matrix_XX, matrix_YX, Y_sq_norms, dim_n, dequantized_weight_matrix=None
    ):
        """Compute the objective value of the solution.

        Compute the objective function value using the dequantized weight matrix hat_W,
        and store the results in instance attributes.

        Uses precomputed matrix_XX (= X^T X), matrix_YX (= Y @ X),
        and Y_sq_norms (= ||Y[i]||^2) to compute without directly referencing
        matrix_X or target_matrix. Computational cost is O(pm^2) (previously O(pmn)).

        Computation details:
        - hat_W: dequantized weight matrix, shape (p, m)
        - squared_errors[i] = ||Y[i]||^2 - 2 (YX)[i] · hat_W[i] + hat_W[i] (X^TX) hat_W[i]^T
        - squared_error = sum(squared_errors) = ||Y - hat_W X^T||_F^2
        - mean_squared_errors[i] = squared_errors[i] / n,  shape: (p,)
        - mean_squared_error = mean(mean_squared_errors) = squared_error / (p * n)

        Parameters
        ----------
        matrix_XX : torch.Tensor
            X^T X, shape (m, m), on device.
        matrix_YX : torch.Tensor
            Y @ X, shape (p, m), on device.
        Y_sq_norms : torch.Tensor
            Squared norms of each row of the target matrix, shape (p,), on device.
        dim_n : int
            Number of rows in the input matrix n. Used for computing mean_squared_errors.
        dequantized_weight_matrix : torch.Tensor, optional
            Dequantized weight matrix, shape (p, m).
            If None, computed internally.

        """

        if dequantized_weight_matrix is None:
            hat_W = self.get_dequantized_weight_matrix()
        else:
            hat_W = dequantized_weight_matrix

        # squared_errors[i] = ||Y[i]||^2 - 2 (YX * W).sum(dim=1) + ((W @ XtX) * W).sum(dim=1)
        W_XX = hat_W @ matrix_XX  # (p, m), cost O(pm²)
        self.squared_errors = (
            Y_sq_norms - 2.0 * torch.sum(matrix_YX * hat_W, dim=1) + torch.sum(W_XX * hat_W, dim=1)
        )
        del W_XX

        self.squared_error = torch.sum(self.squared_errors)
        self.mean_squared_errors = self.squared_errors / dim_n
        self.mean_squared_error = self.mean_squared_errors.mean()

    def try_update_scales(
        self,
        new_scales,
        matrix_XX,
        matrix_YX,
        Y_sq_norms,
        dim_n,
        epsilon,
        update_flags=None,
    ):
        """Try to update scales with a new solution.

        If update_flags is None, all rows are targeted.
        If update_flags is specified, only rows with True are targeted for optimization.

        Parameters
        ----------
        new_scales: torch.Tensor
            The new scales to update.
            When update_flags=None: shape (p, num_groups).
            When update_flags is specified: shape (sum(update_flags), num_groups).
        matrix_XX : torch.Tensor
            X^T X, shape (m, m), on device.
        matrix_YX : torch.Tensor
            Y @ X, shape (p, m), on device.
        Y_sq_norms : torch.Tensor
            Squared norms of each row of the target matrix, shape (p,), on device.
        dim_n : int
            Number of rows in the input matrix X (n).
        epsilon: float
            The epsilon for numerical error.
        update_flags: torch.Tensor or None
            The flags to indicate which rows to optimize, shape (p,).
            If None, all rows are optimized.

        Returns
        -------
        int
            The number of improved solutions.

        """

        new_scales_T = new_scales.T.contiguous()

        # Select target rows
        if update_flags is None:
            integers_z = self.integers_z
            current_YX = matrix_YX
            current_Y_sq_norms = Y_sq_norms
            current_mse = self.mean_squared_errors
        else:
            integers_z = self.integers_z[:, update_flags, :]
            current_YX = matrix_YX[update_flags]
            current_Y_sq_norms = Y_sq_norms[update_flags]
            current_mse = self.mean_squared_errors[update_flags]

        # Compute squared errors with new hat_W
        # ||Y_i - hat_W_i X^T||^2 = ||Y_i||^2 - 2(YX)_i · hat_W_i + hat_W_i (X^T X) hat_W_i^T
        hat_W_new = self.get_dequantized_weight_matrix(new_scales_T, integers_z)
        term1 = current_Y_sq_norms
        term2 = 2 * (current_YX * hat_W_new).sum(dim=1)
        term3 = ((hat_W_new @ matrix_XX) * hat_W_new).sum(dim=1)
        new_squared_errors = term1 - term2 + term3
        new_mean_squared_errors = new_squared_errors / dim_n

        # Improvement criterion
        improved = new_mean_squared_errors + epsilon < current_mse
        improved_count = torch.sum(improved)

        # Early return if no improvement in partial-row mode
        if update_flags is not None and improved_count == 0:
            return 0

        # Update
        self._apply_improved_scales(
            update_flags,
            improved,
            improved_count,
            new_scales_T,
            new_squared_errors,
            new_mean_squared_errors,
        )

        return improved_count

    def try_update_group(
        self,
        group_index,
        new_integers_z,
        new_scales,
        matrix_XX,
        matrix_YX,
        Y_sq_norms,
        dim_n,
        return_hat_W=False,
    ):
        """Update a group's integers and scales, recomputing errors for changed rows.

        Reflect new integer assignments and scales obtained from local search,
        and partially recompute objective function values for changed rows.

        Skip condition: Skip if no rows have changed integer assignments.
        If not skipped, unconditionally update all rows where integer assignments
        or scales have changed, and partially recompute errors.

        Parameters
        ----------
        group_index : int
            Index of the group to update.
        new_integers_z : torch.Tensor
            New integer assignments, shape (p, group_size).
        new_scales : torch.Tensor
            New scales, shape (p,).
        matrix_XX : torch.Tensor
            X^T X, shape (m, m).
        matrix_YX : torch.Tensor
            Y @ X, shape (p, m).
        Y_sq_norms : torch.Tensor
            ||Y[i]||^2, shape (p,).
        dim_n : int
            Number of rows in the input matrix n.
        return_hat_W : bool, optional
            If True, return (changed, hat_W_changed).
            Used in subclasses to reuse hat_W_changed. Default is False.

        Returns
        -------
        updated : torch.Tensor
            Boolean flags for updated rows, shape (p,).
        hat_W_changed : torch.Tensor or None
            If return_hat_W=True, the dequantized weight matrix for changed rows.
            None if no changes.

        """

        # Skip if no rows have changed integer assignments
        z_changed = (self.integers_z[group_index] != new_integers_z).any(dim=1)
        if not z_changed.any():
            if return_hat_W:
                return z_changed, None
            return z_changed  # All-False bool tensor

        # Target rows where integer assignments or scales have changed
        s_changed = self.scales[group_index] != new_scales
        changed = z_changed | s_changed

        # Unconditionally update
        self.integers_z[group_index, changed] = new_integers_z[changed]
        self.scales[group_index, changed] = new_scales[changed]

        # Partially recompute errors for changed rows
        hat_W_changed = self.get_dequantized_weight_matrix()[changed]
        term1 = Y_sq_norms[changed]
        term2 = 2.0 * (matrix_YX[changed] * hat_W_changed).sum(dim=1)
        term3 = ((hat_W_changed @ matrix_XX) * hat_W_changed).sum(dim=1)
        self.squared_errors[changed] = term1 - term2 + term3
        self.mean_squared_errors[changed] = self.squared_errors[changed] / dim_n
        self.squared_error = torch.sum(self.squared_errors)
        self.mean_squared_error = self.mean_squared_errors.mean()

        if return_hat_W:
            return changed, hat_W_changed
        return changed

    def _apply_improved_scales(
        self,
        update_flags,
        improved,
        improved_count,
        new_scales_T,
        new_squared_errors,
        new_mean_squared_errors,
    ):
        """Apply the improved scales to the solution.

        Called after the improvement criterion in try_update_scales.
        Can be overridden in subclasses for additional updates (e.g., penalty terms).

        Parameters
        ----------
        update_flags : torch.Tensor or None
            The flags to indicate which rows were optimized, shape (p,).
        improved : torch.Tensor
            Boolean tensor indicating which rows improved, shape (num_target_rows,).
        improved_count : torch.Tensor
            The number of improved rows.
        new_scales_T : torch.Tensor
            New scales (transposed), shape (num_groups, num_target_rows).
        new_squared_errors : torch.Tensor
            New squared errors, shape (num_target_rows,).
        new_mean_squared_errors : torch.Tensor
            New mean squared errors, shape (num_target_rows,).

        """

        if update_flags is None:
            # All-row mode
            if improved_count == improved.shape[0]:
                # All rows improved: bulk assignment
                self.scales = new_scales_T
                self.squared_errors = new_squared_errors
                self.squared_error = torch.sum(new_squared_errors)
                self.mean_squared_errors = new_mean_squared_errors
                self.mean_squared_error = new_mean_squared_errors.mean()
                return None

            # Partial improvement or no improvement: partially update scales and errors
            if improved_count > 0:
                self.scales[:, improved] = new_scales_T[:, improved]
                self.squared_errors[improved] = new_squared_errors[improved]
                self.squared_error = torch.sum(self.squared_errors)
                self.mean_squared_errors[improved] = new_mean_squared_errors[improved]
                self.mean_squared_error = self.mean_squared_errors.mean()
            return None

        # Partial-row mode (assumes improved_count > 0)
        update_indices = torch.where(update_flags)[0]
        improved_indices = update_indices[improved]
        self.scales[:, improved_indices] = new_scales_T[:, improved]
        self.squared_errors[improved_indices] = new_squared_errors[improved]
        self.squared_error = torch.sum(self.squared_errors)
        self.mean_squared_errors[improved_indices] = new_mean_squared_errors[improved]
        self.mean_squared_error = self.mean_squared_errors.mean()
        return improved_indices

    def get_quantized_result(self):
        """Get the quantized result in the original input format.

        Convert scales, zero points, and integer assignments from the internal storage
        format back to the original input format (the format passed to __init__).

        Internal format to input format conversion:
        - scales: (num_groups, p) -> (p, num_groups)
        - zero_point: (num_groups, p) -> (p, num_groups)
        - integers_z: (num_groups, p, group_size) -> assignment: (p, num_groups, group_size)
          Note: zero point is added back to integers_z to recover assignment

        Returns
        -------
        result_scales : torch.Tensor
            The scales of the quantization, shape (p, num_groups).
        result_assignment : torch.Tensor
            The integer assignment of the quantization, shape (p, num_groups, group_size).
        result_zero_point : torch.Tensor
            The zero point of the quantization, shape (p, num_groups).

        """

        result_scales = self.scales.T.clone()
        result_zero_point = self.zero_point.T.clone()
        result_assignment = self.integers_z.transpose(0, 1) + result_zero_point.unsqueeze(2)

        return result_scales, result_assignment, result_zero_point

    def get_dequantized_weight_matrix(self, scalse=None, integers_z=None):
        """Get the dequantized weight matrix."""

        if scalse is None:
            scalse = self.scales
        if integers_z is None:
            integers_z = self.integers_z

        # (hat_W_1, ..., hat_W_k), shape: (num_groups, p, group_size)
        divided_weight_matrices = scalse.unsqueeze(2) * integers_z

        # (hat_W_1, ..., hat_W_k) -> (p, num_groups * group_size)
        return divided_weight_matrices.transpose(0, 1).contiguous().view(scalse.shape[1], -1)

    def get_error_and_mse(self, target_matrix, matrix_X):
        """Get the error and mean squared error."""

        matrix_hatWX = self.get_dequantized_weight_matrix() @ matrix_X.T
        error = torch.sum((target_matrix - matrix_hatWX) ** 2)
        mse = error / (target_matrix.shape[0] * target_matrix.shape[1])
        return error, mse

    def get_squared_errors(self, target_matrix, matrix_X):
        """Get the squared errors."""

        diff = self.get_dequantized_weight_matrix() @ matrix_X.T
        diff -= target_matrix
        diff.pow_(2)
        return torch.sum(diff, dim=1)
