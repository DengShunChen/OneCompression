"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

"""

import torch
import torch.nn as nn
import gc

import math


def calculate_hessian(input_activations, device):
    """Calculate the Hessian matrix for the layer.

    Reference: QEP-dev src/method/test_methods.py (2024/09/23)
    Notes:
    - Processes in batches for memory efficiency
    - Uses scaling for numerical stabilization (not exactly X^T X)

    Args:
        input_activations: The input activations, shape (batch_size, seq_len, hidden_size)
        device: The device to calculate the Hessian matrix
    """

    hidden_size = input_activations.shape[-1]

    # Initialize Hessian
    hessian = torch.zeros((hidden_size, hidden_size), device=device)
    nsamples = 0
    # Process in batches (small batches for memory efficiency)
    batch_size = min(input_activations.shape[0], 8)
    for i in range(0, input_activations.shape[0], batch_size):
        batch_activations = input_activations[i : i + batch_size].to(device).float()
        assert batch_activations.shape[-1] == hidden_size
        batch_activations = batch_activations.reshape((-1, batch_activations.shape[-1]))
        # Transpose to column vectors
        inp = batch_activations.t()
        # Note: Without scaling, this would simply be: hessian += inp @ inp.T
        tmp = batch_activations.shape[0]
        hessian *= nsamples / (nsamples + tmp)
        nsamples += tmp
        # Scaling
        inp_scaled = math.sqrt(2 / nsamples) * inp.float()
        hessian += inp_scaled.matmul(inp_scaled.t())
    return hessian


#############################################
# GPTQ main function (returns scale, zero, q_int)
#############################################
def run_gptq(
    weight,
    input_activations=None,
    blocksize=128,
    percdamp=0.01,
    wbits=16,
    groupsize=-1,
    actorder=False,
    mse=False,
    sym=False,
    q_grid=600,
    q_norm=2.4,
    hessian=None,
):
    """GPTQ quantization (outputs scale, zero_point, and integer codes).

    Parameters
    ----------
    weight : torch.Tensor
        The weight matrix, shape (p, m).
    input_activations : torch.Tensor, optional
        The input activations, shape (n, m).
        Required when hessian is None.
    hessian : torch.Tensor, optional
        Precomputed Hessian, shape (m, m).
        If provided, input_activations is not needed.
        The Hessian is computed as (2/n) * X^T X.
    """

    device = weight.device
    if hessian is not None:
        H = hessian.to(device)
    elif input_activations is not None:
        H = calculate_hessian(input_activations, device)
    else:
        raise ValueError("Either input_activations or hessian must be provided.")

    quantizer = GPTQExcecutor()
    quantizer.configure(
        wbits,
        perchannel=True,
        sym=sym,
        mse=mse,
        norm=q_norm,
        grid=q_grid,
    )

    W = weight.clone()
    W = W.float()

    if not quantizer.ready():
        quantizer.find_params(W, weight=True)

    # Store scale and zero_point
    if groupsize == -1:
        scale = quantizer.scale.detach()
        zero_point = quantizer.zero.detach()
    else:
        num_groups = weight.shape[1] // groupsize
        assert weight.shape[1] % groupsize == 0
        scale = torch.zeros(weight.shape[0], num_groups, device=device, dtype=torch.float64)
        zero_point = torch.zeros(weight.shape[0], num_groups, device=device, dtype=torch.int8)
        scale[:, 0] = quantizer.scale.detach().T
        zero_point[:, 0] = quantizer.zero.detach().T

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    Q = torch.zeros_like(W)
    Q_int = torch.zeros(W.shape, dtype=torch.int8, device=W.device)  # Storage for integer codes

    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    for block_idx, i1 in enumerate(range(0, H.shape[0], blocksize)):
        i2 = min(i1 + blocksize, H.shape[0])
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if groupsize != -1:
                if (i1 + i) % groupsize == 0:
                    quantizer.find_params(W[:, (i1 + i) : (i1 + i + groupsize)], weight=True)
                    scale[:, (i1 + i) // groupsize] = quantizer.scale.detach().T
                    zero_point[:, (i1 + i) // groupsize] = quantizer.zero.detach().T

            # Quantize (real-valued)
            q = quantize(w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq).flatten()
            Q1[:, i] = q

            # Compute and store integer codes
            q_int = quantize_int(
                w.unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
            ).flatten()
            Q_int[:, i1 + i] = q_int

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    if actorder:
        Q = Q[:, invperm]
        Q_int = Q_int[:, invperm]

    # Output
    q_weights = Q.reshape(weight.shape).to(weight.data.dtype)
    q_int_full = Q_int.reshape(weight.shape)
    maxq = int(quantizer.maxq.item()) if hasattr(quantizer.maxq, "item") else int(quantizer.maxq)

    del H, Hinv, W, Q, Q_int
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "q_weights": q_weights,  # Dequantized float weights
        "q_int": q_int_full,  # Integer codes
        "scale": scale,
        "zero_point": zero_point,
        "maxq": maxq,  # Maximum integer value (e.g., 255)
    }


#############################################
# Function to return quantized integer codes
#############################################
def quantize_int(x, scale, zero, maxq):
    """Return quantized integer codes.

    Args:
        x: Original real-valued tensor
        scale, zero: Broadcastable scale and zero point
        maxq: Maximum integer value (e.g., 255)

    Returns:
        torch.long integer codes (0..maxq)
    """
    if maxq < 0:
        # Special case for ternary etc. (skip for now)
        return torch.zeros_like(x, dtype=torch.long)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q.to(torch.long)


### Code below is ported from QEP-dev: src/method/gptq/quant.py (2024/09/23)


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class GPTQExcecutor(nn.Module):

    def __init__(self, shape=1):
        super(GPTQExcecutor, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        trits=False,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
