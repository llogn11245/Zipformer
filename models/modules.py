# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import math
import random
from typing import Optional, Tuple, Union, Type, List, Callable
import warnings
from contextlib import contextmanager
from packaging import version
import torch.nn.init as init
import k2
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

TORCH_VERSION = version.parse(torch.__version__)

@contextmanager
def torch_autocast(device_type="cuda", **kwargs):
    """
    To fix the following warnings:
    /icefall/egs/librispeech/ASR/zipformer/model.py:323:
    FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated.
    Please use `torch.amp.autocast('cuda', args...)` instead.
      with torch.cuda.amp.autocast(enabled=False):
    """
    if TORCH_VERSION >= version.parse("2.3.0"):
        # Use new unified API
        with torch.amp.autocast(device_type=device_type, **kwargs):
            yield
    else:
        # Suppress deprecation warning and use old CUDA-specific autocast
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            with torch.cuda.amp.autocast(**kwargs):
                yield

def logaddexp_onnx(x: Tensor, y: Tensor) -> Tensor:
    max_value = torch.max(x, y)
    diff = torch.abs(x - y)
    return max_value + torch.log1p(torch.exp(-diff))


# RuntimeError: Exporting the operator logaddexp to ONNX opset version
# 14 is not supported. Please feel free to request support or submit
# a pull request on PyTorch GitHub.
#
# The following function is to solve the above error when exporting
# models to ONNX via torch.jit.trace()
def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    # Caution(fangjun): Put torch.jit.is_scripting() before
    # torch.onnx.is_in_onnx_export();
    # otherwise, it will cause errors for torch.jit.script().
    #
    # torch.logaddexp() works for both torch.jit.script() and
    # torch.jit.trace() but it causes errors for ONNX export.
    #
    if torch.jit.is_scripting():
        # Note: We cannot use torch.jit.is_tracing() here as it also
        # matches torch.onnx.export().
        return torch.logaddexp(x, y)
    elif torch.onnx.is_in_onnx_export():
        return logaddexp_onnx(x, y)
    else:
        # for torch.jit.trace()
        return torch.logaddexp(x, y)


class PiecewiseLinear(object):
    """
    Piecewise linear function, from float to float, specified as nonempty list of (x,y) pairs with
    the x values in order.  x values <[initial x] or >[final x] are map to [initial y], [final y]
    respectively.
    """

    def __init__(self, *args):
        assert len(args) >= 1, len(args)
        if len(args) == 1 and isinstance(args[0], PiecewiseLinear):
            self.pairs = list(args[0].pairs)
        else:
            self.pairs = [(float(x), float(y)) for x, y in args]
        for x, y in self.pairs:
            assert isinstance(x, (float, int)), type(x)
            assert isinstance(y, (float, int)), type(y)

        for i in range(len(self.pairs) - 1):
            assert self.pairs[i + 1][0] > self.pairs[i][0], (
                i,
                self.pairs[i],
                self.pairs[i + 1],
            )

    def __str__(self):
        # e.g. 'PiecewiseLinear((0., 10.), (100., 0.))'
        return f"PiecewiseLinear({str(self.pairs)[1:-1]})"

    def __call__(self, x):
        if x <= self.pairs[0][0]:
            return self.pairs[0][1]
        elif x >= self.pairs[-1][0]:
            return self.pairs[-1][1]
        else:
            cur_x, cur_y = self.pairs[0]
            for i in range(1, len(self.pairs)):
                next_x, next_y = self.pairs[i]
                if x >= cur_x and x <= next_x:
                    return cur_y + (next_y - cur_y) * (x - cur_x) / (next_x - cur_x)
                cur_x, cur_y = next_x, next_y
            assert False

    def __mul__(self, alpha):
        return PiecewiseLinear(*[(x, y * alpha) for x, y in self.pairs])

    def __add__(self, x):
        if isinstance(x, (float, int)):
            return PiecewiseLinear(*[(p[0], p[1] + x) for p in self.pairs])
        s, x = self.get_common_basis(x)
        return PiecewiseLinear(
            *[(sp[0], sp[1] + xp[1]) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def max(self, x):
        if isinstance(x, (float, int)):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            *[(sp[0], max(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def min(self, x):
        if isinstance(x, float) or isinstance(x, int):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            *[(sp[0], min(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def __eq__(self, other):
        return self.pairs == other.pairs

    def get_common_basis(self, p: "PiecewiseLinear", include_crossings: bool = False):
        """
        Returns (self_mod, p_mod) which are equivalent piecewise linear
        functions to self and p, but with the same x values.

          p: the other piecewise linear function
          include_crossings: if true, include in the x values positions
              where the functions indicate by this and p cross.
        """
        assert isinstance(p, PiecewiseLinear), type(p)

        # get sorted x-values without repetition.
        x_vals = sorted(set([x for x, _ in self.pairs] + [x for x, _ in p.pairs]))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]

        if include_crossings:
            extra_x_vals = []
            for i in range(len(x_vals) - 1):
                if (y_vals1[i] > y_vals2[i]) != (y_vals1[i + 1] > y_vals2[i + 1]):
                    # if the two lines in this subsegment potentially cross each other..
                    diff_cur = abs(y_vals1[i] - y_vals2[i])
                    diff_next = abs(y_vals1[i + 1] - y_vals2[i + 1])
                    # `pos`, between 0 and 1, gives the relative x position,
                    # with 0 being x_vals[i] and 1 being x_vals[i+1].
                    pos = diff_cur / (diff_cur + diff_next)
                    extra_x_val = x_vals[i] + pos * (x_vals[i + 1] - x_vals[i])
                    extra_x_vals.append(extra_x_val)
            if len(extra_x_vals) > 0:
                x_vals = sorted(set(x_vals + extra_x_vals))

            y_vals1 = [self(x) for x in x_vals]
            y_vals2 = [p(x) for x in x_vals]

        return (
            PiecewiseLinear(*zip(x_vals, y_vals1)),
            PiecewiseLinear(*zip(x_vals, y_vals2)),
        )


class ScheduledFloat(torch.nn.Module):
    """
    This object is a torch.nn.Module only because we want it to show up in [top_level module].modules();
    it does not have a working forward() function.  You are supposed to cast it to float, as
    in, float(parent_module.whatever), and use it as something like a dropout prob.

    It is a floating point value whose value changes depending on the batch count of the
    training loop.  It is a piecewise linear function where you specify the (x,y) pairs
    in sorted order on x; x corresponds to the batch index.  For batch-index values before the
    first x or after the last x, we just use the first or last y value.

    Example:
       self.dropout = ScheduledFloat((0.0, 0.2), (4000.0, 0.0), default=0.0)

    `default` is used when self.batch_count is not set or not in training mode or in
     torch.jit scripting mode.
    """

    def __init__(self, *args, default: float = 0.0):
        super().__init__()
        # self.batch_count and self.name will be written to in the training loop.
        self.batch_count = None
        self.name = None
        self.default = default
        self.schedule = PiecewiseLinear(*args)

    def extra_repr(self) -> str:
        return (
            f"batch_count={self.batch_count}, schedule={str(self.schedule.pairs[1:-1])}"
        )

    def __float__(self):
        batch_count = self.batch_count
        if (
            batch_count is None
            or not self.training
            or torch.jit.is_scripting()
            or torch.jit.is_tracing()
        ):
            return float(self.default)
        else:
            ans = self.schedule(self.batch_count)
            if random.random() < 0.0002:
                logging.info(
                    f"ScheduledFloat: name={self.name}, batch_count={self.batch_count}, ans={ans}"
                )
            return ans

    def __add__(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule + x, default=self.default)
        else:
            return ScheduledFloat(
                self.schedule + x.schedule, default=self.default + x.default
            )

    def max(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule.max(x), default=self.default)
        else:
            return ScheduledFloat(
                self.schedule.max(x.schedule), default=max(self.default, x.default)
            )


FloatLike = Union[float, ScheduledFloat]


def random_cast_to_half(x: Tensor, min_abs: float = 5.0e-06) -> Tensor:
    """
    A randomized way of casting a floating point value to half precision.
    """
    if x.dtype == torch.float16:
        return x
    x_abs = x.abs()
    is_too_small = x_abs < min_abs
    # for elements where is_too_small is true, random_val will contain +-min_abs with
    # probability (x.abs() / min_abs), and 0.0 otherwise.  [so this preserves expectations,
    # for those elements].
    random_val = min_abs * x.sign() * (torch.rand_like(x) * min_abs < x_abs)
    return torch.where(is_too_small, random_val, x).to(torch.float16)


class CutoffEstimator:
    """
    Estimates cutoffs of an arbitrary numerical quantity such that a specified
    proportion of items will be above the cutoff on average.

      p is the proportion of items that should be above the cutoff.
    """

    def __init__(self, p: float):
        self.p = p
        # total count of items
        self.count = 0
        # total count of items that were above the cutoff
        self.count_above = 0
        # initial cutoff value
        self.cutoff = 0

    def __call__(self, x: float) -> bool:
        """
        Returns true if x is above the cutoff.
        """
        ans = x > self.cutoff
        self.count += 1
        if ans:
            self.count_above += 1
        cur_p = self.count_above / self.count
        delta_p = cur_p - self.p
        if (delta_p > 0) == ans:
            q = abs(delta_p)
            self.cutoff = x * q + self.cutoff * (1 - q)
        return ans


class SoftmaxFunction(torch.autograd.Function):
    """
    Tries to handle half-precision derivatives in a randomized way that should
    be more accurate for training than the default behavior.
    """

    @staticmethod
    def forward(ctx, x: Tensor, dim: int):
        ans = x.softmax(dim=dim)
        # if x dtype is float16, x.softmax() returns a float32 because
        # (presumably) that op does not support float16, and autocast
        # is enabled.
        if torch.is_autocast_enabled():
            ans = ans.to(torch.get_autocast_gpu_dtype())
        ctx.save_for_backward(ans)
        ctx.x_dtype = x.dtype
        ctx.dim = dim
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        (ans,) = ctx.saved_tensors
        with torch_autocast(enabled=False):
            ans_grad = ans_grad.to(torch.float32)
            ans = ans.to(torch.float32)
            x_grad = ans_grad * ans
            x_grad = x_grad - ans * x_grad.sum(dim=ctx.dim, keepdim=True)
            return x_grad, None


def softmax(x: Tensor, dim: int):
    if not x.requires_grad or torch.jit.is_scripting() or torch.jit.is_tracing():
        return x.softmax(dim=dim)

    return SoftmaxFunction.apply(x, dim)


class MaxEigLimiterFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        coeffs: Tensor,
        direction: Tensor,
        channel_dim: int,
        grad_scale: float,
    ) -> Tensor:
        ctx.channel_dim = channel_dim
        ctx.grad_scale = grad_scale
        ctx.save_for_backward(x.detach(), coeffs.detach(), direction.detach())
        return x

    @staticmethod
    def backward(ctx, x_grad, *args):
        with torch.enable_grad():
            (x_orig, coeffs, new_direction) = ctx.saved_tensors
            x_orig.requires_grad = True
            num_channels = x_orig.shape[ctx.channel_dim]
            x = x_orig.transpose(ctx.channel_dim, -1).reshape(-1, num_channels)
            new_direction.requires_grad = False
            x = x - x.mean(dim=0)
            x_var = (x**2).mean()
            x_residual = x - coeffs * new_direction
            x_residual_var = (x_residual**2).mean()
            # `variance_proportion` is the proportion of the variance accounted for
            # by the top eigen-direction.  This is to be minimized.
            variance_proportion = (x_var - x_residual_var) / (x_var + 1.0e-20)
            variance_proportion.backward()
        x_orig_grad = x_orig.grad
        x_extra_grad = (
            x_orig.grad
            * ctx.grad_scale
            * x_grad.norm()
            / (x_orig_grad.norm() + 1.0e-20)
        )
        return x_grad + x_extra_grad.detach(), None, None, None, None


class BiasNormFunction(torch.autograd.Function):
    # This computes:
    #   scales = (torch.mean((x - bias) ** 2, keepdim=True)) ** -0.5 * log_scale.exp()
    #   return x * scales
    # (after unsqueezing the bias), but it does it in a memory-efficient way so that
    # it can just store the returned value (chances are, this will also be needed for
    # some other reason, related to the next operation, so we can save memory).
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        bias: Tensor,
        log_scale: Tensor,
        channel_dim: int,
        store_output_for_backprop: bool,
    ) -> Tensor:
        assert bias.ndim == 1
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.store_output_for_backprop = store_output_for_backprop
        ctx.channel_dim = channel_dim
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (
            torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
        ) * log_scale.exp()
        ans = x * scales
        ctx.save_for_backward(
            ans.detach() if store_output_for_backprop else x,
            scales.detach(),
            bias.detach(),
            log_scale.detach(),
        )
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        ans_or_x, scales, bias, log_scale = ctx.saved_tensors
        if ctx.store_output_for_backprop:
            x = ans_or_x / scales
        else:
            x = ans_or_x
        x = x.detach()
        x.requires_grad = True
        bias.requires_grad = True
        log_scale.requires_grad = True
        with torch.enable_grad():
            # recompute scales from x, bias and log_scale.
            scales = (
                torch.mean((x - bias) ** 2, dim=ctx.channel_dim, keepdim=True) ** -0.5
            ) * log_scale.exp()
            ans = x * scales
            ans.backward(gradient=ans_grad)
        return x.grad, bias.grad.flatten(), log_scale.grad, None, None


class BiasNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    Instead, we give the BiasNorm a trainable bias that it can use when
    computing the scale for normalization.  We also give it a (scalar)
    trainable scale on the output.


    Args:
       num_channels: the number of channels, e.g. 512.
       channel_dim: the axis/dimension corresponding to the channel,
         interpreted as an offset from the input's ndim if negative.
         This is NOT the num_channels; it should typically be one of
         {-2, -1, 0, 1, 2, 3}.
      log_scale: the initial log-scale that we multiply the output by; this
         is learnable.
      log_scale_min: FloatLike, minimum allowed value of log_scale
      log_scale_max: FloatLike, maximum allowed value of log_scale
      store_output_for_backprop: only possibly affects memory use; recommend
         to set to True if you think the output of this module is more likely
         than the input of this module to be required to be stored for the
         backprop.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        log_scale: float = 1.0,
        log_scale_min: float = -1.5,
        log_scale_max: float = 1.5,
        store_output_for_backprop: bool = False,
    ) -> None:
        super(BiasNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(torch.empty(num_channels).normal_(mean=0, std=1e-4))

        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

        self.store_output_for_backprop = store_output_for_backprop

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += x.ndim
            bias = self.bias
            for _ in range(channel_dim + 1, x.ndim):
                bias = bias.unsqueeze(-1)
            scales = (
                torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
            ) * self.log_scale.exp()
            return x * scales

        log_scale = limit_param_value(
            self.log_scale,
            min=float(self.log_scale_min),
            max=float(self.log_scale_max),
            training=self.training,
        )

        return BiasNormFunction.apply(
            x, self.bias, log_scale, self.channel_dim, self.store_output_for_backprop
        )


def ScaledLinear(*args, initial_scale: float = 1.0, **kwargs) -> nn.Linear:
    """
    Behaves like a constructor of a modified version of nn.Linear
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Linear(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


def ScaledConv1d(*args, initial_scale: float = 1.0, **kwargs) -> nn.Conv1d:
    """
    Behaves like a constructor of a modified version of nn.Conv1d
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Conv1d(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


def ScaledConv2d(*args, initial_scale: float = 1.0, **kwargs) -> nn.Conv2d:
    """
    Behaves like a constructor of a modified version of nn.Conv2d
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False, but:
    NO PADDING-RELATED ARGS.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Conv2d(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


class ChunkCausalDepthwiseConv1d(torch.nn.Module):
    """
    Behaves like a depthwise 1d convolution, except that it is causal in
    a chunkwise way, as if we had a block-triangular attention mask.
    The chunk size is provided at test time (it should probably be
    kept in sync with the attention mask).

    This has a little more than twice the parameters of a conventional
    depthwise conv1d module: we implement it by having one
    depthwise convolution, of half the width, that is causal (via
    right-padding); and one depthwise convolution that is applied only
    within chunks, that we multiply by a scaling factor which depends
    on the position within the chunk.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        initial_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        half_kernel_size = (kernel_size + 1) // 2
        # will pad manually, on one side.
        self.causal_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=half_kernel_size,
            padding=0,
            bias=True,
        )

        self.chunkwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        # first row is correction factors added to the scale near the left edge of the chunk,
        # second row is correction factors added to the scale near the right edge of the chunk,
        # both of these are added to a default scale of 1.0.
        self.chunkwise_conv_scale = nn.Parameter(torch.zeros(2, channels, kernel_size))
        self.kernel_size = kernel_size

        with torch.no_grad():
            self.causal_conv.weight[:] *= initial_scale
            self.chunkwise_conv.weight[:] *= initial_scale
            if bias:
                torch.nn.init.uniform_(
                    self.causal_conv.bias, -0.1 * initial_scale, 0.1 * initial_scale
                )

    def forward(self, x: Tensor, chunk_size: int = -1) -> Tensor:
        """Forward function.

        Args:
               x: a Tensor of shape (batch_size, channels, seq_len)
        chunk_size: the chunk size, in frames; does not have to divide seq_len exactly.
        """
        (batch_size, num_channels, seq_len) = x.shape

        # half_kernel_size = self.kernel_size + 1 // 2
        # left_pad is half_kernel_size - 1 where half_kernel_size is the size used
        # in the causal conv.  It's the amount by which we must pad on the left,
        # to make the convolution causal.
        left_pad = self.kernel_size // 2

        if chunk_size < 0 or chunk_size > seq_len:
            chunk_size = seq_len
        right_pad = -seq_len % chunk_size

        x = torch.nn.functional.pad(x, (left_pad, right_pad))

        x_causal = self.causal_conv(x[..., : left_pad + seq_len])
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        num_chunks = x_chunk.shape[2] // chunk_size
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks, chunk_size)
        x_chunk = x_chunk.permute(0, 2, 1, 3).reshape(
            batch_size * num_chunks, num_channels, chunk_size
        )
        x_chunk = self.chunkwise_conv(x_chunk)  # does not change shape

        chunk_scale = self._get_chunk_scale(chunk_size)

        x_chunk = x_chunk * chunk_scale
        x_chunk = x_chunk.reshape(
            batch_size, num_chunks, num_channels, chunk_size
        ).permute(0, 2, 1, 3)
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks * chunk_size)[
            ..., :seq_len
        ]

        return x_chunk + x_causal

    def _get_chunk_scale(self, chunk_size: int):
        """Returns tensor of shape (num_channels, chunk_size) that will be used to
        scale the output of self.chunkwise_conv."""
        left_edge = self.chunkwise_conv_scale[0]
        right_edge = self.chunkwise_conv_scale[1]
        if chunk_size < self.kernel_size:
            left_edge = left_edge[:, :chunk_size]
            right_edge = right_edge[:, -chunk_size:]
        else:
            t = chunk_size - self.kernel_size
            channels = left_edge.shape[0]
            pad = torch.zeros(
                channels, t, device=left_edge.device, dtype=left_edge.dtype
            )
            left_edge = torch.cat((left_edge, pad), dim=-1)
            right_edge = torch.cat((pad, right_edge), dim=-1)
        return 1.0 + (left_edge + right_edge)

    def streaming_forward(
        self,
        x: Tensor,
        cache: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Streaming Forward function.

        Args:
            x: a Tensor of shape (batch_size, channels, seq_len)
            cache: cached left context of shape (batch_size, channels, left_pad)
        """
        (batch_size, num_channels, seq_len) = x.shape

        # left_pad is half_kernel_size - 1 where half_kernel_size is the size used
        # in the causal conv.  It's the amount by which we must pad on the left,
        # to make the convolution causal.
        left_pad = self.kernel_size // 2

        # Pad cache
        assert cache.shape[-1] == left_pad, (cache.shape[-1], left_pad)
        x = torch.cat([cache, x], dim=2)
        # Update cache
        cache = x[..., -left_pad:]

        x_causal = self.causal_conv(x)
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        x_chunk = self.chunkwise_conv(x_chunk)  # does not change shape

        chunk_scale = self._get_chunk_scale(chunk_size=seq_len)
        x_chunk = x_chunk * chunk_scale

        return x_chunk + x_causal, cache


class BalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        min_mean: float,
        max_mean: float,
        min_rms: float,
        max_rms: float,
        grad_scale: float,
        channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        ctx.save_for_backward(x)
        ctx.config = (min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim)
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        (x,) = ctx.saved_tensors
        (min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim) = ctx.config

        try:
            with torch.enable_grad():
                with torch_autocast(enabled=False):
                    x = x.to(torch.float32)
                    x = x.detach()
                    x.requires_grad = True
                    mean_dims = [i for i in range(x.ndim) if i != channel_dim]
                    uncentered_var = (x**2).mean(dim=mean_dims, keepdim=True)
                    mean = x.mean(dim=mean_dims, keepdim=True)
                    stddev = (uncentered_var - (mean * mean)).clamp(min=1.0e-20).sqrt()
                    rms = uncentered_var.clamp(min=1.0e-20).sqrt()

                    m = mean / stddev
                    # part of loss that relates to mean / stddev
                    m_loss = (m - m.clamp(min=min_mean, max=max_mean)).abs()

                    # put a much larger scale on the RMS-max-limit loss, so that if both it and the
                    # m_loss are violated we fix the RMS loss first.
                    rms_clamped = rms.clamp(min=min_rms, max=max_rms)
                    r_loss = (rms_clamped / rms).log().abs()

                    loss = m_loss + r_loss

                    loss.backward(gradient=torch.ones_like(loss))
                    loss_grad = x.grad
                    loss_grad_rms = (
                        (loss_grad**2)
                        .mean(dim=mean_dims, keepdim=True)
                        .sqrt()
                        .clamp(min=1.0e-20)
                    )

                    loss_grad = loss_grad * (grad_scale / loss_grad_rms)

                    x_grad_float = x_grad.to(torch.float32)
                    # scale each element of loss_grad by the absolute value of the corresponding
                    # element of x_grad, which we view as a noisy estimate of its magnitude for that
                    # (frame and dimension).  later we can consider factored versions.
                    x_grad_mod = x_grad_float + (x_grad_float.abs() * loss_grad)
                    x_grad = x_grad_mod.to(x_grad.dtype)
        except Exception as e:
            logging.info(
                f"Caught exception in Balancer backward: {e}, size={list(x_grad.shape)}, will continue."
            )

        return x_grad, None, None, None, None, None, None


class Balancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.

    Args:
           num_channels: the number of channels
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), above which we start to modify the derivatives.
           scale_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_abs and max_abs
              are violated.
           min_abs:  the minimum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_abs:  the maximum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
         prob: determines the minimum probability with which we modify the
             gradients for the {min,max}_positive and {min,max}_abs constraints,
             on each forward().  This is done randomly to prevent all layers
             from doing it at the same time.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        min_positive: FloatLike = 0.05,
        max_positive: FloatLike = 0.95,
        min_abs: FloatLike = 0.2,
        max_abs: FloatLike = 100.0,
        grad_scale: FloatLike = 0.04,
        prob: Optional[FloatLike] = None,
    ):
        super().__init__()

        if prob is None:
            prob = ScheduledFloat((0.0, 0.5), (8000.0, 0.125), default=0.4)
        self.prob = prob
        # 5% of the time we will return and do nothing because memory usage is
        # too high.
        self.mem_cutoff = CutoffEstimator(0.05)

        # actually self.num_channels is no longer needed except for an assertion.
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.grad_scale = grad_scale

    def forward(self, x: Tensor) -> Tensor:
        if (
            torch.jit.is_scripting()
            or not x.requires_grad
            or (x.is_cuda and self.mem_cutoff(torch.cuda.memory_allocated()))
        ):
            return _no_op(x)

        prob = float(self.prob)
        if random.random() < prob:
            # The following inner-functions convert from the way we historically specified
            # these limitations, as limits on the absolute value and the proportion of positive
            # values, to limits on the RMS value and the (mean / stddev).
            def _abs_to_rms(x):
                # for normally distributed data, if the expected absolute value is x, the
                # expected rms value will be sqrt(pi/2) * x.
                return 1.25331413732 * x

            def _proportion_positive_to_mean(x):
                def _atanh(x):
                    eps = 1.0e-10
                    # eps is to prevent crashes if x is exactly 0 or 1.
                    # we'll just end up returning a fairly large value.
                    return (math.log(1 + x + eps) - math.log(1 - x + eps)) / 2.0

                def _approx_inverse_erf(x):
                    # 1 / (sqrt(pi) * ln(2)),
                    # see https://math.stackexchange.com/questions/321569/approximating-the-error-function-erf-by-analytical-functions
                    # this approximation is extremely crude and gets progressively worse for
                    # x very close to -1 or +1, but we mostly care about the "middle" region
                    # e.g. _approx_inverse_erf(0.05) = 0.0407316414078772,
                    # and math.erf(0.0407316414078772) = 0.045935330944660666,
                    # which is pretty close to 0.05.
                    return 0.8139535143 * _atanh(x)

                # first convert x from the range 0..1 to the range -1..1 which the error
                # function returns
                x = -1 + (2 * x)
                return _approx_inverse_erf(x)

            min_mean = _proportion_positive_to_mean(float(self.min_positive))
            max_mean = _proportion_positive_to_mean(float(self.max_positive))
            min_rms = _abs_to_rms(float(self.min_abs))
            max_rms = _abs_to_rms(float(self.max_abs))
            grad_scale = float(self.grad_scale)

            assert x.shape[self.channel_dim] == self.num_channels

            return BalancerFunction.apply(
                x, min_mean, max_mean, min_rms, max_rms, grad_scale, self.channel_dim
            )
        else:
            return _no_op(x)


def penalize_abs_values_gt(
    x: Tensor, limit: float, penalty: float, name: str = None
) -> Tensor:
    """
    Returns x unmodified, but in backprop will put a penalty for the excess of
    the absolute values of elements of x over the limit "limit".  E.g. if
    limit == 10.0, then if x has any values over 10 it will get a penalty.

    Caution: the value of this penalty will be affected by grad scaling used
    in automatic mixed precision training.  For this reasons we use this,
    it shouldn't really matter, or may even be helpful; we just use this
    to disallow really implausible values of scores to be given to softmax.

    The name is for randomly printed debug info.
    """
    x_sign = x.sign()
    over_limit = (x.abs() - limit) > 0
    # The following is a memory efficient way to penalize the absolute values of
    # x that's over the limit.  (The memory efficiency comes when you think
    # about which items torch needs to cache for the autograd, and which ones it
    # can throw away).  The numerical value of aux_loss as computed here will
    # actually be larger than it should be, by limit * over_limit.sum(), but it
    # has the same derivative as the real aux_loss which is penalty * (x.abs() -
    # limit).relu().
    aux_loss = penalty * ((x_sign * over_limit).to(torch.int8) * x)
    # note: we don't do sum() here on aux)_loss, but it's as if we had done
    # sum() due to how with_loss() works.
    x = with_loss(x, aux_loss, name)
    # you must use x for something, or this will be ineffective.
    return x


def _diag(x: Tensor):  # like .diag(), but works for tensors with 3 dims.
    if x.ndim == 2:
        return x.diag()
    else:
        (batch, dim, dim) = x.shape
        x = x.reshape(batch, dim * dim)
        x = x[:, :: dim + 1]
        assert x.shape == (batch, dim)
        return x


def _whitening_metric(x: Tensor, num_groups: int):
    """
    Computes the "whitening metric", a value which will be 1.0 if all the eigenvalues of
    of the centered feature covariance are the same within each group's covariance matrix
    and also between groups.
    Args:
        x: a Tensor of shape (*, num_channels)
     num_groups:  the number of groups of channels, a number >=1 that divides num_channels
    Returns:
        Returns a scalar Tensor that will be 1.0 if the data is "perfectly white" and
    greater than 1.0 otherwise.
    """
    assert x.dtype != torch.float16
    x = x.reshape(-1, x.shape[-1])
    (num_frames, num_channels) = x.shape
    assert num_channels % num_groups == 0
    channels_per_group = num_channels // num_groups
    x = x.reshape(num_frames, num_groups, channels_per_group).transpose(0, 1)
    # x now has shape (num_groups, num_frames, channels_per_group)
    # subtract the mean so we use the centered, not uncentered, covariance.
    # My experience has been that when we "mess with the gradients" like this,
    # it's better not do anything that tries to move the mean around, because
    # that can easily cause instability.
    x = x - x.mean(dim=1, keepdim=True)
    # x_covar: (num_groups, channels_per_group, channels_per_group)
    x_covar = torch.matmul(x.transpose(1, 2), x)
    x_covar_mean_diag = _diag(x_covar).mean()
    # the following expression is what we'd get if we took the matrix product
    # of each covariance and measured the mean of its trace, i.e.
    # the same as _diag(torch.matmul(x_covar, x_covar)).mean().
    x_covarsq_mean_diag = (x_covar**2).sum() / (num_groups * channels_per_group)
    # this metric will be >= 1.0; the larger it is, the less 'white' the data was.
    metric = x_covarsq_mean_diag / (x_covar_mean_diag**2 + 1.0e-20)
    return metric


class WhiteningPenaltyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, module: nn.Module) -> Tensor:
        ctx.save_for_backward(x)
        ctx.module = module
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x_orig,) = ctx.saved_tensors
        w = ctx.module

        try:
            with torch.enable_grad():
                with torch_autocast(enabled=False):
                    x_detached = x_orig.to(torch.float32).detach()
                    x_detached.requires_grad = True

                    metric = _whitening_metric(x_detached, w.num_groups)

                    # if random.random() < 0.005 or __name__ == "__main__":
                    #     logging.info(
                    #         f"Whitening: name={w.name}, num_groups={w.num_groups}, num_channels={x_orig.shape[-1]}, "
                    #         f"metric={metric.item():.2f} vs. limit={float(w.whitening_limit)}"
                    #     )

                    if metric < float(w.whitening_limit):
                        w.prob = w.min_prob
                        return x_grad, None
                    else:
                        w.prob = w.max_prob
                        metric.backward()
                        penalty_grad = x_detached.grad
                        scale = float(w.grad_scale) * (
                            x_grad.to(torch.float32).norm()
                            / (penalty_grad.norm() + 1.0e-20)
                        )
                        penalty_grad = penalty_grad * scale
                        return x_grad + penalty_grad.to(x_grad.dtype), None
        except Exception as e:
            logging.info(
                f"Caught exception in Whiten backward: {e}, size={list(x_grad.shape)}, will continue."
            )
        return x_grad, None


class Whiten(nn.Module):
    def __init__(
        self,
        num_groups: int,
        whitening_limit: FloatLike,
        prob: Union[float, Tuple[float, float]],
        grad_scale: FloatLike,
    ):
        """
        Args:
          num_groups: the number of groups to divide the channel dim into before
            whitening.  We will attempt to make the feature covariance
            within each group, after mean subtraction, as "white" as possible,
            while having the same trace across all groups.
         whitening_limit: a value greater than 1.0, that dictates how much
           freedom we have to violate the constraints.  1.0 would mean perfectly
           white, with exactly the same trace across groups; larger values
           give more freedom.  E.g. 2.0.
         prob: the probability with which we apply the gradient modification
           (also affects the grad scale).  May be supplied as a float,
           or as a pair (min_prob, max_prob)

          grad_scale: determines the scale on the gradient term from this object,
            relative to the rest of the gradient on the attention weights.
            E.g. 0.02 (you may want to use smaller values than this if prob is large)
        """
        super(Whiten, self).__init__()
        assert num_groups >= 1
        assert float(whitening_limit) >= 1
        assert float(grad_scale) >= 0
        self.num_groups = num_groups
        self.whitening_limit = whitening_limit
        self.grad_scale = grad_scale

        if isinstance(prob, float):
            prob = (prob, prob)
        (self.min_prob, self.max_prob) = prob
        assert 0 < self.min_prob <= self.max_prob <= 1
        self.prob = self.max_prob
        self.name = None  # will be set in training loop

    def forward(self, x: Tensor) -> Tensor:
        """
        In the forward pass, this function just returns the input unmodified.
        In the backward pass, it will modify the gradients to ensure that the
        distribution in each group has close to (lambda times I) as the covariance
        after mean subtraction, with the same lambda across groups.
        For whitening_limit > 1, there will be more freedom to violate this
        constraint.

        Args:
           x: the input of shape (*, num_channels)

        Returns:
            x, unmodified.   You should make sure
        you use the returned value, or the graph will be freed
        and nothing will happen in backprop.
        """
        grad_scale = float(self.grad_scale)
        if not x.requires_grad or random.random() > self.prob or grad_scale == 0:
            return _no_op(x)
        else:
            return WhiteningPenaltyFunction.apply(x, self)


class WithLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, name: str):
        ctx.y_shape = y.shape
        if random.random() < 0.002 and name is not None:
            loss_sum = y.sum().item()
            logging.info(f"WithLoss: name={name}, loss-sum={loss_sum:.3e}")
        return x

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        return (
            ans_grad,
            torch.ones(ctx.y_shape, dtype=ans_grad.dtype, device=ans_grad.device),
            None,
        )


def with_loss(x, y, name):
    # returns x but adds y.sum() to the loss function.
    return WithLoss.apply(x, y, name)


class ScaleGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad: Tensor):
        return grad * ctx.alpha, None


def scale_grad(x: Tensor, alpha: float):
    return ScaleGradFunction.apply(x, alpha)


class ScaleGrad(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return x
        return scale_grad(x, self.alpha)


class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors
        # where x < ctx.min, ensure all grads are negative (this will tend to make
        # x more positive).
        x_grad = x_grad * torch.where(
            torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0
        )
        # where x > ctx.max, ensure all grads are positive (this will tend to make
        # x more negative).
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(
    x: Tensor, min: float, max: float, prob: float = 0.6, training: bool = True
):
    # You apply this to (typically) an nn.Parameter during training to ensure that its
    # (elements mostly) stays within a supplied range.  This is done by modifying the
    # gradients in backprop.
    # It's not necessary to do this on every batch: do it only some of the time,
    # to save a little time.
    if training and random.random() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x


def _no_op(x: Tensor) -> Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x
    else:
        # a no-op function that will have a node in the autograd graph,
        # to avoid certain bugs relating to backward hooks
        return x.chunk(1, dim=-1)[0]


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return _no_op(x)


class DoubleSwishFunction(torch.autograd.Function):
    """
      double_swish(x) = x * torch.sigmoid(x-1)

    This is a definition, originally motivated by its close numerical
    similarity to swish(swish(x)), where swish(x) =  x * sigmoid(x).

    Memory-efficient derivative computation:
     double_swish(x) = x * s, where s(x) = torch.sigmoid(x-1)
     double_swish'(x) = d/dx double_swish(x) =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
     Now, s'(x) = s(x) * (1-s(x)).
     double_swish'(x) =  x * s'(x) + s(x).
                      =  x * s(x) * (1-s(x)) + s(x).
                     = double_swish(x) * (1-s(x)) + s(x)
     ... so we just need to remember s(x) but not x itself.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.to(torch.float32)

        s = torch.sigmoid(x - 1.0)
        y = x * s

        if requires_grad:
            deriv = y * (1 - s) + s

            # notes on derivative of x * sigmoid(x - 1):
            # https://www.wolframalpha.com/input?i=d%2Fdx+%28x+*+sigmoid%28x-1%29%29
            # min \simeq -0.043638.  Take floor as -0.044 so it's a lower bund
            # max \simeq 1.1990.   Take ceil to be 1.2 so it's an upper bound.
            # the combination of "+ torch.rand_like(deriv)" and casting to torch.uint8 (which
            # floors), should be expectation-preserving.
            floor = -0.044
            ceil = 1.2
            d_scaled = (deriv - floor) * (255.0 / (ceil - floor)) + torch.rand_like(
                deriv
            )
            if __name__ == "__main__":
                # for self-testing only.
                assert d_scaled.min() >= 0.0
                assert d_scaled.max() < 256.0
            d_int = d_scaled.to(torch.uint8)
            ctx.save_for_backward(d_int)
        if x.dtype == torch.float16 or torch.is_autocast_enabled():
            y = y.to(torch.float16)
        return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.
        floor = -0.043637
        ceil = 1.2

        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class DoubleSwish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return x * torch.sigmoid(x - 1.0)
        return DoubleSwishFunction.apply(x)


# Dropout2 is just like normal dropout, except it supports schedules on the dropout rates.
class Dropout2(nn.Module):
    def __init__(self, p: FloatLike):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.dropout(x, p=float(self.p), training=self.training)


class MulForDropout3(torch.autograd.Function):
    # returns (x * y * alpha) where alpha is a float and y doesn't require
    # grad and is zero-or-one.
    @staticmethod
    @custom_fwd
    def forward(ctx, x, y, alpha):
        assert not y.requires_grad
        ans = x * y * alpha
        ctx.save_for_backward(ans)
        ctx.alpha = alpha
        return ans

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad):
        (ans,) = ctx.saved_tensors
        x_grad = ctx.alpha * ans_grad * (ans != 0)
        return x_grad, None, None


# Dropout3 is just like normal dropout, except it supports schedules on the dropout rates,
# and it lets you choose one dimension to share the dropout mask over
class Dropout3(nn.Module):
    def __init__(self, p: FloatLike, shared_dim: int):
        super().__init__()
        self.p = p
        self.shared_dim = shared_dim

    def forward(self, x: Tensor) -> Tensor:
        p = float(self.p)
        if not self.training or p == 0:
            return _no_op(x)
        scale = 1.0 / (1 - p)
        rand_shape = list(x.shape)
        rand_shape[self.shared_dim] = 1
        mask = torch.rand(*rand_shape, device=x.device) > p
        ans = MulForDropout3.apply(x, mask, scale)
        return ans


class SwooshLFunction(torch.autograd.Function):
    """
    swoosh_l(x) =  log(1 + exp(x-4)) - 0.08*x - 0.035
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.to(torch.float32)

        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        coeff = -0.08

        with torch_autocast(enabled=False):
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad = True
                y = torch.logaddexp(zero, x - 4.0) + coeff * x - 0.035

                if not requires_grad:
                    return y

                y.backward(gradient=torch.ones_like(y))

                grad = x.grad
                floor = coeff
                ceil = 1.0 + coeff + 0.005

                d_scaled = (grad - floor) * (255.0 / (ceil - floor)) + torch.rand_like(
                    grad
                )
                if __name__ == "__main__":
                    # for self-testing only.
                    assert d_scaled.min() >= 0.0
                    assert d_scaled.max() < 256.0

                d_int = d_scaled.to(torch.uint8)
                ctx.save_for_backward(d_int)
                if x.dtype == torch.float16 or torch.is_autocast_enabled():
                    y = y.to(torch.get_autocast_gpu_dtype())
                return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.

        coeff = -0.08
        floor = coeff
        ceil = 1.0 + coeff + 0.005
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshL(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-L activation."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return logaddexp(zero, x - 4.0) - 0.08 * x - 0.035
        if not x.requires_grad:
            return k2.swoosh_l_forward(x).to(x.dtype)
        else:
            return k2.swoosh_l(x).to(x.dtype)
        # return SwooshLFunction.apply(x)


class SwooshLOnnx(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-L activation."""
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return logaddexp_onnx(zero, x - 4.0) - 0.08 * x - 0.035


class SwooshRFunction(torch.autograd.Function):
    """
     swoosh_r(x) =  log(1 + exp(x-1)) - 0.08*x - 0.313261687

    derivatives are between -0.08 and 0.92.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad

        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.to(torch.float32)

        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        with torch_autocast(enabled=False):
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad = True
                y = torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687

                if not requires_grad:
                    return y
                y.backward(gradient=torch.ones_like(y))

                grad = x.grad
                floor = -0.08
                ceil = 0.925

                d_scaled = (grad - floor) * (255.0 / (ceil - floor)) + torch.rand_like(
                    grad
                )
                if __name__ == "__main__":
                    # for self-testing only.
                    assert d_scaled.min() >= 0.0
                    assert d_scaled.max() < 256.0

                d_int = d_scaled.to(torch.uint8)
                ctx.save_for_backward(d_int)
                if x.dtype == torch.float16 or torch.is_autocast_enabled():
                    y = y.to(torch.get_autocast_gpu_dtype())
                return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.
        floor = -0.08
        ceil = 0.925
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshR(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-R activation."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687
        if not x.requires_grad:
            return k2.swoosh_r_forward(x).to(x.dtype)
        else:
            return k2.swoosh_r(x).to(x.dtype)
        # return SwooshRFunction.apply(x)


class SwooshROnnx(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-R activation."""
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return logaddexp_onnx(zero, x - 1.0) - 0.08 * x - 0.313261687


# simple version of SwooshL that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshLForward(x: Tensor):
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.035


# simple version of SwooshR that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshRForward(x: Tensor):
    x_offset = x - 1.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.313261687


class ActivationDropoutAndLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        activation: str,
        dropout_p: float,
        dropout_shared_dim: Optional[int],
    ):
        if dropout_p != 0.0:
            dropout_shape = list(x.shape)
            if dropout_shared_dim is not None:
                dropout_shape[dropout_shared_dim] = 1
            # else it won't be very memory efficient.
            dropout_mask = (1.0 / (1.0 - dropout_p)) * (
                torch.rand(*dropout_shape, device=x.device, dtype=x.dtype) > dropout_p
            )
        else:
            dropout_mask = None

        ctx.save_for_backward(x, weight, bias, dropout_mask)

        ctx.activation = activation

        forward_activation_dict = {
            "SwooshL": k2.swoosh_l_forward,
            "SwooshR": k2.swoosh_r_forward,
        }
        # it will raise a KeyError if this fails.  This will be an error.  We let it
        # propagate to the user.
        activation_func = forward_activation_dict[activation]
        x = activation_func(x)
        if dropout_mask is not None:
            x = x * dropout_mask
        x = torch.nn.functional.linear(x, weight, bias)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad: Tensor):
        saved = ctx.saved_tensors
        (x, weight, bias, dropout_mask) = saved

        forward_and_deriv_activation_dict = {
            "SwooshL": k2.swoosh_l_forward_and_deriv,
            "SwooshR": k2.swoosh_r_forward_and_deriv,
        }
        # the following lines a KeyError if the activation is unrecognized.
        # This will be an error.  We let it propagate to the user.
        func = forward_and_deriv_activation_dict[ctx.activation]

        y, func_deriv = func(x)
        if dropout_mask is not None:
            y = y * dropout_mask
        # now compute derivative of y w.r.t. weight and bias..
        # y: (..., in_channels), ans_grad: (..., out_channels),
        (out_channels, in_channels) = weight.shape

        in_channels = y.shape[-1]
        g = ans_grad.reshape(-1, out_channels)
        weight_deriv = torch.matmul(g.t(), y.reshape(-1, in_channels))
        y_deriv = torch.matmul(ans_grad, weight)
        bias_deriv = None if bias is None else g.sum(dim=0)
        x_deriv = y_deriv * func_deriv
        if dropout_mask is not None:
            # order versus func_deriv does not matter
            x_deriv = x_deriv * dropout_mask

        return x_deriv, weight_deriv, bias_deriv, None, None, None


class ActivationDropoutAndLinear(torch.nn.Module):
    """
     This merges an activation function followed by dropout and then a nn.Linear module;
     it does so in a memory efficient way so that it only stores the input to the whole
     module.  If activation == SwooshL and dropout_shared_dim != None, this will be
     equivalent to:
       nn.Sequential(SwooshL(),
                     Dropout3(dropout_p, shared_dim=dropout_shared_dim),
                     ScaledLinear(in_channels, out_channels, bias=bias,
                                  initial_scale=initial_scale))
    If dropout_shared_dim is None, the dropout would be equivalent to
    Dropout2(dropout_p).  Note: Dropout3 will be more memory efficient as the dropout
    mask is smaller.

     Args:
        in_channels: number of input channels, e.g. 256
        out_channels: number of output channels, e.g. 256
        bias: if true, have a bias
        activation: the activation function, for now just support SwooshL.
        dropout_p: the dropout probability or schedule (happens after nonlinearity).
        dropout_shared_dim: the dimension, if any, across which the dropout mask is
             shared (e.g. the time dimension).  If None, this may be less memory
             efficient if there are modules before this one that cache the input
             for their backprop (e.g. Balancer or Whiten).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
        dropout_p: FloatLike = 0.0,
        dropout_shared_dim: Optional[int] = -1,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        # create a temporary module of nn.Linear that we'll steal the
        # weights and bias from
        l = ScaledLinear(
            in_channels, out_channels, bias=bias, initial_scale=initial_scale
        )

        self.weight = l.weight
        # register_parameter properly handles making it a parameter when l.bias
        # is None. I think there is some reason for doing it this way rather
        # than just setting it to None but I don't know what it is, maybe
        # something to do with exporting the module..
        self.register_parameter("bias", l.bias)

        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim

    def forward(self, x: Tensor):
        if not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            if self.activation == "SwooshL":
                x = SwooshLForward(x)
            elif self.activation == "SwooshR":
                x = SwooshRForward(x)
            else:
                assert False, self.activation
            return torch.nn.functional.linear(x, self.weight, self.bias)

        return ActivationDropoutAndLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.activation,
            float(self.dropout_p),
            self.dropout_shared_dim,
        )


def convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    if num_channels <= x.shape[-1]:
        return x[..., :num_channels]
    else:
        shape = list(x.shape)
        shape[-1] = num_channels - shape[-1]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return torch.cat((x, zeros), dim=-1)

class Conv2dSubampling(nn.Module):
    """
    Convolutional 2D subsampling (to 1/4 length)

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns: outputs, output_lengths
        - **outputs** (batch, time, dim): Tensor produced by the convolution
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubampling, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor):
        x = x.unsqueeze(1)  # (batch, 1, time, dim)
        B, C, T, F = x.shape
        for layer in self.sequential:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                k = layer.kernel_size[0]
                s = layer.stride[0]
                d = layer.dilation[0]
                p = layer.padding[0]
                out_T = (T + 2 * p - d * (k - 1) - 1) // s + 1
                pad_len = T - input_lengths
                data_len = input_lengths
                new_len = calc_data_len(
                    result_len=out_T,
                    pad_len=pad_len,
                    data_len=data_len,
                    kernel_size=k,
                    stride=s,
                )
                T = out_T
        B, C, T, F = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, C * F)
        return x, new_len


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ConvNeXt(nn.Module):
    """
    Our interpretation of the ConvNeXt module as used in https://arxiv.org/pdf/2206.14747.pdf
    """

    def __init__(
        self,
        channels: int,
        hidden_ratio: int = 3,
        kernel_size: Tuple[int, int] = (7, 7),
        layerdrop_rate: FloatLike = None,
    ):
        super().__init__()
        self.padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        hidden_channels = channels * hidden_ratio
        if layerdrop_rate is None:
            layerdrop_rate = ScheduledFloat((0.0, 0.2), (20000.0, 0.015))
        self.layerdrop_rate = layerdrop_rate

        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=self.padding,
        )

        self.pointwise_conv1 = nn.Conv2d(
            in_channels=channels, out_channels=hidden_channels, kernel_size=1
        )

        self.hidden_balancer = Balancer(
            hidden_channels,
            channel_dim=1,
            min_positive=0.3,
            max_positive=1.0,
            min_abs=0.75,
            max_abs=5.0,
        )

        self.activation = SwooshL()
        self.pointwise_conv2 = ScaledConv2d(
            in_channels=hidden_channels,
            out_channels=channels,
            kernel_size=1,
            initial_scale=0.01,
        )

        self.out_balancer = Balancer(
            channels,
            channel_dim=1,
            min_positive=0.4,
            max_positive=0.6,
            min_abs=1.0,
            max_abs=6.0,
        )
        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=5.0,
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return self.forward_internal(x)
        layerdrop_rate = float(self.layerdrop_rate)

        if layerdrop_rate != 0.0:
            batch_size = x.shape[0]
            mask = (
                torch.rand((batch_size, 1, 1, 1), dtype=x.dtype, device=x.device)
                > layerdrop_rate
            )
        else:
            mask = None
        # turns out this caching idea does not work with --world-size > 1
        # return caching_eval(self.forward_internal, x, mask)
        return self.forward_internal(x, mask)

    def forward_internal(
        self, x: Tensor, layer_skip_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        x layout: (N, C, H, W), i.e. (batch_size, num_channels, num_frames, num_freqs)

        The returned value has the same shape as x.
        """
        bypass = x
        x = self.depthwise_conv(x)
        x = self.pointwise_conv1(x)
        x = self.hidden_balancer(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        if layer_skip_mask is not None:
            x = x * layer_skip_mask

        x = bypass + x
        x = self.out_balancer(x)

        if x.requires_grad:
            x = x.transpose(1, 3)  # (N, W, H, C); need channel dim to be last
            x = self.out_whiten(x)
            x = x.transpose(1, 3)  # (N, C, H, W)

        return x

    def streaming_forward(
        self,
        x: Tensor,
        cached_left_pad: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x layout: (N, C, H, W), i.e. (batch_size, num_channels, num_frames, num_freqs)
            cached_left_pad: (batch_size, num_channels, left_pad, num_freqs)

        Returns:
            - The returned value has the same shape as x.
            - Updated cached_left_pad.
        """
        padding = self.padding

        # The length without right padding for depth-wise conv
        T = x.size(2) - padding[0]

        bypass = x[:, :, :T, :]

        # Pad left side
        assert cached_left_pad.size(2) == padding[0], (
            cached_left_pad.size(2),
            padding[0],
        )
        x = torch.cat([cached_left_pad, x], dim=2)
        # Update cached left padding
        cached_left_pad = x[:, :, T : padding[0] + T, :]

        # depthwise_conv
        x = torch.nn.functional.conv2d(
            x,
            weight=self.depthwise_conv.weight,
            bias=self.depthwise_conv.bias,
            padding=(0, padding[1]),
            groups=self.depthwise_conv.groups,
        )
        x = self.pointwise_conv1(x)
        x = self.hidden_balancer(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)

        x = bypass + x
        return x, cached_left_pad

# class Conv2dSubsampling(nn.Module):
#     """Convolutional 2D subsampling (to 1/2 length).

#     Convert an input of shape (N, T, idim) to an output
#     with shape (N, T', odim), where
#     T' = (T-3)//2 - 2 == (T-7)//2

#     It is based on
#     https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py  # noqa
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         layer1_channels: int = 8,
#         layer2_channels: int = 32,
#         layer3_channels: int = 128,
#         dropout: FloatLike = 0.1,
#     ) -> None:
#         """
#         Args:
#           in_channels:
#             Number of channels in. The input shape is (N, T, in_channels).
#             Caution: It requires: T >=7, in_channels >=7
#           out_channels
#             Output dim. The output shape is (N, (T-3)//2, out_channels)
#           layer1_channels:
#             Number of channels in layer1
#           layer1_channels:
#             Number of channels in layer2
#           bottleneck:
#             bottleneck dimension for 1d squeeze-excite
#         """
#         assert in_channels >= 7
#         super().__init__()

#         # The ScaleGrad module is there to prevent the gradients
#         # w.r.t. the weight or bias of the first Conv2d module in self.conv from
#         # exceeding the range of fp16 when using automatic mixed precision (amp)
#         # training.  (The second one is necessary to stop its bias from getting
#         # a too-large gradient).

#         self.conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=layer1_channels,
#                 kernel_size=3,
#                 padding=(0, 1),  # (time, freq)
#             ),
#             ScaleGrad(0.2),
#             Balancer(layer1_channels, channel_dim=1, max_abs=1.0),
#             SwooshR(),
#             nn.Conv2d(
#                 in_channels=layer1_channels,
#                 out_channels=layer2_channels,
#                 kernel_size=3,
#                 stride=2,
#                 padding=0,
#             ),
#             Balancer(layer2_channels, channel_dim=1, max_abs=4.0),
#             SwooshR(),
#             nn.Conv2d(
#                 in_channels=layer2_channels,
#                 out_channels=layer3_channels,
#                 kernel_size=3,
#                 stride=(1, 2),  # (time, freq)
#             ),
#             Balancer(layer3_channels, channel_dim=1, max_abs=4.0),
#             SwooshR(),
#         )

#         # just one convnext layer
#         self.convnext = ConvNeXt(layer3_channels, kernel_size=(7, 7))

#         # (in_channels-3)//4
#         self.out_width = (((in_channels - 1) // 2) - 1) // 2
#         self.layer3_channels = layer3_channels

#         self.out = nn.Linear(self.out_width * layer3_channels, out_channels)
#         # use a larger than normal grad_scale on this whitening module; there is
#         # only one such module, so there is not a concern about adding together
#         # many copies of this extra gradient term.
#         self.out_whiten = Whiten(
#             num_groups=1,
#             whitening_limit=ScheduledFloat((0.0, 4.0), (20000.0, 8.0), default=4.0),
#             prob=(0.025, 0.25),
#             grad_scale=0.02,
#         )

#         # max_log_eps=0.0 is to prevent both eps and the output of self.out from
#         # getting large, there is an unnecessary degree of freedom.
#         self.out_norm = BiasNorm(out_channels)
#         self.dropout = Dropout3(dropout, shared_dim=1)

#     def forward(
#         self, x: torch.Tensor, x_lens: torch.Tensor
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Subsample x.

#         Args:
#           x:
#             Its shape is (N, T, idim).
#           x_lens:
#             A tensor of shape (batch_size,) containing the number of frames in

#         Returns:
#           - a tensor of shape (N, (T-7)//2, odim)
#           - output lengths, of shape (batch_size,)
#         """
#         # On entry, x is (N, T, idim)
#         x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)
#         # scaling x by 0.1 allows us to use a larger grad-scale in fp16 "amp" (automatic mixed precision)
#         # training, since the weights in the first convolution are otherwise the limiting factor for getting infinite
#         # gradients.
#         x = self.conv(x)
#         x = self.convnext(x)

#         # Now x is of shape (N, odim, (T-7)//2, (idim-3)//4)
#         b, c, t, f = x.size()

#         x = x.transpose(1, 2).reshape(b, t, c * f)
#         # now x: (N, (T-7)//2, out_width * layer3_channels))

#         x = self.out(x)
#         # Now x is of shape (N, (T-7)//2, odim)
#         x = self.out_whiten(x)
#         x = self.out_norm(x)
#         x = self.dropout(x)

#         if torch.jit.is_scripting() or torch.jit.is_tracing():
#             x_lens = (x_lens - 7) // 2
#         else:
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 x_lens = (x_lens - 7) // 2
#         assert x.size(1) == x_lens.max().item(), (x.size(1), x_lens.max())

#         return x, x_lens

#     def streaming_forward(
#         self,
#         x: torch.Tensor,
#         x_lens: torch.Tensor,
#         cached_left_pad: Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Subsample x.

#         Args:
#           x:
#             Its shape is (N, T, idim).
#           x_lens:
#             A tensor of shape (batch_size,) containing the number of frames in

#         Returns:
#           - a tensor of shape (N, (T-7)//2, odim)
#           - output lengths, of shape (batch_size,)
#           - updated cache
#         """
#         # On entry, x is (N, T, idim)
#         x = x.unsqueeze(1)  # (N, T, idim) -> (N, 1, T, idim) i.e., (N, C, H, W)

#         # T' = (T-7)//2
#         x = self.conv(x)

#         # T' = (T-7)//2-3
#         x, cached_left_pad = self.convnext.streaming_forward(
#             x, cached_left_pad=cached_left_pad
#         )

#         # Now x is of shape (N, odim, T', ((idim-1)//2 - 1)//2)
#         b, c, t, f = x.size()

#         x = x.transpose(1, 2).reshape(b, t, c * f)
#         # now x: (N, T', out_width * layer3_channels))

#         x = self.out(x)
#         # Now x is of shape (N, T', odim)
#         x = self.out_norm(x)

#         if torch.jit.is_scripting() or torch.jit.is_tracing():
#             assert self.convnext.padding[0] == 3
#             # The ConvNeXt module needs 3 frames of right padding after subsampling
#             x_lens = (x_lens - 7) // 2 - 3
#         else:
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 # The ConvNeXt module needs 3 frames of right padding after subsampling
#                 assert self.convnext.padding[0] == 3
#                 x_lens = (x_lens - 7) // 2 - 3

#         assert x.size(1) == x_lens.max().item(), (x.shape, x_lens.max())

#         return x, x_lens, cached_left_pad

#     @torch.jit.export
#     def get_init_states(
#         self,
#         batch_size: int = 1,
#         device: torch.device = torch.device("cpu"),
#     ) -> Tensor:
#         """Get initial states for Conv2dSubsampling module.
#         It is the cached left padding for ConvNeXt module,
#         of shape (batch_size, num_channels, left_pad, num_freqs)
#         """
#         left_pad = self.convnext.padding[0]
#         freq = self.out_width
#         channels = self.layer3_channels
#         cached_embed_left_pad = torch.zeros(batch_size, channels, left_pad, freq).to(
#             device
#         )

#         return cached_embed_left_pad


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        residual: bool = False,
        conv_module: Type[nn.Module] = nn.Conv2d,
        activation: Callable = nn.LeakyReLU,  # 👉 Dùng LeakyReLU
        norm: Optional[Type[nn.Module]] = nn.BatchNorm2d,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            conv_stride = stride if i == num_layers - 1 else 1
            conv = conv_module(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=conv_stride,
                dilation=dilation,
                padding=(kernel_size // 2)
            )
            layers.append(conv)
            if norm:
                layers.append(norm(out_channels))  # Gọi instance
            layers.append(activation())
            layers.append(nn.Dropout(dropout))

        self.main = nn.Sequential(*layers)
        self.residual = residual

        if residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                norm(out_channels) if norm else nn.Identity(),
                nn.Dropout(dropout),
            )
        elif residual:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x, mask):
        B, C, T, F = x.shape
        residual_input = x  

        for layer in self.main:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                k = layer.kernel_size[0]
                s = layer.stride[0]
                d = layer.dilation[0]
                p = layer.padding[0]
                out_T = (T + 2 * p - d * (k - 1) - 1) // s + 1
                pad_len = T - mask.sum(dim=1)
                data_len = mask.sum(dim=1)
                new_len = calc_data_len(
                    result_len=out_T,
                    pad_len=pad_len,
                    data_len=data_len,
                    kernel_size=k,
                    stride=s,
                )
                mask = get_mask_from_lens(new_len, out_T)
                T = out_T

        if self.residual:
            shortcut = self.shortcut(residual_input)  # 👉 fix chỗ này
            x = x + shortcut

        return x, mask
        
class Conv2dSubsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_blocks: int,
        num_layers_per_block: int,
        out_channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        residuals: List[bool],
        activation: Callable = nn.LeakyReLU, 
        norm: Optional[Callable] = nn.BatchNorm2d, 
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []

        for i in range(num_blocks):
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels[i],
                num_layers=num_layers_per_block,
                kernel_size=kernel_sizes[i],
                stride=strides[i],
                residual=residuals[i],
                activation=activation,
                norm=norm,
                dropout=dropout
            )
            blocks.append(block)
            in_channels = out_channels[i]

        self.model = nn.ModuleList(blocks)

    def forward(self, x, mask):
        for i, block in enumerate(self.model):
            x, mask = block(x, mask)
        return x, mask

def calc_data_len(
    result_len: int,
    pad_len,
    data_len,
    kernel_size: int,
    stride: int,
):
    """Calculates the new data portion size after applying convolution on a padded tensor

    Args:

        result_len (int): The length after the convolution is applied.

        pad_len Union[Tensor, int]: The original padding portion length.

        data_len Union[Tensor, int]: The original data portion legnth.

        kernel_size (int): The convolution kernel size.

        stride (int): The convolution stride.

    Returns:

        Union[Tensor, int]: The new data portion length.

    """
    if type(pad_len) != type(data_len):
        raise ValueError(
            f"""expected both pad_len and data_len to be of the same type
            but {type(pad_len)}, and {type(data_len)} passed"""
        )
    inp_len = data_len + pad_len
    new_pad_len = 0
    # if padding size less than the kernel size
    # then it will be convolved with the data.
    convolved_pad_mask = pad_len >= kernel_size
    # calculating the size of the discarded items (not convolved)
    unconvolved = (inp_len - kernel_size) % stride
    undiscarded_pad_mask = unconvolved < pad_len
    convolved = pad_len - unconvolved
    new_pad_len = (convolved - kernel_size) // stride + 1
    # setting any condition violation to zeros using masks
    new_pad_len *= convolved_pad_mask
    new_pad_len *= undiscarded_pad_mask
    return result_len - new_pad_len

def get_mask_from_lens(lengths, max_len: int):
    """Creates a mask tensor from lengths tensor.

    Args:
        lengths (Tensor): The lengths of the original tensors of shape [B].

        max_len (int): the maximum lengths.

    Returns:
        Tensor: The mask of shape [B, max_len] and True whenever the index in the data portion.
    """
    indices = torch.arange(max_len).to(lengths.device)
    indices = indices.expand(len(lengths), max_len)
    return indices < lengths.unsqueeze(dim=1)