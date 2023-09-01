"""Torch modules for diarization."""

# All the code in this file is taken from NVIDIA's NeMo project:
# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/jasper.py
# or
# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/tdnn_attention.py

import math
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger as logging
from numpy import inf
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from torch.nn.init import _calculate_correct_fan

try:
    from pytorch_quantization import nn as quant_nn

    PYTORCH_QUANTIZATION_AVAILABLE = True
except ImportError:
    PYTORCH_QUANTIZATION_AVAILABLE = False


def compute_new_kernel_size(kernel_size, kernel_width):
    """Compute new kernel size based on kernel width."""
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


def get_asymtric_padding(kernel_size, stride, dilation, future_context):
    """Asymmetric padding for future context."""
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")

    left_context = kernel_size - 1 - future_context
    right_context = future_context

    symmetric_padding = get_same_padding(kernel_size, stride, dilation)

    if kernel_size <= future_context:
        # kernel size is smaller than future context, equivalent to using entire context of kernel
        # simply return symmetric padding for this scenario
        logging.warning(
            "Future context window is larger than the kernel size!\nLeft context ="
            f" {left_context} | Right context = greater than {right_context} | Kernel"
            f" size = {kernel_size}\nSwitching to symmetric padding (left context ="
            f" right context = {symmetric_padding})"
        )
        return symmetric_padding

    if left_context < symmetric_padding:
        logging.warning(
            "Future context window is larger than half the kernel size!\nConv layer"
            " therefore uses more future information than past to compute its"
            f" output!\nLeft context = {left_context} | Right context ="
            f" {right_context} | Kernel size = {kernel_size}"
        )

    if dilation > 1:
        left_context = dilation * kernel_size - 1 - dilation * future_context
        right_context = dilation * future_context
        return (left_context, right_context)

    return (left_context, right_context)


def get_same_padding(kernel_size, stride, dilation) -> int:
    """Same padding calculation."""
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (dilation * (kernel_size - 1)) // 2


def get_statistics_with_mask(
    x: torch.Tensor, m: torch.Tensor, dim: int = 2, eps: float = 1e-10
):
    """
    Compute mean and standard deviation of input(x) provided with its masking labels (m).

    input:
        x: feature input
        m: averaged mask labels
    output:
        mean: mean of input features
        std: stadard deviation of input features
    """
    mean = torch.sum((m * x), dim=dim)
    std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
    return mean, std


def init_weights(m, mode: Optional[str] = "xavier_uniform"):  # noqa: C901
    """Initialize weights of the model."""
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode is not None:
            if mode == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif mode == "xavier_normal":
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif mode == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif mode == "tds_uniform":
                tds_uniform_(m.weight)
            elif mode == "tds_normal":
                tds_normal_(m.weight)
            else:
                raise ValueError(f"Unknown Initialization mode: {mode}")
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def lens_to_mask(lens: List[int], max_len: int, device: str = None):
    """Outputs masking labels.

    Outputs masking labels for list of lengths of audio features, with max length of any
    mask as max_len.

    input:
        lens: list of lens
        max_len: max length of any audio feature
    output:
        mask: masked labels
        num_values: sum of mask values for each feature (useful for computing statistics later)
    """
    lens_mat = torch.arange(max_len).to(device)
    mask = lens_mat[:max_len].unsqueeze(0) < lens.unsqueeze(1)
    mask = mask.unsqueeze(1)
    num_values = torch.sum(mask, dim=2, keepdim=True)
    return mask, num_values


@torch.jit.script
def _masked_conv_init_lens(
    lens: torch.Tensor, current_maxlen: int, original_maxlen: torch.Tensor
):
    """Masked Conv Init Lens."""
    if current_maxlen > original_maxlen:
        new_lens = torch.arange(current_maxlen)
        new_max_lens = torch.tensor(current_maxlen)
    else:
        new_lens = lens
        new_max_lens = original_maxlen
    return new_lens, new_max_lens


@torch.jit.script_if_tracing
def make_seq_mask_like(
    like: torch.Tensor,
    lengths: torch.Tensor,
    valid_ones: bool = True,
    time_dim: int = -1,
) -> torch.Tensor:
    """Make sequence mask like."""
    mask = (
        torch.arange(like.shape[time_dim], device=like.device)
        .repeat(lengths.shape[0], 1)
        .lt(lengths.unsqueeze(-1))
    )
    # Match number of dims in `like` tensor
    for _ in range(like.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    # If time dim != -1, transpose to proper dim.
    if time_dim != -1:
        mask = mask.transpose(time_dim, -1)
    if not valid_ones:
        mask = ~mask
    return mask


@torch.jit.script
def _se_pool_step_script_infer(
    x: torch.Tensor, context_window: int, mask: torch.Tensor
):
    """
    Calculates the masked average over padded limited context segment during inference mode.

    Args:
        x: Input tensor. Shape = [B, C, T]
        context_window: Integer context window, must be 0 or greater.
        mask: Mask tensor, 1 represents value index, 0 represents padded index. Shape = [B, 1, T].

    Returns:
        A tensor reduced via masked average pool over some limited context. Shape = [B, C, 1]
    """
    timesteps = x.shape[-1]
    if timesteps < context_window:
        y = torch.sum(x, dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).to(
            x.dtype
        )
    else:
        # << During inference prefer to use entire context >>
        # x = x[:, :, :context_window]  # [B, C, context_window]
        # mask = mask[:, :, :context_window]  # [B, 1, context_window]
        #
        # mask = mask.sum(dim=-1, keepdim=True).to(x.dtype)  # [B, C, 1]
        # y = x.sum(dim=-1, keepdim=True)  # [B, 1, 1]
        # y = y / (mask + 1e-8)  # [B, C, 1]
        y = torch.sum(x, dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).to(
            x.dtype
        )

    return y


@torch.jit.script
def _se_pool_step_script_train(
    x: torch.Tensor, context_window: int, mask: torch.Tensor
):
    """Calculates the masked average over padded limited context segment during training mode.

    Randomly slices a segment of length `context_window` from signal+padded input tensor across all channels and
    uses it for computing masked limited context.

    Args:
        x: Input tensor. Shape = [B, C, T]
        context_window: Integer context window, must be 0 or greater.
        mask: Mask tensor, 1 represents value index, 0 represents padded index. Shape = [B, 1, T].

    Returns:
        A tensor reduced via masked average pool over some limited context. Shape = [B, C, 1]
    """
    timesteps = x.shape[-1]
    if timesteps < context_window:
        y = torch.sum(x, dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).to(
            x.dtype
        )
    else:
        start_idx = torch.randint(
            0, timesteps - context_window, size=[1], dtype=torch.int32
        )[0]
        x = x[:, :, start_idx : (start_idx + context_window)]  # [B, C, context_window]
        mask = mask[
            :, :, start_idx : (start_idx + context_window)
        ]  # [B, 1, context_window]

        mask = mask.sum(dim=-1, keepdim=True).to(x.dtype)  # [B, C, 1]
        y = x.sum(dim=-1, keepdim=True)  # [B, 1, 1]
        y = y / (mask + 1e-8)  # [B, C, 1]

    return y


def tds_uniform_(tensor, mode="fan_in"):
    r"""TDS Uniform Initialization.

    Uniform Initialization from the paper
    Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions:
    https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf
    Normalized to -
    .. math::
        \\text{bound} = \\text{2} \\times \\sqrt{\\frac{1}{\\text{fan\\_mode}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = 2.0  # sqrt(4.0) = 2
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def tds_normal_(tensor, mode="fan_in"):
    r"""TDS Normal.

    Normal Initialization from the paper
    Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions
    https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2460.pdf
    Normalized to -
    .. math::
        \\text{bound} = \\text{2} \\times \\sqrt{\\frac{1}{\\text{fan\\_mode}}}

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = 2.0
    std = gain / math.sqrt(fan)  # sqrt(4.0 / fan_in)
    bound = std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.normal_(0.0, bound)


class MaskedConv1d(nn.Module):
    """Masked Convolution Module."""

    __constants__ = ["use_conv_mask", "real_out_channels", "heads"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        heads=-1,
        bias=False,
        use_mask=True,
        quantize=False,
    ):
        """Initialize the module."""
        super().__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        # preserve original padding
        self._padding = padding

        # if padding is a tuple/list, it is considered as asymmetric padding
        if type(padding) in (tuple, list):
            self.pad_layer = nn.ConstantPad1d(padding, value=0.0)
            # reset padding for conv since pad_layer will handle this
            padding = 0
        else:
            self.pad_layer = None

        if PYTORCH_QUANTIZATION_AVAILABLE and quantize:
            self.conv = quant_nn.QuantConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        elif not PYTORCH_QUANTIZATION_AVAILABLE and quantize:
            raise ImportError(
                "pytorch-quantization is not installed. Install from "
                "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
            )
        else:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        self.use_mask = use_mask
        self.heads = heads

        # Calculations for "same" padding cache
        self.same_padding = (self.conv.stride[0] == 1) and (
            2 * self.conv.padding[0]
            == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
        )
        if self.pad_layer is None:
            self.same_padding_asymmetric = False
        else:
            self.same_padding_asymmetric = (self.conv.stride[0] == 1) and (
                sum(self._padding)
                == self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
            )

        # `self.lens` caches consecutive integers from 0 to `self.max_len` that are used to compute the mask for a
        # batch. Recomputed to bigger size as needed. Stored on a device of the latest batch lens.
        if self.use_mask:
            self.max_len = torch.tensor(0)
            self.lens = torch.tensor(0)

    def get_seq_len(self, lens):
        """Get sequence length after convolution."""
        if self.same_padding or self.same_padding_asymmetric:
            return lens

        if self.pad_layer is None:
            return (
                torch.div(
                    lens
                    + 2 * self.conv.padding[0]
                    - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
                    - 1,
                    self.conv.stride[0],
                    rounding_mode="trunc",
                )
                + 1
            )
        else:
            return (
                torch.div(
                    lens
                    + sum(self._padding)
                    - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1)
                    - 1,
                    self.conv.stride[0],
                    rounding_mode="trunc",
                )
                + 1
            )

    def forward(self, x, lens):
        """Forward pass."""
        if self.use_mask:
            # Generally will be called by ConvASREncoder, but kept as single gpu backup.
            if x.size(2) > self.max_len:
                self.update_masked_length(x.size(2), device=lens.device)
            x = self.mask_input(x, lens)

        # Update lengths
        lens = self.get_seq_len(lens)

        # asymmtric pad if necessary
        if self.pad_layer is not None:
            x = self.pad_layer(x)

        sh = x.shape
        if self.heads != -1:
            x = x.view(-1, self.heads, sh[-1])

        out = self.conv(x)

        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)

        return out, lens

    def update_masked_length(self, max_len, seq_range=None, device=None):
        """Updates the cached masked length."""
        if seq_range is None:
            self.lens, self.max_len = _masked_conv_init_lens(
                self.lens, max_len, self.max_len
            )
            self.lens = self.lens.to(device)
        else:
            self.lens = seq_range
            self.max_len = max_len

    def mask_input(self, x, lens):
        """Mask input."""
        max_len = x.size(2)
        mask = self.lens[:max_len].unsqueeze(0).to(lens.device) < lens.unsqueeze(1)
        x = x * mask.unsqueeze(1).to(device=x.device)
        return x


class GroupShuffle(nn.Module):
    """Group Shuffle Module."""

    def __init__(self, groups, channels):
        """Initialize the module."""
        super().__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        """Forward pass."""
        sh = x.shape

        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])

        return x


class SqueezeExcite(nn.Module):
    """Squeeze Excite Module."""

    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        context_window: int = -1,
        interpolation_mode: str = "nearest",
        activation: Optional[Callable] = None,
        quantize: bool = False,
    ):
        """
        Squeeze-and-Excitation sub-module.

        Args:
            channels: Input number of channels.
            reduction_ratio: Reduction ratio for "squeeze" layer.
            context_window: Integer number of timesteps that the context
                should be computed over, using stride 1 average pooling.
                If value < 1, then global context is computed.
            interpolation_mode: Interpolation mode of timestep dimension.
                Used only if context window is > 1.
                The modes available for resizing are: `nearest`, `linear` (3D-only),
                `bilinear`, `area`
            activation: Intermediate activation function used. Must be a
                callable activation function.
            quantize: Wether to quantize or not.
        """
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self._quantize = quantize

        self.pool = None  # prepare a placeholder which will be updated

        if activation is None:
            activation = nn.ReLU(inplace=True)

        if PYTORCH_QUANTIZATION_AVAILABLE and quantize:
            self.fc = nn.Sequential(
                quant_nn.QuantLinear(channels, channels // reduction_ratio, bias=False),
                activation,
                quant_nn.QuantLinear(channels // reduction_ratio, channels, bias=False),
            )
        elif not PYTORCH_QUANTIZATION_AVAILABLE and quantize:
            raise ImportError(
                "pytorch-quantization is not installed. Install from "
                "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction_ratio, bias=False),
                activation,
                nn.Linear(channels // reduction_ratio, channels, bias=False),
            )
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Set default context window
        self.change_context_window(context_window=context_window)

        # Set default max sequence length
        self.set_max_len(16)

    def forward(self, x, lengths):
        """Forward pass."""
        return self.forward_for_export(x, lengths)

    def forward_for_export(self, x, lengths):
        """Forward pass for export."""
        # The use of negative indices on the transpose allow for expanded SqueezeExcite
        max_len = x.shape[-1]
        if max_len > self.max_len:
            self.set_max_len(max_len)

        # Computes in float32 to avoid instabilities during training with AMP.
        with torch.cuda.amp.autocast(enabled=False):
            # Create sample mask - 1 represents value, 0 represents pad
            mask = self.make_pad_mask(
                lengths, max_audio_length=max_len, device=x.device
            )
            mask = ~mask  # 0 represents value, 1 represents pad
            x = x.float()  # For stable AMP, SE must be computed at fp32.
            x.masked_fill_(mask, 0.0)  # mask padded values explicitly to 0
            y = self._se_pool_step(x, mask)  # [B, C, 1]
            y = y.transpose(1, -1)  # [B, 1, C]
            y = self.fc(y)  # [B, 1, C]
            y = y.transpose(1, -1)  # [B, C, 1]

            # Note: Keep for future, in case we improve WER from doing so.
            # if self.context_window >= 0:
            #     y = F.interpolate(y, size=x.shape[-1], mode=self.interpolation_mode)

            y = torch.sigmoid(y)
            y = x * y
        return y, lengths

    def _se_pool_step(self, x, mask):
        """Performs the Squeeze and Excitation pooling step."""
        # Negate mask back to represent 1 for signal and 0 for padded timestep.
        mask = ~mask

        if self.context_window < 0:
            # [B, C, 1] - Masked Average over value + padding.
            y = torch.sum(x, dim=-1, keepdim=True) / mask.sum(
                dim=-1, keepdim=True
            ).type(x.dtype)
        else:
            # [B, C, 1] - Masked Average over value + padding with limited context.
            # During training randomly subsegments a context_window chunk of timesteps.
            # During inference selects only the first context_window chunk of timesteps.
            if self.training:
                y = _se_pool_step_script_train(x, self.context_window, mask)
            else:
                y = _se_pool_step_script_infer(x, self.context_window, mask)
        return y

    def set_max_len(self, max_len, seq_range=None):
        """Sets maximum input length.

        Pre-calculates internal seq_range mask.
        """
        self.max_len = max_len
        if seq_range is None:
            device = next(self.parameters()).device
            seq_range = torch.arange(0, self.max_len, device=device)
        if hasattr(self, "seq_range"):
            self.seq_range = seq_range
        else:
            self.register_buffer("seq_range", seq_range, persistent=False)

    def make_pad_mask(self, seq_lens, max_audio_length, device=None):
        """Make masking for padding."""
        if device and self.seq_range.device != device:
            self.seq_range = self.seq_range.to(device)
        if self.seq_range.device != seq_lens.device:
            seq_lens = seq_lens.to(self.seq_range.device)

        mask = self.seq_range[:max_audio_length].expand(
            seq_lens.size(0), -1
        ) < seq_lens.unsqueeze(
            -1
        )  # [B, T]; bool
        mask = mask.unsqueeze(1)  # [B, 1, T]

        return mask

    def change_context_window(self, context_window: int):
        """Update the context window of the SqueezeExcitation module, in-place if possible.

        Will update the pooling layer to either nn.AdaptiveAvgPool1d() (for global SE) or nn.AvgPool1d()
        (for limited context SE).
        If only the context window is changing but still a limited SE context block - then
        the earlier instance of nn.AvgPool1d() will be updated.

        Args:
            context_window: An integer representing the number of input timeframes that will be used
                to compute the context. Each timeframe corresponds to a single window stride of the
                STFT features.

                Say the window_stride = 0.01s, then a context window of 128 represents 128 * 0.01 s
                of context to compute the Squeeze step.
        """
        if hasattr(self, "context_window"):
            logging.info(
                "Changing Squeeze-Excitation context window from"
                f" {self.context_window} to {context_window}"
            )

        self.context_window = context_window


class JasperBlock(nn.Module):
    """Jasper Block Module.

    Constructs a single "Jasper" block. With modified parameters, also constructs other blocks for models
    such as `QuartzNet` and `Citrinet`.

    Args:
        inplanes: Number of input channels.
        planes: Number of output channels.
        repeat: Number of repeated sub-blocks (R) for this block.
        kernel_size: Convolution kernel size across all repeated sub-blocks.
        kernel_size_factor: Floating point scale value that is multiplied with kernel size,
            then rounded down to nearest odd integer to compose the kernel size. Defaults to 1.0.
        stride: Stride of the convolutional layers.
        dilation: Integer which defined dilation factor of kernel. Note that when dilation > 1, stride must
            be equal to 1.
        padding: String representing type of padding. Currently only supports "same" padding,
            which symmetrically pads the input tensor with zeros.
        dropout: Floating point value, determins percentage of output that is zeroed out.
        activation: String representing activation functions. Valid activation functions are :
            {"hardtanh": nn.Hardtanh, "relu": nn.ReLU, "selu": nn.SELU, "swish": Swish}.
            Defaults to "relu".
        residual: Bool that determined whether a residual branch should be added or not.
            All residual branches are constructed using a pointwise convolution kernel, that may or may not
            perform strided convolution depending on the parameter `residual_mode`.
        groups: Number of groups for Grouped Convolutions. Defaults to 1.
        separable: Bool flag that describes whether Time-Channel depthwise separable convolution should be
            constructed, or ordinary convolution should be constructed.
        heads: Number of "heads" for the masked convolution. Defaults to -1, which disables it.
        normalization: String that represents type of normalization performed. Can be one of
            "batch", "group", "instance" or "layer" to compute BatchNorm1D, GroupNorm1D, InstanceNorm or
            LayerNorm (which are special cases of GroupNorm1D).
        norm_groups: Number of groups used for GroupNorm (if `normalization` == "group").
        residual_mode: String argument which describes whether the residual branch should be simply
            added ("add") or should first stride, then add ("stride_add"). Required when performing stride on
            parallel branch as well as utilizing residual add.
        residual_panes: Number of residual panes, used for Jasper-DR models. Please refer to the paper.
        conv_mask: Bool flag which determines whether to utilize masked convolutions or not. In general,
            it should be set to True.
        se: Bool flag that determines whether Squeeze-and-Excitation layer should be used.
        se_reduction_ratio: Integer value, which determines to what extend the hidden dimension of the SE
            intermediate step should be reduced. Larger values reduce number of parameters, but also limit
            the effectiveness of SE layers.
        se_context_window: Integer value determining the number of timesteps that should be utilized in order
            to compute the averaged context window. Defaults to -1, which means it uses global context - such
            that all timesteps are averaged. If any positive integer is used, it will utilize limited context
            window of that size.
        se_interpolation_mode: String used for interpolation mode of timestep dimension for SE blocks.
            Used only if context window is > 1.
            The modes available for resizing are: `nearest`, `linear` (3D-only),
            `bilinear`, `area`.
        stride_last: Bool flag that determines whether all repeated blocks should stride at once,
            (stride of S^R when this flag is False) or just the last repeated block should stride
            (stride of S when this flag is True).
        future_context: Int value that determins how many "right" / "future" context frames will be utilized
            when calculating the output of the conv kernel. All calculations are done for odd kernel sizes only.
        quantize: Bool flag whether to quantize the Convolutional blocks.
        layer_idx (int, optional): can be specified to allow layer output capture for InterCTC loss. Defaults to -1.
    """

    __constants__ = ["conv_mask", "separable", "residual_mode", "res", "mconv"]

    def __init__(  # noqa: C901
        self,
        inplanes,
        planes,
        repeat=3,
        kernel_size=11,
        kernel_size_factor=1,
        stride=1,
        dilation=1,
        padding="same",
        dropout=0.2,
        activation=None,
        residual=True,
        groups=1,
        separable=False,
        heads=-1,
        normalization="batch",
        norm_groups=1,
        residual_mode="add",
        residual_panes=[],  # noqa: B006
        conv_mask=False,
        se=False,
        se_reduction_ratio=16,
        se_context_window=-1,
        se_interpolation_mode="nearest",
        stride_last=False,
        future_context: int = -1,
        quantize=False,
        layer_idx: int = -1,  # only used for capturing tensors for interctc loss
    ):
        """Constructs a Jasper Block."""
        super().__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        kernel_size_factor = float(kernel_size_factor)
        if isinstance(kernel_size, Iterable):
            kernel_size = [
                compute_new_kernel_size(k, kernel_size_factor) for k in kernel_size
            ]
        else:
            kernel_size = [compute_new_kernel_size(kernel_size, kernel_size_factor)]

        if future_context < 0:
            padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        else:
            padding_val = get_asymtric_padding(
                kernel_size[0], stride[0], dilation[0], future_context
            )

        self.inplanes = inplanes
        self.planes = planes
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode
        self.se = se
        self.quantize = quantize
        self.layer_idx = layer_idx
        # will be set in self.forward() if defined in AccessMixin config
        self.interctc_should_capture = None

        inplanes_loop = inplanes
        conv = nn.ModuleList()

        for _ in range(repeat - 1):
            # Stride last means only the last convolution in block will have stride
            if stride_last:
                stride_val = [1]
            else:
                stride_val = stride

            conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride_val,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    heads=heads,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups,
                    quantize=quantize,
                )
            )

            conv.extend(
                self._get_act_dropout_layer(drop_prob=dropout, activation=activation)
            )

            inplanes_loop = planes

        conv.extend(
            self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                groups=groups,
                heads=heads,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups,
                quantize=quantize,
            )
        )

        if se:
            conv.append(
                SqueezeExcite(
                    planes,
                    reduction_ratio=se_reduction_ratio,
                    context_window=se_context_window,
                    interpolation_mode=se_interpolation_mode,
                    activation=activation,
                    quantize=quantize,
                )
            )

        self.mconv = conv

        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            res_list = nn.ModuleList()

            if residual_mode == "stride_add":
                stride_val = stride
            else:
                stride_val = [1]

            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res = nn.ModuleList(
                    self._get_conv_bn_layer(
                        ip,
                        planes,
                        kernel_size=1,
                        normalization=normalization,
                        norm_groups=norm_groups,
                        stride=stride_val,
                        quantize=quantize,
                    )
                )

                res_list.append(res)

            self.res = res_list
            if PYTORCH_QUANTIZATION_AVAILABLE and self.quantize:
                self.residual_quantizer = quant_nn.TensorQuantizer(
                    quant_nn.QuantConv2d.default_quant_desc_input
                )
            elif not PYTORCH_QUANTIZATION_AVAILABLE and quantize:
                raise ImportError(
                    "pytorch-quantization is not installed. Install from "
                    "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
                )
        else:
            self.res = None

        self.mout = nn.Sequential(
            *self._get_act_dropout_layer(drop_prob=dropout, activation=activation)
        )

    def _get_conv(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
        quantize=False,
    ):
        """Get convolution layer."""
        use_mask = self.conv_mask
        if use_mask:
            return MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
                heads=heads,
                use_mask=use_mask,
                quantize=quantize,
            )
        else:
            if PYTORCH_QUANTIZATION_AVAILABLE and quantize:
                return quant_nn.QuantConv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                )
            elif not PYTORCH_QUANTIZATION_AVAILABLE and quantize:
                raise ImportError(
                    "pytorch-quantization is not installed. Install from "
                    "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
                )
            else:
                return nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                )

    def _get_conv_bn_layer(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
        normalization="batch",
        norm_groups=1,
        quantize=False,
    ):
        """Get the Conv layer."""
        if norm_groups == -1:
            norm_groups = out_channels

        if separable:
            layers = [
                self._get_conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=in_channels,
                    heads=heads,
                    quantize=quantize,
                ),
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=bias,
                    groups=groups,
                    quantize=quantize,
                ),
            ]
        else:
            layers = [
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                    quantize=quantize,
                )
            ]

        if normalization == "group":
            layers.append(
                nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
            )
        elif normalization == "instance":
            layers.append(
                nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)
            )
        elif normalization == "layer":
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        elif normalization == "batch":
            layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
        else:
            raise ValueError(
                f"Normalization method ({normalization}) does not match"
                " one of [batch, layer, group, instance]."
            )

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        """Define activation and dropout layers."""
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [activation, nn.Dropout(p=drop_prob)]

        return layers

    def forward(  # noqa: C901
        self, input_tuple: Tuple[List[Tensor], Optional[Tensor]]
    ) -> Tuple[List[Tensor], Optional[Tensor]]:
        """
        Forward pass of the module.

        Args:
            input_tuple:
                The input is a tuple of two values - the preprocessed audio signal as well as the lengths
                of the audio signal. The audio signal is padded to the shape [B, D, T] and the lengths are
                a torch vector of length B.

        Returns:
            The output of the block after processing the input through `repeat` number of sub-blocks,
            as well as the lengths of the encoded audio after padding/striding.
        """
        lens_orig = None
        xs = input_tuple[0]
        if len(input_tuple) == 2:
            xs, lens_orig = input_tuple

        # compute forward convolutions
        out = xs[-1]

        lens = lens_orig
        for _, conv_len in enumerate(self.mconv):
            # if we're doing masked convolutions, we need to pass in and
            # possibly update the sequence lengths
            # if (i % 4) == 0 and self.conv_mask:
            if isinstance(conv_len, (MaskedConv1d, SqueezeExcite)):
                out, lens = conv_len(out, lens)
            else:
                out = conv_len(out)

        # compute the residuals
        if self.res is not None:
            for _i, layer in enumerate(self.res):
                res_out = xs[_i]
                for _, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, lens_orig)
                    else:
                        res_out = res_layer(res_out)

                if self.residual_mode == "add" or self.residual_mode == "stride_add":
                    if PYTORCH_QUANTIZATION_AVAILABLE and self.quantize:
                        out = self.residual_quantizer(out) + res_out
                    elif not PYTORCH_QUANTIZATION_AVAILABLE and self.quantize:
                        raise ImportError(
                            "pytorch-quantization is not installed. Install from "
                            "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
                        )
                    else:
                        out = out + res_out
                else:
                    out = torch.max(out, res_out)

        # compute the output
        out = self.mout(out)

        if self.res is not None and self.dense_residual:
            return xs + [out], lens

        return [out], lens


# https://github.com/NVIDIA/NeMo/blob/nemo/nemo/collections/asr/parts/submodules/tdnn_attention.py#L271
class AttentivePoolLayer(nn.Module):
    """Attention pooling layer for pooling speaker embeddings.

    Reference: ECAPA-TDNN Embeddings for Speaker Diarization (https://arxiv.org/pdf/2104.01466.pdf).

    inputs:
        inp_filters: input feature channel length from encoder
        attention_channels: intermediate attention channel size
        kernel_size: kernel_size for TDNN and attention conv1d layers (default: 1)
        dilation: dilation size for TDNN and attention conv1d layers  (default: 1)
    """

    def __init__(
        self,
        inp_filters: int,
        attention_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        eps: float = 1e-10,
    ):
        """Initialize Attentive Pooling Layer."""
        super().__init__()

        self.feat_in = 2 * inp_filters

        self.attention_layer = nn.Sequential(
            TDNNModule(
                inp_filters * 3,
                attention_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
            nn.Tanh(),
            nn.Conv1d(
                in_channels=attention_channels,
                out_channels=inp_filters,
                kernel_size=kernel_size,
                dilation=dilation,
            ),
        )
        self.eps = eps

    def forward(self, x, length=None):
        """Forward pass of Attentive Pooling Layer."""
        max_len = x.size(2)

        if length is None:
            length = torch.ones(x.shape[0], device=x.device)

        mask, num_values = lens_to_mask(length, max_len=max_len, device=x.device)

        # encoder statistics
        mean, std = get_statistics_with_mask(x, mask / num_values)
        mean = mean.unsqueeze(2).repeat(1, 1, max_len)
        std = std.unsqueeze(2).repeat(1, 1, max_len)
        attn = torch.cat([x, mean, std], dim=1)

        # attention statistics
        attn = self.attention_layer(attn)  # attention pass
        attn = attn.masked_fill(mask == 0, -inf)
        alpha = F.softmax(attn, dim=2)  # attention values, α
        mu, sg = get_statistics_with_mask(x, alpha)  # µ and ∑

        # gather
        return torch.cat((mu, sg), dim=1).unsqueeze(2)


# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/tdnn_attention.py#L138
class TDNNModule(nn.Module):
    """
    Time Delayed Neural Module (TDNN) - 1D.

    input:
        inp_filters: input filter channels for conv layer
        out_filters: output filter channels for conv layer
        kernel_size: kernel weight size for conv layer
        dilation: dilation for conv layer
        stride: stride for conv layer
        padding:
            padding for conv layer (default None: chooses padding value such
            that input and output feature shape matches)

    output:
        tdnn layer output
    """

    def __init__(
        self,
        inp_filters: int,
        out_filters: int,
        kernel_size: int = 1,
        dilation: int = 1,
        stride: int = 1,
        padding: int = None,
    ):
        """Initialize the module."""
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size, stride=stride, dilation=dilation)

        self.conv_layer = nn.Conv1d(
            in_channels=inp_filters,
            out_channels=out_filters,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_filters)

    def forward(self, x, length=None):
        """Forward pass of TDNNModule."""
        x = self.conv_layer(x)
        x = self.activation(x)

        return self.bn(x)


# https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/parts/submodules/tdnn_attention.py#L25
class StatsPoolLayer(nn.Module):
    """Statistics and time average pooling (TAP) layer.

    This computes mean and, optionally, standard deviation statistics across the time dimension.

    Args:
        feat_in: Input features with shape [B, D, T]
        pool_mode: Type of pool mode. Supported modes are 'xvector' (mean and standard deviation) and 'tap' (time
            average pooling, i.e., mean)
        eps: Epsilon, minimum value before taking the square root, when using 'xvector' mode.
        biased: Whether to use the biased estimator for the standard deviation when using 'xvector' mode. The default
            for torch.Tensor.std() is True.

    Returns:
        Pooled statistics with shape [B, D].

    Raises:
        ValueError if an unsupported pooling mode is specified.
    """

    def __init__(
        self,
        feat_in: int,
        pool_mode: str = "xvector",
        eps: float = 1e-10,
        biased: bool = True,
    ):
        """Initialize the module."""
        super().__init__()
        supported_modes = {"xvector", "tap"}
        if pool_mode not in supported_modes:
            raise ValueError(
                f"Pool mode must be one of {supported_modes}; got {pool_mode}"
            )
        self.pool_mode = pool_mode
        self.feat_in = feat_in
        self.eps = eps
        self.biased = biased
        if self.pool_mode == "xvector":
            # Mean + std
            self.feat_in *= 2

    def forward(self, encoder_output, length=None):
        """Forward pass of StatsPoolLayer."""
        if length is None:
            mean = encoder_output.mean(dim=-1)  # Time Axis
            if self.pool_mode == "xvector":
                std = encoder_output.std(dim=-1)
                pooled = torch.cat([mean, std], dim=-1)
            else:
                pooled = mean
        else:
            mask = make_seq_mask_like(
                like=encoder_output, lengths=length, valid_ones=False
            )
            encoder_output = encoder_output.masked_fill(mask, 0.0)
            # [B, D, T] -> [B, D]
            means = encoder_output.mean(dim=-1)
            # Re-scale to get padded means
            means = means * (encoder_output.shape[-1] / length).unsqueeze(-1)

            if self.pool_mode == "xvector":
                stds = (
                    encoder_output.sub(means.unsqueeze(-1))
                    .masked_fill(mask, 0.0)
                    .pow(2.0)
                    .sum(-1)  # [B, D, T] -> [B, D]
                    .div(length.view(-1, 1).sub(1 if self.biased else 0))
                    .clamp(min=self.eps)
                    .sqrt()
                )
                pooled = torch.cat((means, stds), dim=-1)
            else:
                pooled = means

        return pooled
