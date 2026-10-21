import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Remove trailing padding to maintain causal convolution."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    One residual dilated causal convolution block.
    Two conv layers with the same dilation, connected by a residual.

    Residual causal convolution with configurable dilation and dropout.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation,
                 padding, dropout=0.15):
        super().__init__()
        # First dilated conv
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1   = Chomp1d(padding)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated conv (same dilation)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2   = Chomp1d(padding)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        # 1×1 conv for residual when channels change
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Stack of TemporalBlocks with exponentially growing dilation.

    The number of blocks, their channels, and receptive field depend on
    the user-supplied architecture configuration.
    """
    def __init__(self, num_inputs, num_channels, kernel_size=5, dropout=0.15):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation  = 2 ** i          # 1, 2, 4, 8, ...
            in_ch     = num_inputs if i == 0 else num_channels[i-1]
            padding   = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size,
                              stride=1, dilation=dilation,
                              padding=padding, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, L)  →  out: (B, num_channels[-1], L)
        return self.network(x)
