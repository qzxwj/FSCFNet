import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
import math


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class FSConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, d=1):
        super().__init__()
        self.c1 = c1

        self.conv1 = Conv(c1, 2 * c1, 1, g=c1)
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv2 = Conv(c1, c2, k, 2, g=c1)
        self.conv3 = Conv(c1 * 3, c2, 3, d=1, g=math.gcd(c1 * 3, c2))
        self.se = SEBlock(c2)
        self.conv4 = Conv(c1, c2, 3, g=c1)
        self.conv5 = Conv(2 * c2, c2, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1, x2 = torch.split(x0, self.c1, dim=1)
        conv_spatial = self.conv2(x1)

        yL, yH = self.wt(x2)

        # Extract the high-frequency subbands
        y_HL = yH[0][:, :, 0, :]
        y_LH = yH[0][:, :, 1, :]
        y_HH = yH[0][:, :, 2, :]

        high_frequency_fused = torch.cat([y_HL, y_LH, y_HH], dim=1)
        high_frequency_fused_output = self.conv3(high_frequency_fused)

        # Apply SE attention
        high_frequency_fused_output = self.se(high_frequency_fused_output)

        low_frequency_fused_output = self.conv4(yL)

        spatial_output = conv_spatial * high_frequency_fused_output

        fused = torch.cat([spatial_output, low_frequency_fused_output], dim=1)
        out = self.conv5(fused)

        return out
