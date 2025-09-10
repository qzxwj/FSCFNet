import torch
import torch.nn as nn
import math



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def make_divisible(x, divisor):
    return int(math.ceil(x / divisor) * divisor)

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


class ACA(nn.Module):
    def __init__(self, in_channels, out_channels, expansion: float = 0.5):
        super().__init__()

        self.padding_1x3 = nn.ZeroPad2d(padding=(2, 0, 0, 0))
        self.padding_3x1 = nn.ZeroPad2d(padding=(0, 0, 2, 0))
        self.padding_3x3 = nn.ZeroPad2d(padding=(0, 2, 0, 2))

        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.conv1x3_q = Conv(in_channels, hidden_channels, k=(1, 3), g=hidden_channels)  # Query
        self.conv3x1_k = Conv(in_channels, hidden_channels, k=(3, 1), g=hidden_channels)  # Key
        self.conv3x3_v = Conv(in_channels, hidden_channels, k=(3, 3), g=hidden_channels)  # Value

        self.cross_attn_conv = Conv(hidden_channels * 3, out_channels, k=1, g=1)

        self.act = nn.Sigmoid()

    def forward(self, x):
        q = self.conv1x3_q(self.padding_1x3(x))
        k = self.conv3x1_k(self.padding_3x1(x))
        v = self.conv3x3_v(self.padding_3x3(x))
        b, c, h, w = q.shape

        d_k = q.size(1)
        attn_map = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_map = attn_map.flatten(2)

        attn_weights = torch.nn.functional.softmax(attn_map, dim=-1)

        attn_weights = attn_weights.view(b, c, h, h)

        attn_output = torch.matmul(attn_weights, v)

        attn_output_cat = torch.cat([attn_output, q, k], dim=1)

        out = self.cross_attn_conv(attn_output_cat)

        attn_factor = self.act(out)

        return attn_factor




class Conv_ACA(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, use_attention=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        self.use_attention = use_attention
        if self.use_attention:
            self.attention = ACA(c2, c2)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        out = self.conv(x)
        out = self.bn(out)

        # 如果需要，应用注意力机制
        if self.use_attention:
            attn_factor = self.attention(out)
            out = out * attn_factor
        return out





