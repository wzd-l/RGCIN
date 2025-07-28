# Ultralytics YOLO 馃殌, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath,trunc_normal_

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    'C2f_mfm',
    'MFM_4',
    'MSAAM_mff'    
)


class C2f_mfm(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    
    
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
 
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else nn.Identity()
 
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        return self.act(self.conv(x))
 
 
class Conv2(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
 
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
 
        return torch.cat((y[0], y[1]), 1)

class Mix(nn.Module):
    def __init__(self, m=-0.50):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class MFM_4(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.5, bias=False):
        super(MFM_4, self).__init__()
        
        self.project_2 = Conv2(dim, dim*2, k=1)
        self.conv_1x1_exp = Conv2(dim, dim*2 ,k=1) ##扩展通道
        self.conv_1x1_2 = Conv2(dim, dim*2 ,k=1) ##扩展通道
        
        self.dwconv3x3_1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv3x3_2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dwconv7x7_1 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.dwconv7x7_2 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)

        self.relu = nn.ReLU(inplace=True)
        
        self.sigmoid = nn.Sigmoid()

        self.avgpool_3 = nn.AvgPool2d(kernel_size=3, stride=1,padding=1)
        self.avgpool_5 = nn.AvgPool2d(kernel_size=3, stride=1,padding=1)

        self.convpool1 = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        self.convpool2 = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        self.mix = Mix()

    def forward(self, x):
        # x_mid = self.project(x)
        x1, x2 = self.project_2(x).chunk(2, dim=1)
        x3_1, x3_2 = self.relu(self.conv_1x1_exp(x1)).chunk(2, dim=1)
        x5_1, x5_2 = self.relu(self.conv_1x1_2(x2)).chunk(2, dim=1)

        x3_1 = self.dwconv3x3_1(x3_1)
        x3_2 = self.dwconv3x3_2(x3_2)
        x3 = x3_2 * self.sigmoid(x3_1 + x)    
        
        x5_1 = self.dwconv7x7_1(x5_1)
        x5_2 = self.dwconv7x7_2(x5_2)
        x5 = x5_2 * self.sigmoid(x5_1 + x)

        x1_pool = self.avgpool_3(x)
        x2_pool = self.avgpool_5(x)
        
        x1_cat = torch.cat([x3, x1_pool], dim=1)
        x2_cat = torch.cat([x5, x2_pool], dim=1)
        x1_out = self.convpool1(x1_cat)
        x2_out = self.convpool2(x2_cat)
        x_out = self.mix(x1_out, x2_out)
        return x_out
   

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.q = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.ky = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.ky_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, y):
        b, c, h, w = x.shape
        x_mid = x + y
        
        q = self.kv_dwconv(self.kv(x_mid))
        q1, q2 = q.chunk(2, dim=1)
        kvx = self.q_dwconv(self.q(x))
        kx, vx = kvx.chunk(2, dim=1)
        kvy = self.ky_dwconv(self.ky(y))
        ky, vy = kvy.chunk(2, dim=1)

        q1 = rearrange(q1, 'b (head c) h w -> b head c (h w)', head=self.num_heads) #[1,2,64,4096]
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads) #[1,2,64,4096]
        kx = rearrange(kx, 'b (head c) h w -> b head c (h w)', head=self.num_heads) #[1,2,64,4096]
        ky = rearrange(ky, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vx = rearrange(vx, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vy = rearrange(vy, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        kx = torch.nn.functional.normalize(kx, dim=-1)
        ky = torch.nn.functional.normalize(ky, dim=-1)
        
        attn1 = (q1 @ ky.transpose(-2, -1)) * self.temperature  #[1,4,32,32]
        attn2 = (q2 @ kx.transpose(-2, -1)) * self.temperature
        attn = attn1 + attn2
        attn = attn.softmax(dim=-1)
        
        out1 = (attn @ vx)
        out2 = (attn @ vy)
        
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out2 = rearrange(out2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.gamma1 * out1 + self.gamma2 * out2
        return out


class Re(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1),
            # nn.ReLU(True),
        )

    def forward(self, x):
        return self.mlp(x)

##########################################################################
class SEM(nn.Module):
    def __init__(self,dim_1,  dim_2, num_heads=4, bias=False, LayerNorm_type='WithBias'):
        super(SEM, self).__init__()
        
        self.norm1 = LayerNorm(dim_2, LayerNorm_type)
        self.attn = Attention(dim_2, num_heads, bias)
        self.norm2 = LayerNorm(dim_2, LayerNorm_type)
        self.re = Re(in_features=dim_2, hidden_features=dim_2 ,out_features=dim_2)


    def forward(self, input_R, input_S):
        
        input_R = self.norm1(input_R)
        x_p = self.norm2(input_S)
        
        x_attn = self.attn(x_p, input_R)
        x_out = self.re(x_attn)
        
        return x_out
    
class MFF(nn.Module):
    def __init__(self, feature_channel=128):
        super(MFF, self).__init__()

        if feature_channel == 128:
            self.conv1 = nn.Sequential(nn.Conv2d(feature_channel, feature_channel // 2, kernel_size=3, stride=1, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv2 = nn.Sequential(nn.ConvTranspose2d(feature_channel * 2, feature_channel // 2, kernel_size=3, stride=2, padding=1,output_padding=1),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.ConvTranspose2d(feature_channel*2, feature_channel // 2, kernel_size=5, stride=4, padding=1,output_padding=1),
                                       nn.ReLU(inplace=False))
            self.feature_channel = feature_channel // 2
            self.conv_out = nn.Conv2d(self.feature_channel, feature_channel, kernel_size=1, stride=1, padding=0)

        elif feature_channel == 256:
            self.conv1 = nn.Sequential(nn.Conv2d(feature_channel // 2, feature_channel // 2, kernel_size=3, stride=2, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv2 = nn.Sequential(nn.Conv2d(feature_channel, feature_channel // 2, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.ConvTranspose2d(feature_channel, feature_channel // 2, kernel_size=3, stride=2, padding=1,output_padding=1),
                                       nn.ReLU(inplace=False))
            self.feature_channel = feature_channel // 2
            self.conv_out = nn.Conv2d(self.feature_channel, feature_channel, kernel_size=1, stride=1, padding=0)

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(feature_channel // 4, feature_channel // 4, kernel_size=5, stride=4, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv2 = nn.Sequential(nn.Conv2d(feature_channel // 2, feature_channel // 4, kernel_size=3, stride=2, padding=1),
                                       nn.ReLU(inplace=False))
            self.conv3 = nn.Sequential(nn.Conv2d(feature_channel // 2, feature_channel // 4, kernel_size=1, stride=1, padding=0),
                                       nn.ReLU(inplace=False))
            self.feature_channel = feature_channel // 4
            self.conv_out = nn.Conv2d(self.feature_channel, feature_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, f1, f2, f3):
        feature1 = self.conv1(f1)
        feature2 = self.conv2(f2)
        feature3 = self.conv3(f3)
        feature = feature1 + feature2 + feature3
        out = self.conv_out(feature)
        return out


class MSAAM_mff(nn.Module):
    def __init__(self, in_channel,xf_channel):
        super(MSAAM_mff, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//2, 1, padding=0, bias=False),
            nn.Conv2d(in_channel//2, in_channel//4, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//4, 3, 1, padding=0, bias=False),
        )
        
        self.mff = MFF(xf_channel)
        self.sem_mlp = SEM(xf_channel, xf_channel)
        
    def forward(self, xs):
        
        x_dn1 = xs[1]
        x_dn2 = xs[2]
        x_dn3 = xs[3]
        x_avg1 = self.avg_pool(x_dn1)
        x_avg2 = self.avg_pool(x_dn2)
        x_avg3 = self.avg_pool(x_dn3)
        fea_avg = torch.cat([x_avg1, x_avg2, x_avg3], dim=1)
        attention_score = self.ca(fea_avg)
        w1, w2, w3 = torch.chunk(attention_score, 3, dim=1)
        x_down1_reweight = x_dn1 * w1
        x_down2_reweight = x_dn2 * w2
        x_down3_reweight = x_dn3 * w3
        x_fusion = self.mff(x_down1_reweight, x_down2_reweight, x_down3_reweight)
        x_out = self.sem_mlp(x_fusion, xs[0])
        return x_out


    

#####Convblock####
class ConvEncoder(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, dim, out_channel ,hidden_dim=64, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, out_channel, kernel_size=kernel_size, stride=2, padding=1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, out_channel, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(out_channel).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = self.conv1(x)
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.norm2(x)
        # x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        x = self.act(x)
        return x
    

def default_conv(in_channels, out_channels, kernel_size, padding=0, bias=True, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, groups=groups)

#####
class ESA(nn.Module):
    def __init__(self, n_feats):  #n_feats :杈撳叆鐗瑰緛鍥剧殑閫氶亾
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = default_conv(n_feats, f, kernel_size=1)
        self.conv_f = default_conv(f, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)

        # self.conv_max = conv(f, f, kernel_size=3, padding=1)
        # self.conv3 = conv(f, f, kernel_size=3, padding=1)
        # self.conv3_ = conv(f, f, kernel_size=3, padding=1)

        self.convGroup = nn.Sequential(
            default_conv(f, f, kernel_size=3, padding=1),
            default_conv(f, f // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            default_conv(f // 4, f, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.conv4 = default_conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        # print(x.shape)
        c1_ = (self.conv1(x))
        # print(c1_.shape)
        c1 = self.conv2(c1_)
        # print(c1.shape)
        v_max = F.max_pool2d(c1, kernel_size=2, stride=2)
        # v_range = self.relu(self.conv_max(v_max))
        # c3 = self.relu(self.conv3(v_range))
        # c3 = self.conv3_(c3)
        c3 = self.convGroup(v_max)

        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m
    
####Feature level attention###

class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2,groups=nf, bias=True)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=True)  # 3x3 convolution

    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out




import numbers
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_x = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_s = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_r = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        
        self.conv_1x1_exp = nn.Conv2d(hidden_features,hidden_features *2 ,kernel_size=1,bias=bias) ##鎵╁睍閫氶亾

        self.conv_1x1 = nn.Conv2d(hidden_features ,hidden_features,kernel_size=1,bias=bias) 
        self.conv_1x1_gate = nn.Conv2d(hidden_features ,hidden_features*2,kernel_size=1,bias=bias) #闂ㄦ帶璋冩暣閫氶亾

        self.conv_1x1_3 = nn.Conv2d(hidden_features *3 ,hidden_features ,kernel_size=1,bias=bias) #璺宠繛閮ㄥ垎闄嶉€氶亾

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 3, dim, kernel_size=1, bias=bias)

    def forward(self, s, r):
        x = s + r
        x = self.project_x(x)
        s = self.project_s(s)
        r = self.project_r(r)
        ####涓婅竟涓€灞傜殑闂ㄦ帶###
        s1_1,s1_2 = self.relu3(self.dwconv3x3(self.conv_1x1_exp(s))).chunk(2, dim=1) #s1_2浣滀负涓嬪垎鏀?        s1_2 = self.conv_1x1(s1_2)
        s1_2 = self.dwconv3x3(torch.cat([s1_2, x], dim=1))
        s1_1 = self.dwconv3x3(self.conv_1x1_gate(s1_1))
        s1_end = s1_1 * F.gelu(s1_2)
        s_end = self.conv_1x1_3(torch.cat([s1_end, s], dim=1))

#############
        ####涓嬭竟涓€灞傜殑闂ㄦ帶###
        r1_1,r1_2 = self.relu5(self.dwconv5x5(self.conv_1x1_exp(r))).chunk(2, dim=1) #r1_2浣滀负涓嬪垎鏀?        r1_2 = self.conv_1x1(r1_2)

        r1_2 = self.dwconv5x5(torch.cat([r1_2, x], dim=1))

        r1_1 = self.dwconv5x5(self.conv_1x1_gate(r1_1))

        r1_end = r1_1 * F.gelu(r1_2)

        r_end = self.conv_1x1_3(torch.cat([r1_end, r], dim=1))

        ###########鍚堝苟涓夊眰杈撳嚭
        x_end = self.project_out(torch.cat([s_end,r_end, x],dim=1))

        return x_end

class DGFN(nn.Module):
    def __init__(self, dim_1, dim, ffn_expansion_factor=1.5, dimension=1,  bias=False, LayerNorm_type='WithBias'):
        super(DGFN, self).__init__()
        self.d = dimension
        self.conv1 = nn.Conv2d(dim_1, dim, (1, 1))
        # self.cca = CCA(dim,dim)

        self.norm = LayerNorm(dim, LayerNorm_type)
        self.norm1 = LayerNorm(dim, LayerNorm_type)

        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_lst): #input_S涓婁竴涓垎鏀殑杈撳嚭鐗瑰緛锛宨nput_R褰撳墠鍒嗘敮鐨勭壒寰?        input_S = input_lst[1]  #[1, 128, 32, 32]
        input_R = input_lst[0]  #[1, 256, 32, 32]
        input_S = input_lst[1]
        input_S = self.conv1(input_S)
        S_after, R_after = self.cca(input_S, input_R)
        S_after = self.norm(S_after)
        R_after = self.norm1(R_after)
        output = self.ffn(S_after, R_after)

        return output

#######################################

# #####DGFN###
# ####weight rerange fusion module####

# class CALayer(nn.Module):
#     def __init__(self, channel):
#         super(CALayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.ca = nn.Sequential(
#             nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // 8, channel // 2, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.ca(y)
#         return y

# # class WRF(nn.Module):
# #     def __init__(self, channel):
# #         super(WRF,self).__init__()
# #         self.ca = CALayer(channel*2)
# #         self.conv = nn.Sequential(
# #             # default_conv(channel, channel, 3, padding=1, groups=channel),
# #             default_conv(channel, channel // 8, 3, padding=1),
# #             nn.BatchNorm2d(channel // 8),
# #             nn.ReLU(inplace=True),
# #             default_conv(channel // 8, channel, 3, padding=1),
# #             nn.Sigmoid()
# #         )
# #         self.norm = nn.Sequential(
# #             nn.BatchNorm2d(channel),
# #             nn.ReLU(inplace=True),
# #         )
        
    
# #     def forward(self, x1, x2):
# #         x_cat = self.ca(torch.cat([x1, x2],dim=1))
# #         # x1 = self.conv(x1)
# #         # x2 = self.conv(x2)
# #         x1_1 = x1 * x_cat
# #         x1_1 = self.norm(x1_1)
# #         x2_1 = x2 * x_cat
# #         x2_1 = self.norm(x2_1)
# #         return x1_1 + x2_1



# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

# class CCA(nn.Module):
#     """
#     CCA Block
#     """
#     def __init__(self, F_g, F_x):
#         super().__init__()
#         self.mlp_x = nn.Sequential(
#             Flatten(),
#             nn.Linear(F_x, F_x))
#         self.mlp_g = nn.Sequential(
#             Flatten(),
#             nn.Linear(F_g, F_x))
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x): 
#         # channel-wise attention
#         avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
#         channel_att_x = self.mlp_x(avg_pool_x)
#         avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
#         channel_att_g = self.mlp_g(avg_pool_g)
#         # channel_att_sum = (channel_att_x + channel_att_g)/2.0
#         channel_att_sum = torch.cat([channel_att_x, channel_att_g],dim=0)
#         # scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
#         scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3)
#         x_after_channel = x * scale[0].expand_as(x)
#         g_after_channel = g * scale[1].expand_as(g)
#         x_after = self.relu(x_after_channel)
#         g_after = self.relu(g_after_channel)
#         out = x_after + g_after
#         return out
    
# class WRF(nn.Module):
#     def __init__(self, channel):
#         super(WRF,self).__init__()
#         self.cca = CCA(channel, channel)
#         # self.ca = CALayer(channel*2)
#         # self.conv = nn.Sequential(
#         #     default_conv(channel, channel, 3, padding=1, groups=channel),
#         #     default_conv(channel, channel // 8, 3, padding=1),
#         #     nn.ReLU(inplace=True),
#         #     default_conv(channel // 8, channel, 3, padding=1),
#         #     nn.Sigmoid()
#         # )
    
#     def forward(self, x1, x2):
#         x_cat = self.cca(x1, x2)
        
#         return x_cat

# # class WRF(nn.Module):
# #     def __init__(self, channel):
# #         super(WRF,self).__init__()
# #         self.ca = CALayer(channel*2)
# #         self.conv = nn.Sequential(
# #             default_conv(channel, channel, 3, padding=1, groups=channel),
# #             default_conv(channel, channel // 8, 3, padding=1),
# #             nn.ReLU(inplace=True),
# #             default_conv(channel // 8, channel, 3, padding=1),
# #             nn.Sigmoid()
# #         )
    
# #     def forward(self, x1, x2):
# #         x_cat = self.ca(torch.cat([x1, x2],dim=1))
# #         x1 = self.conv(x1)
# #         x2 = self.conv(x2)
# #         x1_1 = x1 * x_cat
# #         x2_1 = x2 * x_cat
# #         return x1_1 + x2_1

# import numbers
# from einops import rearrange


# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')

# def to_4d(x,h,w):
#     return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma+1e-5) * self.weight


# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type =='BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias, groups=dim)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=bias)

#     def forward(self, x, y):
#         b, c, h, w = x.shape

#         kv = self.kv_dwconv(self.kv(x))
#         k, v = kv.chunk(2, dim=1)
#         q = self.q_dwconv(self.q(y))

#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)

#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)
#         return out
    
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()

#         hidden_features = int(dim * ffn_expansion_factor)

#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#         self.conv_1x1_2 = nn.Conv2d(hidden_features *2 ,hidden_features*2 ,kernel_size=1,bias=bias) #瀵逛腑闂村眰鐨勬暣浣撶壒寰佽繘琛屼竴涓竴缁村嵎绉粏鍖栫壒寰?#         self.conv_1x1_exp = nn.Conv2d(hidden_features,hidden_features *2 ,kernel_size=1,bias=bias) ##鎵╁睍閫氶亾

#         self.conv_1x1 = nn.Conv2d(hidden_features ,hidden_features,kernel_size=1,bias=bias) 
#         self.conv_1x1_gate = nn.Conv2d(hidden_features ,hidden_features*2,kernel_size=1,bias=bias) #闂ㄦ帶璋冩暣閫氶亾

#         self.conv_1x1_3 = nn.Conv2d(hidden_features *3 ,hidden_features ,kernel_size=1,bias=bias) #璺宠繛閮ㄥ垎闄嶉€氶亾

#         self.dwconv3x3_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
#         self.dwconv5x5_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)


#         self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
#         self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
#         self.relu3 = nn.ReLU()
#         self.relu5 = nn.ReLU()


#         # self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
#         # self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

#         self.relu3_1 = nn.ReLU()
#         self.relu5_1 = nn.ReLU()

#         self.project_out = nn.Conv2d(hidden_features * 4, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x) #[1, 254, 256, 256]
#         x1_1, x3_1 = x.chunk(2, dim=1) #鍒嗘垚涓や釜涓€涓綔涓虹涓€灞傞棬鎺х殑杈撳叆涓€涓綔涓虹涓夊眰闂ㄦ帶杈撳叆
#         x2_1,x2_3 = self.conv_1x1_2(x).chunk(2, dim=1)

#         ####涓婅竟涓€灞傜殑闂ㄦ帶###
#         x1_2,x1_3 = self.relu3(self.dwconv3x3(self.conv_1x1_exp(x1_1))).chunk(2, dim=1) #x1_3浣滀负涓嬪垎鏀?#         x1_3 = self.conv_1x1(x1_3)
#         x1_3 = self.dwconv3x3(torch.cat([x1_3, x2_1],dim=1))
#         x1_2 = self.dwconv3x3(self.conv_1x1_gate(x1_2))
#         x1_end = x1_2 * F.gelu(x1_3)
#         x1_end = self.conv_1x1_3(torch.cat([x1_end, x1_1],dim=1))

# #############
#         ####涓嬭竟涓€灞傜殑闂ㄦ帶###
#         x3_2,x3_3 = self.relu3(self.dwconv5x5(self.conv_1x1_exp(x3_1))).chunk(2, dim=1) #x3_3浣滀负涓嬪垎鏀?#         x3_3 = self.conv_1x1(x3_3)
#         x3_3 = self.dwconv5x5(torch.cat([x3_3, x2_3],dim=1))
#         x3_2 = self.dwconv5x5(self.conv_1x1_gate(x3_2))
#         x3_end = x3_2 * F.gelu(x3_3)
#         x3_end = self.conv_1x1_3(torch.cat([x3_end, x3_1],dim=1))

#         ###########鍚堝苟涓夊眰杈撳嚭
#         x_end = self.project_out(torch.cat([x1_end,x3_end, x],dim=1))

#         return x_end

# class DGFN(nn.Module):
#     def __init__(self, dim_1, dim, ffn_expansion_factor=1.5, dimension=1 , num_heads=2,  bias=False, LayerNorm_type='WithBias'):
#         super(DGFN, self).__init__()
#         self.d = dimension
#         self.conv1 = nn.Conv2d(dim_1, dim, (1, 1))
#         self.wrf = WRF(dim)
#         # self.conv2 = nn.Conv2d(dim_2, dim, (1, 1))
#         self.norm = LayerNorm(dim, LayerNorm_type)
#         # self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention(dim, num_heads, bias)
#         # self.norm2 = LayerNorm(dim, LayerNorm_type)
#         # self.norm3 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

#     def forward(self, input_lst): #input_S涓婁竴涓垎鏀殑杈撳嚭鐗瑰緛锛宨nput_R褰撳墠鍒嗘敮鐨勭壒寰?#         # input_ch = input_R.size()[1]
#         # input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
#         input_S = input_lst[1]  #[1, 128, 32, 32]
#         input_R = input_lst[0]  #[1, 256, 32, 32]
#         input_S = self.conv1(input_S)
#         input_F = self.wrf(input_S, input_R)
#         # input_R = self.conv2(input_R)
#         # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
#         input_F = self.norm(input_F)
#         # input_S = self.norm1(input_S)
#         # input_R = self.norm2(input_R)
#         output_attn = input_F + self.attn(input_F, input_F)
#         output = output_attn + self.ffn(self.norm(output_attn))

#         return output

# #######################################


# #####DGFN###

# import numbers
# from einops import rearrange


# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')

# def to_4d(x,h,w):
#     return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma+1e-5) * self.weight


# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type =='BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias, groups=dim)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=bias)

#     def forward(self, x, y):
#         b, c, h, w = x.shape

#         kv = self.kv_dwconv(self.kv(x))
#         k, v = kv.chunk(2, dim=1)
#         q = self.q_dwconv(self.q(y))

#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)

#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)
#         return out
    
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()

#         hidden_features = int(dim * ffn_expansion_factor)

#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
#         self.conv_1x1_2 = nn.Conv2d(hidden_features *2 ,hidden_features*2 ,kernel_size=1,bias=bias) #瀵逛腑闂村眰鐨勬暣浣撶壒寰佽繘琛屼竴涓竴缁村嵎绉粏鍖栫壒寰?#         self.conv_1x1_exp = nn.Conv2d(hidden_features,hidden_features *2 ,kernel_size=1,bias=bias) ##鎵╁睍閫氶亾

#         self.conv_1x1 = nn.Conv2d(hidden_features ,hidden_features,kernel_size=1,bias=bias) 
#         self.conv_1x1_gate = nn.Conv2d(hidden_features ,hidden_features*2,kernel_size=1,bias=bias) #闂ㄦ帶璋冩暣閫氶亾

#         self.conv_1x1_3 = nn.Conv2d(hidden_features *3 ,hidden_features ,kernel_size=1,bias=bias) #璺宠繛閮ㄥ垎闄嶉€氶亾

#         self.dwconv3x3_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
#         self.dwconv5x5_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)


#         self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
#         self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
#         self.relu3 = nn.ReLU()
#         self.relu5 = nn.ReLU()


#         # self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
#         # self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

#         self.relu3_1 = nn.ReLU()
#         self.relu5_1 = nn.ReLU()

#         self.project_out = nn.Conv2d(hidden_features * 4, dim, kernel_size=1, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x) #[1, 254, 256, 256]
#         x1_1, x3_1 = x.chunk(2, dim=1) #鍒嗘垚涓や釜涓€涓綔涓虹涓€灞傞棬鎺х殑杈撳叆涓€涓綔涓虹涓夊眰闂ㄦ帶杈撳叆
#         x2_1,x2_3 = self.conv_1x1_2(x).chunk(2, dim=1)

#         ####涓婅竟涓€灞傜殑闂ㄦ帶###
#         x1_2,x1_3 = self.relu3(self.dwconv3x3(self.conv_1x1_exp(x1_1))).chunk(2, dim=1) #x1_3浣滀负涓嬪垎鏀?#         x1_3 = self.conv_1x1(x1_3)
#         x1_3 = self.dwconv3x3(torch.cat([x1_3, x2_1],dim=1))
#         x1_2 = self.dwconv3x3(self.conv_1x1_gate(x1_2))
#         x1_end = x1_2 * F.gelu(x1_3)
#         x1_end = self.conv_1x1_3(torch.cat([x1_end, x1_1],dim=1))

# #############
#         ####涓嬭竟涓€灞傜殑闂ㄦ帶###
#         x3_2,x3_3 = self.relu3(self.dwconv5x5(self.conv_1x1_exp(x3_1))).chunk(2, dim=1) #x3_3浣滀负涓嬪垎鏀?#         x3_3 = self.conv_1x1(x3_3)
#         x3_3 = self.dwconv5x5(torch.cat([x3_3, x2_3],dim=1))
#         x3_2 = self.dwconv5x5(self.conv_1x1_gate(x3_2))
#         x3_end = x3_2 * F.gelu(x3_3)
#         x3_end = self.conv_1x1_3(torch.cat([x3_end, x3_1],dim=1))

#         ###########鍚堝苟涓夊眰杈撳嚭
#         x_end = self.project_out(torch.cat([x1_end,x3_end, x],dim=1))

#         return x_end

# class DGFN(nn.Module):
#     def __init__(self, dim_1, dim, ffn_expansion_factor=1.5, dimension=1 , num_heads=2,  bias=False, LayerNorm_type='WithBias'):
#         super(DGFN, self).__init__()
#         self.d = dimension
#         self.conv1 = nn.Conv2d(dim_1, dim, (1, 1))
#         # self.conv2 = nn.Conv2d(dim_2, dim, (1, 1))
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention(dim, num_heads, bias)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         # self.norm3 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

#     def forward(self, input_lst): #input_S涓婁竴涓垎鏀殑杈撳嚭鐗瑰緛锛宨nput_R褰撳墠鍒嗘敮鐨勭壒寰?#         # input_ch = input_R.size()[1]
#         # input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
#         input_S = input_lst[1]  #[1, 128, 32, 32]
#         input_R = input_lst[0]  #[1, 256, 32, 32]
#         input_S = self.conv1(input_S)
#         # input_R = self.conv2(input_R)
#         # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
#         input_S = self.norm1(input_S)
#         input_R = self.norm2(input_R)
#         input_R = input_R + self.attn(input_R, input_S)
#         input_R = input_R + self.ffn(self.norm2(input_R))

#         return input_R

# #######################################



# #####DGFN###

# import numbers
# from einops import rearrange


# def to_3d(x):
#     return rearrange(x, 'b c h w -> b (h w) c')

# def to_4d(x,h,w):
#     return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

# class BiasFree_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(BiasFree_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return x / torch.sqrt(sigma+1e-5) * self.weight


# class WithBias_LayerNorm(nn.Module):
#     def __init__(self, normalized_shape):
#         super(WithBias_LayerNorm, self).__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             normalized_shape = (normalized_shape,)
#         normalized_shape = torch.Size(normalized_shape)

#         assert len(normalized_shape) == 1

#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.normalized_shape = normalized_shape

#     def forward(self, x):
#         mu = x.mean(-1, keepdim=True)
#         sigma = x.var(-1, keepdim=True, unbiased=False)
#         return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

# class LayerNorm(nn.Module):
#     def __init__(self, dim, LayerNorm_type):
#         super(LayerNorm, self).__init__()
#         if LayerNorm_type =='BiasFree':
#             self.body = BiasFree_LayerNorm(dim)
#         else:
#             self.body = WithBias_LayerNorm(dim)

#     def forward(self, x):
#         h, w = x.shape[-2:]
#         return to_4d(self.body(to_3d(x)), h, w)

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias, groups=dim)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=bias)

#     def forward(self, x, y):
#         b, c, h, w = x.shape

#         kv = self.kv_dwconv(self.kv(x))
#         k, v = kv.chunk(2, dim=1)
#         q = self.q_dwconv(self.q(y))

#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)

#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)
#         return out
    
# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, bias):
#         super(FeedForward, self).__init__()

#         hidden_features = int(dim * ffn_expansion_factor)

#         self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1,groups=dim, bias=bias)
#         self.conv_1x1_2 = nn.Conv2d(hidden_features *2 ,hidden_features*2 ,kernel_size=1,groups=hidden_features * 2,bias=bias) #瀵逛腑闂村眰鐨勬暣浣撶壒寰佽繘琛屼竴涓竴缁村嵎绉粏鍖栫壒寰?#         self.conv_1x1_exp = nn.Conv2d(hidden_features,hidden_features *2 ,kernel_size=1,groups=hidden_features,bias=bias) ##鎵╁睍閫氶亾

#         self.conv_1x1 = nn.Conv2d(hidden_features ,hidden_features,kernel_size=1,groups=hidden_features,bias=bias) 
#         self.conv_1x1_gate = nn.Conv2d(hidden_features ,hidden_features*2,kernel_size=1,groups=hidden_features,bias=bias) #闂ㄦ帶璋冩暣閫氶亾

#         self.conv_1x1_3 = nn.Conv2d(hidden_features *3 ,hidden_features ,kernel_size=1,groups=hidden_features,bias=bias) #璺宠繛閮ㄥ垎闄嶉€氶亾

#         self.dwconv3x3_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
#         self.dwconv5x5_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)


#         self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
#         self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
#         self.relu3 = nn.ReLU()
#         self.relu5 = nn.ReLU()

#         #######################
#         self.project_in_1 = nn.Conv2d(dim, hidden_features*2, kernel_size=1,groups=dim, bias=bias)
#         self.dwconv_3 = nn.Conv2d(hidden_features*4, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

#         #######################

#         # self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
#         # self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

#         self.relu3_1 = nn.ReLU()
#         self.relu5_1 = nn.ReLU()

#         self.project_out = nn.Conv2d(hidden_features * 4, dim, kernel_size=1,groups=dim, bias=bias)

#     def forward(self, x):
#         x = self.project_in(x) #[1, 254, 256, 256]
#         x1_1, x3_1 = x.chunk(2, dim=1) #鍒嗘垚涓や釜涓€涓綔涓虹涓€灞傞棬鎺х殑杈撳叆涓€涓綔涓虹涓夊眰闂ㄦ帶杈撳叆
#         x2_1,x2_3 = self.conv_1x1_2(x).chunk(2, dim=1)

#         ####涓婅竟涓€灞傜殑闂ㄦ帶###
#         x1_2,x1_3 = self.relu3(self.dwconv3x3(self.conv_1x1_exp(x1_1))).chunk(2, dim=1) #x1_3浣滀负涓嬪垎鏀?#         x1_3 = self.conv_1x1(x1_3)
#         x1_3 = self.dwconv3x3(torch.cat([x1_3, x2_1],dim=1))
#         x1_2 = self.dwconv3x3(self.conv_1x1_gate(x1_2))
#         x1_end = x1_2 * F.gelu(x1_3)
#         x1_end = self.conv_1x1_3(torch.cat([x1_end, x1_1],dim=1))

# #############
#         ####涓嬭竟涓€灞傜殑闂ㄦ帶###
#         x3_2,x3_3 = self.relu3(self.dwconv5x5(self.conv_1x1_exp(x3_1))).chunk(2, dim=1) #x3_3浣滀负涓嬪垎鏀?#         x3_3 = self.conv_1x1(x3_3)
#         x3_3 = self.dwconv5x5(torch.cat([x3_3, x2_3],dim=1))
#         x3_2 = self.dwconv5x5(self.conv_1x1_gate(x3_2))
#         x3_end = x3_2 * F.gelu(x3_3)
#         x3_end = self.conv_1x1_3(torch.cat([x3_end, x3_1],dim=1))

#         ###########鍚堝苟涓夊眰杈撳嚭
#         x_end = self.project_out(torch.cat([x1_end,x3_end, x],dim=1))

#         return x_end

# class DGFN(nn.Module):
#     def __init__(self, dim_1, dim, ffn_expansion_factor=1.5, dimension=1 , num_heads=2,  bias=False, LayerNorm_type='WithBias'):
#         super(DGFN, self).__init__()
#         self.d = dimension
#         self.conv1 = nn.Conv2d(dim_1, dim, (1, 1))
#         # self.conv2 = nn.Conv2d(dim_2, dim, (1, 1))
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention(dim, num_heads, bias)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         # self.norm3 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

#     def forward(self, input_lst): #input_S涓婁竴涓垎鏀殑杈撳嚭鐗瑰緛锛宨nput_R褰撳墠鍒嗘敮鐨勭壒寰?#         # input_ch = input_R.size()[1]
#         # input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
#         input_S = input_lst[1]  #[1, 128, 32, 32]
#         input_R = input_lst[0]  #[1, 256, 32, 32]
#         input_S = self.conv1(input_S)
#         # input_R = self.conv2(input_R)
#         # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
#         input_S = self.norm1(input_S)
#         input_R = self.norm2(input_R)
#         input_R = input_R + self.attn(input_R, input_S)
#         input_R = input_R + self.ffn(self.norm2(input_R))

#         return input_R

# #######################################


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out
