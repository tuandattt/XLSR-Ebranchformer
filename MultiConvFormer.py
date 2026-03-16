import torch 
import logging
from typing import Dict, Optional
import numpy as np
from torch import nn
import math
from functools import partial
from torch import Tensor
import torch.nn.functional as F
#from mamba_ssm.modules.mamba_simple import Mamba, Block

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bidirectional=False,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if not bidirectional:
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    else:
        block = BiBlock(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    block.layer_idx = layer_idx
    return block

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class ELiSH(torch.nn.Module):
    def forward(self, x):
        return torch.where(x > 0, x * torch.sigmoid(x), (torch.exp(x) - 1) * torch.sigmoid(x))

def get_activation(act):
    """Return activation function."""
    # Lazy load to avoid unused import

    activation_funcs = {
        "hardtanh": torch.nn.Hardtanh,
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "selu": torch.nn.SELU,
        "swish": Swish,
        "silu": torch.nn.SiLU,
    }

    return activation_funcs[act]()


class MultiConvolutionalSpatialGatingUnit(torch.nn.Module):
    """Multi Convolutional Spatial Gating Unit (M-CSGU)."""

    def __init__(
        self,
        size: int,
        arch_type: str,
        kernel_sizes: str,
        merge_conv_kernel: int,
        use_non_linear: bool,
        dropout_rate: float,
        use_linear_after_conv: bool,
        activation,
        gate_activation: str,
    ):
        super().__init__()

        n_channels = size // 2  # split input channels
        self.norm = nn.LayerNorm(n_channels)

        kernel_sizes = list(map(int, kernel_sizes.split(",")))
        no_kernels = len(kernel_sizes)

        assert (
            n_channels % no_kernels == 0
        ), f"{n_channels} input channels cannot be divided between {no_kernels} kernels"

        self.arch_type = arch_type
        if arch_type in ["sum", "weighted_sum"]:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels,
                        kernel_size,
                        1,
                        (kernel_size - 1) // 2,
                        groups=n_channels,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        elif arch_type in ["concat", "concat_fusion"]:
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        n_channels,
                        n_channels // no_kernels,
                        kernel_size,
                        1,
                        (kernel_size - 1) // 2,
                        groups=n_channels // no_kernels,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
        else:
            raise NotImplementedError(
                f"Unknown architecture type for MultiConvCGMLP: {arch_type}"
            )
        self.use_non_linear = use_non_linear
        if arch_type == "weighted_sum":
            self.kernel_prob_gen = torch.nn.Sequential(
                torch.nn.Linear(n_channels * no_kernels, no_kernels),
                torch.nn.Softmax(dim=-1),
            )
            self.depthwise_conv_fusion = None
        elif arch_type == "concat_fusion":
            self.kernel_prob_gen = None
            self.depthwise_conv_fusion = torch.nn.Conv1d(
                n_channels,
                n_channels,
                kernel_size=merge_conv_kernel,
                stride=1,
                padding=(merge_conv_kernel - 1) // 2,
                groups=n_channels,
                bias=True,
            )
        else:
            self.kernel_prob_gen = None
            self.depthwise_conv_fusion = None

        if use_linear_after_conv:
            self.linear = torch.nn.Linear(n_channels, n_channels)
        else:
            self.linear = None

        self.model_act = activation
        if gate_activation == "identity":
            self.act = torch.nn.Identity()
        else:
            # self.act = ELiSH()
            self.act = torch.nn.SiLU()
            # self.act = get_activation(gate_activation)

        self.dropout = torch.nn.Dropout(dropout_rate)

    def espnet_initialization_fn(self):
        for conv in self.convs:
            torch.nn.init.normal_(conv.weight, std=1e-6)
            torch.nn.init.ones_(conv.bias)
        if self.depthwise_conv_fusion is not None:
            torch.nn.init.normal_(self.depthwise_conv_fusion.weight, std=1e-6)
            torch.nn.init.ones_(self.depthwise_conv_fusion.bias)
        if self.linear is not None:
            torch.nn.init.normal_(self.linear.weight, std=1e-6)
            torch.nn.init.ones_(self.linear.bias)

    def forward(self, x, gate_add=None):
        """Forward method

        Args:
            x (torch.Tensor): (N, T, D)
            gate_add (torch.Tensor): (N, T, D/2)

        Returns:
            out (torch.Tensor): (N, T, D/2)
        """
        x_r, x_i = x.chunk(2, dim=-1)

        x_i = self.norm(x_i).transpose(1, 2)  # (N, D/2, T)

        # TODO: Parallelize this convolution computation
        xs = []
        for conv in self.convs:
            xi = conv(x_i).transpose(1, 2)  # (N, T, D/2)
            if self.arch_type == "sum" and self.use_non_linear:
                xi = self.model_act(xi)
            xs.append(xi)

        if self.arch_type in ["sum", "weighted_sum"]:
            x = torch.stack(xs, dim=-2)
            if self.arch_type == "weighted_sum":
                prob = self.kernel_prob_gen(torch.cat(xs, dim=-1))
                x = prob.unsqueeze(-1) * x

            x_g = x.sum(dim=-2)
        else:
            x_concat = torch.cat(xs, dim=-1)  # (N, T, D)

            if self.arch_type == "concat_fusion":
                x_tmp = x_concat.transpose(1, 2)
                x_tmp = self.depthwise_conv_fusion(x_tmp)
                x_concat = x_concat + x_tmp.transpose(1, 2)

            x_g = x_concat

        if self.linear is not None:
            x_g = self.linear(x_g)

        if gate_add is not None:
            x_g = x_g + gate_add

        x_g = self.act(x_g)
        out = x_r * x_g  # (N, T, D/2)
        out = self.dropout(out)
        return out

class MultiConvolutionalGatingMLP(torch.nn.Module):
    """Convolutional Gating MLP (cgMLP)."""

    def __init__(
        self,
        size: int,
        linear_units: int,
        arch_type: str,
        kernel_sizes: str,
        merge_conv_kernel: int,
        use_non_linear: bool,
        dropout_rate: float,
        use_linear_after_conv: bool,
        activation,
        gate_activation: str,
    ):
        super().__init__()

        if arch_type not in ["sum", "weighted_sum", "concat", "concat_fusion"]:
            raise NotImplementedError(f"Unknown MultiConvCGMLP type: {type}")

        self.channel_proj1 = torch.nn.Sequential(
            torch.nn.Linear(size, linear_units), torch.nn.GELU()
        )
        self.csgu = MultiConvolutionalSpatialGatingUnit(
            size=linear_units,
            arch_type=arch_type,
            kernel_sizes=kernel_sizes,
            merge_conv_kernel=merge_conv_kernel,
            use_non_linear=use_non_linear,
            dropout_rate=dropout_rate,
            use_linear_after_conv=use_linear_after_conv,
            activation=activation,
            gate_activation=gate_activation,
        )
        self.channel_proj2 = torch.nn.Linear(linear_units // 2, size)

    def forward(self, x, mask=None):
        if isinstance(x, tuple):
            xs_pad, pos_emb = x
        else:
            xs_pad, pos_emb = x, None

        
        xs_pad = self.channel_proj1(xs_pad)  # size -> linear_units
        xs_pad = self.csgu(xs_pad)  # linear_units -> linear_units/2
        xs_pad = self.channel_proj2(xs_pad)  # linear_units/2 -> size

        if pos_emb is not None:
            out = (xs_pad, pos_emb)
        else:
            out = xs_pad
        return out

class SEModule(nn.Module):
    def __init__(self, channels, SE_ratio=8):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return x

class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=torch.nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x):
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class MultiHeadedAttention(nn.Module):

    def __init__(
        self,
        n_head,
        n_feat,
        dropout_rate,
        qk_norm=False,
        use_flash_attn=False,
        causal=False,
        cross_attn=False,
        use_sdpa=False,
    ):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()

        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = (
            nn.Dropout(p=dropout_rate) if not use_flash_attn else nn.Identity()
        )
        self.dropout_rate = dropout_rate

        # LayerNorm for q and k
        self.q_norm = nn.LayerNorm(self.d_k) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.d_k) if qk_norm else nn.Identity()

        self.use_flash_attn = use_flash_attn
        self.causal = causal  # only used with flash_attn
        self.cross_attn = cross_attn  # only used with flash_attn

        self.use_sdpa = use_sdpa

    def forward_qkv(self, query, key, value, expand_kv=False):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)

        if expand_kv:
            k_shape = key.shape
            k = (
                self.linear_k(key[:1, :, :])
                .expand(n_batch, k_shape[1], k_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
            v_shape = value.shape
            v = (
                self.linear_v(value[:1, :, :])
                .expand(n_batch, v_shape[1], v_shape[2])
                .view(n_batch, -1, self.h, self.d_k)
            )
        else:
            k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
            v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)

        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        q = self.q_norm(q)
        k = self.k_norm(k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask, expand_kv=False):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
            expand_kv (bool): Used only for partially autoregressive (PAR) decoding.
                When set to `True`, `Linear` layers are computed only for the first
                batch. This is useful to reduce the memory usage during decoding
                when the batch size is #beam_size x #mask_count, which can be large.
                Typically, in single waveform inference of PAR, `Linear` layers
                should not be computed for all batches for source-attention.

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        # Use PyTorch's Scaled Dot Product Attention implementation
        if getattr(self, "use_sdpa", False):
            q, k, v = self.forward_qkv(query, key, value, expand_kv)

            # The shape of mask must be broadcastable to the shape of attention weights
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                mask.unsqueeze(1) if mask is not None else None,
                dropout_p=self.dropout_rate if self.training else 0.0,
            )  # (batch, head, time1, d_k)

            out = out.transpose(1, 2)  # (batch, time1, head, d_k)
            out = out.reshape(out.shape[0], out.shape[1], -1)  # (batch, time1, d_model)
            return self.linear_out(out)  # (batch, time1, d_model)

        q, k, v = self.forward_qkv(query, key, value, expand_kv)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)

class TCMAttention(nn.Module):
    # Head Token attention: https://arxiv.org/pdf/2210.05958.pdf
    def __init__(self, dim, heads=8, dim_head=64, qkv_bias=False, dropout=0., proj_drop=0.):
        super().__init__()
        self.num_heads = heads
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.act = nn.GELU()
        self.ht_proj = nn.Linear(dim_head, dim,bias=True)
        self.ht_norm = nn.LayerNorm(dim_head)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_heads, dim))
    
    def forward(self, x, mask=None):
        B, N, C = x.shape

        # head token
        head_pos = self.pos_embed.expand(x.shape[0], -1, -1)
        x_ = x.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) 
        x_ = x_.mean(dim=2)  # now the shape is [B, h, 1, d//h]
        x_ = self.ht_proj(x_).reshape(B, -1, self.num_heads, C // self.num_heads)
        x_ = self.act(self.ht_norm(x_)).flatten(2)
        x_ = x_ + head_pos
        x = torch.cat([x, x_], dim=1)
        
        # normal mhsa
        qkv = self.qkv(x).reshape(B, N+self.num_heads, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N+self.num_heads, C)
        x = self.proj(x)
        
        # merge head tokens into cls token
        cls, patch, ht = torch.split(x, [1, N-1, self.num_heads], dim=1)
        cls = cls + torch.mean(ht, dim=1, keepdim=True) + torch.mean(patch, dim=1, keepdim=True)
        x = torch.cat([cls, patch], dim=1)

        x = self.proj_drop(x)

        return x


class MultiConvformerBlock(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        dim_head: int = 64,
        output_size: int = 256,
        attention_heads: int = 4,
        cgmlp_linear_units: int = 2048,
        cgmlp_conv_kernel: int = 31,
        use_linear_after_conv: bool = False,
        gate_activation: str = "identity",
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        layer_drop_rate: float = 0.0,
        ffn_activation_type: str = "silu",
        linear_units: int = 1024,
        positionwise_layer_type: str = "linear",
        merge_conv_kernel: int = 3,
        qk_norm: bool = False,
    ):
        super().__init__()

        self.size = output_size
        self.attn = MultiHeadedAttention(
            attention_heads,
            output_size,
            attention_dropout_rate,
            qk_norm,
            False,
            False,
            False,
        )
        # self.attn = create_block(
        #     output_size,
        #     ssm_cfg=None,
        #     norm_epsilon=1e-5,
        #     rms_norm=False,
        #     residual_in_fp32=False,
        #     fused_add_norm=False,
        # )

        activation = get_activation(ffn_activation_type)
        self.cgmlp = MultiConvolutionalGatingMLP(
            size=output_size,
            linear_units=1024,
            arch_type="concat_fusion",
            kernel_sizes="7,15,23,31",
            merge_conv_kernel=31,
            use_non_linear=True,
            dropout_rate=0.1,
            use_linear_after_conv=True,
            activation="silu",
            gate_activation="silu"
        )

        self.feed_forward = PositionwiseFeedForward(
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        self.feed_forward_macaron = PositionwiseFeedForward(
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )

        self.ff_scale = 1.0
        if self.feed_forward is not None:
            self.norm_ff = nn.LayerNorm(output_size)
        if self.feed_forward_macaron is not None:
            self.ff_scale = 0.5
            self.norm_ff_macaron = nn.LayerNorm(output_size)

        self.norm_mha = nn.LayerNorm(output_size)
        self.norm_mlp = nn.LayerNorm(output_size)
        self.norm_final = nn.LayerNorm(output_size)
        #self.norm_mamba = nn.LayerNorm(output_size)

        self.dropout = torch.nn.Dropout(dropout_rate)

        self.depthwise_conv_fusion = torch.nn.Conv1d(
            output_size*2,
            output_size*2,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=(merge_conv_kernel - 1) // 2,
            groups=output_size*2,
            bias=True,
        )

        self.merge_proj = torch.nn.Linear(output_size + output_size, output_size)
        

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if cache is not None:
            raise NotImplementedError("cache is not None, which is not tested")

        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
        
        residual = x
        x = self.norm_mha(x)
        x_att = self.attn(x, x, x, mask)
        # x_att, residual = self.attn(x1, None, inference_params=None)
        # f_residual = (x_att + residual) if residual is not None else x_att
        # x_att = self.norm_mamba(f_residual)

        x = self.dropout(x_att)
        x = residual + x

        residual = x
        x = self.norm_mlp(x)
        x = self.cgmlp(x, mask)
        x = self.dropout(x)
        x = residual + x

        if self.feed_forward is not None:
            # feed forward module
            residual = x
            x = self.norm_ff(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        x = self.norm_final(x)

        return x, mask