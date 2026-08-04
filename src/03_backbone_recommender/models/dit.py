import math
import typing

try:
  import flash_attn
  import flash_attn.layers.rotary
  _FLASH_ATTN_AVAILABLE = True
except ImportError:
  flash_attn = None
  _FLASH_ATTN_AVAILABLE = False
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class ContextEncoder(nn.Module):
  """单层 Transformer encoder，将 context item embeddings 编码为单个向量。
  参考 DreamRec 的 cacu_h，但使用冻结的 CLHE 预训练 embeddings 作为输入。
  输入: [B, n_context, hidden_size]  （n_context 个 context item 的 embedding）
  输出: [B, hidden_size]              （mean-pool attention 输出）
  """
  def __init__(self, hidden_size, n_heads=4, dropout=0.1):
    super().__init__()
    self.attn = nn.MultiheadAttention(
      hidden_size, n_heads, dropout=dropout, batch_first=True)
    self.ffn = nn.Sequential(
      nn.Linear(hidden_size, hidden_size * 4),
      nn.GELU(),
      nn.Linear(hidden_size * 4, hidden_size),
    )
    self.norm1 = nn.LayerNorm(hidden_size)
    self.norm2 = nn.LayerNorm(hidden_size)
    self.drop = nn.Dropout(dropout)

  def forward(self, x):  # x: [B, n_context, hidden]
    attn_out, _ = self.attn(x, x, x)
    x = self.norm1(x + self.drop(attn_out))
    x = self.norm2(x + self.drop(self.ffn(x)))
    return x.mean(dim=1)  # [B, hidden]

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
  return x * (1 + scale) + shift


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)


@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
  return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
  def __init__(self, dim, base=10_000):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    self.register_buffer('inv_freq', inv_freq)
    self.seq_len_cached = None
    self.cos_cached = None
    self.sin_cached = None

  def forward(self, x, seq_dim=1):
    seq_len = x.shape[seq_dim]
    if seq_len != self.seq_len_cached:
      self.seq_len_cached = seq_len
      t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
      freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
      emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
      # dims are: batch, seq_len, qkv, head, dim
      self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
      self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
      # This makes the transformation on v an identity.
      self.cos_cached[:,:,2,:,:].fill_(1.)
      self.sin_cached[:,:,2,:,:].fill_(0.)

    return self.cos_cached, self.sin_cached


def rotate_half(x):
  x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(qkv, cos, sin):
  cos = cos[0,:,0,0,:cos.shape[-1]//2]
  sin = sin[0,:,0,0,:sin.shape[-1]//2]
  if _FLASH_ATTN_AVAILABLE:
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)
  # PyTorch fallback uses the same non-interleaved half-rotation convention.
  rotary_half = cos.shape[-1]
  cos = cos[None, :, None, :]
  sin = sin[None, :, None, :]
  output = qkv.clone()
  for qk_index in (0, 1):
    values = qkv[:, :, qk_index]
    first = values[..., :rotary_half]
    second = values[..., rotary_half:2 * rotary_half]
    output[:, :, qk_index, ..., :rotary_half] = first * cos - second * sin
    output[:, :, qk_index, ..., rotary_half:2 * rotary_half] = first * sin + second * cos
  return output


# function overload
def modulate(x, shift, scale):
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.weight = nn.Parameter(torch.ones([dim]))
    self.dim = dim
  def forward(self, x):
    with torch.cuda.amp.autocast(enabled=False):
      x = F.layer_norm(x.float(), [self.dim])
    return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
  """x_skip + residual_scale * W @ x"""
  dim_out, dim_in = W.shape[0], W.shape[1]
  return torch.addmm(
    x_skip.view(-1, dim_out),
    x.view(-1, dim_in),
    W.T,
    alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps into vector representations.
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half, dtype=torch.float32)
      / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb


class LabelEmbedder(nn.Module):
  """Embeds class labels into vector representations.
  
  Also handles label dropout for classifier-free guidance.
  """
  def __init__(self, num_classes, cond_size):
    super().__init__()
    self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
    self.num_classes = num_classes

    # TODO think of initializing with 0.02 std deviation like in original DiT paper

  def forward(self, labels):
    embeddings = self.embedding_table(labels)
    return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
  def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
    super().__init__()
    self.n_heads = n_heads

    self.norm1 = LayerNorm(dim)
    self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.attn_out = nn.Linear(dim, dim, bias=False)
    self.dropout1 = nn.Dropout(dropout)

    self.norm2 = LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, mlp_ratio * dim, bias=True),
      nn.GELU(approximate='tanh'),
      nn.Linear(mlp_ratio * dim, dim, bias=True))
    self.dropout2 = nn.Dropout(dropout)
    self.dropout = dropout

    self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return bias_dropout_add_scale_fused_inference


  def forward(self, x, rotary_cos_sin, c, seqlens=None, sequence_mask=None):
    batch_size, seq_len = x.shape[0], x.shape[1]

    bias_dropout_scale_fn = self._get_bias_dropout_scale()

    (shift_msa, scale_msa, gate_msa, shift_mlp,
     scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

    # attention operation
    x_skip = x
    x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

    qkv = self.attn_qkv(x)
    qkv = rearrange(qkv,
                    'b s (three h d) -> b s three h d',
                    three=3,
                    h=self.n_heads)
    with torch.cuda.amp.autocast(enabled=False):
      cos, sin = rotary_cos_sin
      qkv = apply_rotary_pos_emb(
        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
    use_flash = _FLASH_ATTN_AVAILABLE and (
      sequence_mask is None or bool(sequence_mask.all()))
    if use_flash:
      packed_qkv = rearrange(qkv, 'b s ... -> (b s) ...')
      if seqlens is None:
        cu_seqlens = torch.arange(
          0, (batch_size + 1) * seq_len, step=seq_len,
          dtype=torch.int32, device=packed_qkv.device)
      else:
        cu_seqlens = seqlens.cumsum(-1)
      x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
        packed_qkv, cu_seqlens, seq_len, 0., causal=False)
      x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)
    else:
      if seqlens is not None:
        raise ValueError("Variable-length packed attention requires flash-attn")
      q, k, v = (rearrange(qkv[:, :, index], 'b s h d -> b h s d')
                 for index in range(3))
      sdpa_mask = (
        sequence_mask[:, None, None, :].to(torch.bool)
        if sequence_mask is not None else None)
      x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=sdpa_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False)
      x = rearrange(x, 'b h s d -> b s (h d)')

    x = bias_dropout_scale_fn(self.attn_out(x),
                              None,
                              gate_msa,
                              x_skip,
                              self.dropout)

    # mlp operation
    x = bias_dropout_scale_fn(
      self.mlp(modulate_fused(
        self.norm2(x), shift_mlp, scale_mlp)),
      None, gate_mlp, x, self.dropout)
    return x



class EmbeddingLayer(nn.Module):
  def __init__(self, dim, vocab_dim,config):
    super().__init__()
    self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
    torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))
    # # if self.config['cir'] == 'none': #no rqvae
    # weight_path = f'/model/tteng/GPC/datasets_v1/BundleConstruction/snapshots/cd4ca80e829c1a520f5ccdb228177b99f6caef1f/{config["dataset"]}/{config["feature_type"]}.pt'
    # weight = torch.load(weight_path)
    # # else:
    # Initialize RVQ token embeddings from the explicit codebook artifact.
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[3]
    configured_root = config.get("data_root", None)
    data_root = Path(configured_root).expanduser() if configured_root else repo_root / "data" / "dataset"
    preferred = data_root / "rvq_codebook_weights.npy"
    legacy = data_root / f'{config["feature_type"]}_weight.npy'
    weight_path = preferred if preferred.is_file() else legacy
    if not weight_path.is_file():
      raise FileNotFoundError(
        "Missing RVQ codebook weights. Expected "
        f"{preferred} (preferred) or {legacy} (legacy name).")
    weight = torch.from_numpy(
      np.load(weight_path, allow_pickle=False).astype(np.float32))
    expected_rows = int(config['rq_n_codebooks']) * int(config['rq_codebook_size'])
    feature_dim = int(config.model.get('context_dim', 64))
    if tuple(weight.shape) != (expected_rows, feature_dim):
      raise ValueError(
        f"RVQ codebook weights must have shape {(expected_rows, feature_dim)}, "
        f"got {tuple(weight.shape)}")
    if dim < feature_dim:
      raise ValueError(f"Model hidden_size={dim} cannot hold {feature_dim}-D CLHE weights")
    # Codebook initialization occupies the CLHE subspace. Remaining dimensions
    # keep their normal random initialization and are trainable.
    self.embedding.data[1:weight.shape[0]+1, :feature_dim] = weight

  def forward(self, x):
    return self.embedding[x]


class DDitFinalLayer(nn.Module):
  def __init__(self, hidden_size, out_channels, cond_dim):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

    self.adaLN_modulation = nn.Linear(cond_dim,
                                      2 * hidden_size,
                                      bias=True)
    self.adaLN_modulation.weight.data.zero_()
    self.adaLN_modulation.bias.data.zero_()


  def forward(self, x, c):
    shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
    x = modulate_fused(self.norm_final(x), shift, scale)
    x = self.linear(x)
    return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
  def __init__(self, config, vocab_size: int):
    super().__init__()
    if type(config) == dict:
      config = omegaconf.OmegaConf.create(config)

    self.config = config
    self.vocab_size = vocab_size

    self.vocab_embed = EmbeddingLayer(config.model.hidden_size,
                                      vocab_size,config)
    self.sigma_map = TimestepEmbedder(config.model.cond_dim)
    # Playlist structure is a first-class condition, not reconstructed from a
    # sparse integer item ID.  Zero biases make zero-valued conditions neutral.
    context_dim = int(config.model.get('context_dim', 64))
    self.context_dim = context_dim
    self.mu_c_map = nn.Linear(context_dim, config.model.cond_dim, bias=False)
    self.sigma_c2_map = nn.Linear(1, config.model.cond_dim, bias=False)
    self.context_map = nn.Linear(context_dim, config.model.hidden_size, bias=False)
    self.rotary_emb = Rotary(
      config.model.hidden_size // config.model.n_heads)

    blocks = []
    for _ in range(config.model.n_blocks):
      blocks.append(DDiTBlock(config.model.hidden_size,
                              config.model.n_heads,
                              config.model.cond_dim,
                              dropout=config.model.dropout))
    self.blocks = nn.ModuleList(blocks)

    self.output_layer = DDitFinalLayer(
      config.model.hidden_size,
      vocab_size,
      config.model.cond_dim)
    self.scale_by_sigma = config.model.scale_by_sigma

    # CFG prefix token: learnable null embedding (used when context is dropped)
    self.cfg_enabled = getattr(config.sampling, 'cfg_enabled', False)
    self.cfg_encoder_enabled = getattr(config.sampling, 'cfg_encoder', False)
    if self.cfg_enabled:
      self.none_embedding = nn.Parameter(
        torch.randn(config.model.hidden_size) * 0.02)
      if self.cfg_encoder_enabled:
        self.context_encoder = ContextEncoder(config.model.hidden_size)

  def _get_bias_dropout_scale(self):
    if self.training:
      return bias_dropout_add_scale_fused_train
    else:
      return  bias_dropout_add_scale_fused_inference

  def forward(
      self, indices, sigma, context_emb=None, mu_c=None, sigma_c2=None,
      sequence_mask=None):
    x = self.vocab_embed(indices)
    c = F.silu(self.sigma_map(sigma))
    if mu_c is not None:
      if mu_c.ndim != 2 or mu_c.shape[-1] != self.context_dim:
        raise ValueError(
          f"mu_c must be [B, {self.context_dim}], got {tuple(mu_c.shape)}")
      c = c + F.silu(self.mu_c_map(mu_c))
    if sigma_c2 is not None:
      if sigma_c2.ndim == 1:
        sigma_c2 = sigma_c2[:, None]
      if sigma_c2.ndim != 2 or sigma_c2.shape[-1] != 1:
        raise ValueError(f"sigma_c2 must be [B] or [B, 1], got {tuple(sigma_c2.shape)}")
      c = c + F.silu(self.sigma_c2_map(sigma_c2))

    # CFG prefix token: prepend context embedding (or none_embedding) as position 0
    if self.cfg_enabled:
      B = x.shape[0]
      none_prefix = self.none_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # [B, 1, hidden]

      if context_emb is not None:
        context_emb = self.context_map(context_emb)
        if self.cfg_encoder_enabled:
          # Encoder 模式: context_emb shape [B, n_context, hidden]
          if context_emb.ndim == 2:
            context_emb = context_emb[:, None, :]
          emb = self.context_encoder(context_emb)                            # [B, hidden]
        else:
          # Mean-pool mode accepts either [B, n_context, hidden] or [B, hidden].
          emb = context_emb.mean(dim=1) if context_emb.ndim == 3 else context_emb
        if self.training:
          p_drop = getattr(self.config.sampling, 'cfg_p_drop', 0.1)
          keep = (torch.rand(B, device=emb.device) >= p_drop).float()[:, None]
          none_emb = self.none_embedding.unsqueeze(0).expand(B, -1)
          emb = emb * keep + none_emb * (1 - keep)
        prefix = emb.unsqueeze(1)                                            # [B, 1, hidden]
      else:
        prefix = none_prefix

      x = torch.cat([prefix, x], dim=1)                                     # [B, L+1, hidden]
      if sequence_mask is not None:
        prefix_mask = torch.ones(
          (sequence_mask.shape[0], 1), dtype=torch.bool,
          device=sequence_mask.device)
        sequence_mask = torch.cat([prefix_mask, sequence_mask.bool()], dim=1)

    rotary_cos_sin = self.rotary_emb(x)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      for i in range(len(self.blocks)):
        x = self.blocks[i](
          x, rotary_cos_sin, c, seqlens=None, sequence_mask=sequence_mask)
      x = self.output_layer(x, c)

    if self.cfg_enabled:
      x = x[:, 1:]  # strip prefix position → [B, L, vocab_size]

    return x
