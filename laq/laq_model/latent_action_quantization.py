from pathlib import Path
import math

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, pack, repeat
from einops.layers.torch import Rearrange

from laq_model.attention import Transformer, ContinuousPositionBias
from laq_model.nsvq import NSVQ

def exists(val):
    return val is not None


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


class ContinuousLatentBottleneck(nn.Module):
    """
    Continuous latent action bottleneck with NSVQ-compatible interface.
    """

    def __init__(self, *, dim, quant_dim, codebook_size, code_seq_len):
        super().__init__()
        self.dim = dim
        self.quant_dim = quant_dim
        self.code_seq_len = code_seq_len
        self.codebook_size = codebook_size

        self.to_latent = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, quant_dim)
        )
        self.to_mu = nn.Linear(quant_dim, quant_dim)
        self.to_logvar = nn.Linear(quant_dim, quant_dim)
        self.to_actions = nn.Linear(quant_dim, code_seq_len * dim)

        # Compatibility-only prototypes to provide stable "indices" outward.
        self.codebook = nn.Parameter(torch.randn(codebook_size, quant_dim) * 0.02)
        self.register_buffer("codebooks_used", torch.zeros(codebook_size, dtype=torch.int32), persistent=False)
        self.last_kl_loss = torch.tensor(0.0)

    def _encode_delta(self, input_data_first, input_data_last):
        pooled_first = input_data_first.mean(dim=1)
        pooled_last = input_data_last.mean(dim=1)
        delta = pooled_last - pooled_first
        hidden = self.to_latent(delta)
        mu = self.to_mu(hidden)
        logvar = self.to_logvar(hidden)
        return mu, logvar

    def _sample(self, mu, logvar, training: bool):
        if training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def _decode_actions(self, latent):
        b = latent.shape[0]
        actions = self.to_actions(latent)
        return actions.reshape(b, self.code_seq_len, self.dim)

    def _nearest_indices(self, latent):
        distances = torch.cdist(latent, self.codebook)
        return torch.argmin(distances, dim=1)

    def _compute_perplexity(self, indices):
        encodings = F.one_hot(indices, num_classes=self.codebook_size).float()
        avg_probs = encodings.mean(dim=0)
        eps = 1e-12
        return torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + eps)))

    def forward(self, input_data_first, input_data_last, codebook_training_only=False):
        mu, logvar = self._encode_delta(input_data_first, input_data_last)
        latent = self._sample(mu, logvar, training=self.training and (not codebook_training_only))
        quantized_input = self._decode_actions(latent)
        self.last_kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        nearest = self._nearest_indices(mu.detach())
        with torch.no_grad():
            self.codebooks_used.index_add_(
                0,
                nearest,
                torch.ones_like(nearest, dtype=self.codebooks_used.dtype)
            )

        perplexity = self._compute_perplexity(nearest)
        indices = nearest.unsqueeze(1).repeat(1, self.code_seq_len)
        return quantized_input, perplexity, self.codebooks_used.cpu().numpy(), indices

    def inference(self, input_data_first, input_data_last, user_action_token_num=None):
        mu, _ = self._encode_delta(input_data_first, input_data_last)
        self.last_kl_loss = torch.zeros((), device=mu.device, dtype=mu.dtype)

        if user_action_token_num is not None:
            if isinstance(user_action_token_num, list):
                user_idx = torch.tensor(user_action_token_num, device=mu.device, dtype=torch.long)
            else:
                user_idx = torch.tensor([user_action_token_num], device=mu.device, dtype=torch.long)

            if user_idx.numel() == 1:
                user_idx = user_idx.repeat(mu.shape[0])
            elif user_idx.numel() != mu.shape[0]:
                user_idx = user_idx.flatten()[:1].repeat(mu.shape[0])

            user_idx = user_idx.clamp_(0, self.codebook_size - 1)
            latent = self.codebook[user_idx]
            indices = user_idx.unsqueeze(1).repeat(1, self.code_seq_len)
        else:
            latent = mu
            nearest = self._nearest_indices(mu.detach())
            indices = nearest.unsqueeze(1).repeat(1, self.code_seq_len)

        quantized_input = self._decode_actions(latent)
        return quantized_input, indices

    def replace_unused_codebooks(self, num_batches):
        # No discrete codebook replacement in continuous mode.
        return

    def decode_from_indices(self, indices):
        if indices.ndim > 1:
            indices = indices[:, 0]
        indices = indices.clamp_(0, self.codebook_size - 1).long()
        latent = self.codebook[indices]
        return self._decode_actions(latent)


class LatentActionQuantization(nn.Module):
    def __init__(
        self,
        *,
        dim,
        quant_dim,
        codebook_size,
        image_size,
        patch_size,
        spatial_depth,
        temporal_depth,
        dim_head = 64,
        heads = 8,
        channels = 3,
        decoder_out_channels = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        code_seq_len = 1,
        use_continuous_bottleneck = False,
        kl_weight = 1e-4,
        kl_warmup_steps = 0,
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.code_seq_len = code_seq_len
        self.use_continuous_bottleneck = use_continuous_bottleneck
        self.kl_weight = kl_weight
        self.kl_warmup_steps = kl_warmup_steps
        self.in_channels = channels
        self.decoder_out_channels = channels if decoder_out_channels is None else decoder_out_channels
        assert self.decoder_out_channels <= self.in_channels, "decoder_out_channels must be <= input channels"
        self.last_recon_loss = None
        self.last_kl_loss = None
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )


        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )
        
        transformer_with_action_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
            has_cross_attn = True,
            dim_context = dim,
        )

        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)


        if self.use_continuous_bottleneck:
            self.vq = ContinuousLatentBottleneck(
                dim=dim,
                quant_dim=quant_dim,
                codebook_size=codebook_size,
                code_seq_len=code_seq_len
            )
        else:
            self.vq = NSVQ(
                dim=dim,
                num_embeddings=codebook_size,
                embedding_dim=quant_dim,
                device='cuda',
                code_seq_len=code_seq_len,
                patch_size=patch_size,
                image_size=image_size
            )
            
            
        self.dec_spatial_transformer = Transformer(depth = spatial_depth, **transformer_with_action_kwargs)
        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, self.decoder_out_channels * patch_width * patch_height),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', c = self.decoder_out_channels, p1 = patch_height, p2 = patch_width)
        )


    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False, assign=False):
        # Always non-strict (VQ keys may differ between continuous/discrete).
        return super().load_state_dict(state_dict, strict=False, assign=assign)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        pt = {k.replace('module.', '') if 'module.' in k else k: v for k, v in pt.items()}
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices):
        if self.use_continuous_bottleneck:
            return self.vq.decode_from_indices(indices)

        codebook = self.vq.codebooks if hasattr(self.vq, "codebooks") else self.vq.codebook
        codes = codebook[indices]
        if codes.ndim == 3:
            b, t, d = codes.shape
            return codes.reshape(b, t * d)
        return codes

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        
        first_tokens = tokens[:, :1]
        last_tokens = tokens[:, 1:]
        
        return first_tokens, last_tokens

        

    def decode(
        self,
        tokens,
        actions,
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        video_shape = tuple(tokens.shape[:-1])


        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        actions = rearrange(actions, 'b t h w d -> (b t) (h w) d')

        attn_bias = self.spatial_rel_pos_bias(h, w, device = tokens.device)

        tokens = self.dec_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape, context=actions)
        

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        rest_frames_tokens = tokens

        recon_video = self.to_pixels_first_frame(rest_frames_tokens)

        return recon_video
    

    def forward(
        self,
        video,
        step = 0,
        mask = None,
        return_recons_only = False,
        return_only_codebook_ids = False,
        return_embeddings = False,
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]


        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb_first_frame(rest_frames)
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)

        shape = tokens.shape
        *_, h, w, _ = shape

        first_tokens, last_tokens = self.encode(tokens)

        first_tokens, first_packed_fhw_shape = pack([first_tokens], 'b * d')
        last_tokens, last_packed_fhw_shape = pack([last_tokens], 'b * d')
        

        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)
        self.lookup_free_quantization = False
        vq_kwargs = dict(mask = vq_mask) if not self.lookup_free_quantization else dict()

        
        tokens, perplexity, codebook_usage, indices = self.vq(first_tokens, last_tokens, codebook_training_only = False)
        
        num_unique_indices = indices.unique().size(0)
        

        
        if ((step % 10 == 0 and step < 100)  or (step % 100 == 0 and step < 1000) or (step % 500 == 0 and step < 5000)) and step != 0:
            print(f"update codebook {step}")
            self.vq.replace_unused_codebooks(tokens.shape[0])

        if return_only_codebook_ids:
            return indices

        if return_embeddings:
            return tokens
        
        if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            ## error
            print("code_seq_len should be square number or defined as 2")
            return
        
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w)
        concat_tokens = first_frame_tokens.detach() # + tokens
        recon_video = self.decode(concat_tokens, tokens)

        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w')
        video = rest_frames 

        if return_recons_only:
            return returned_recon

        target_video = rest_frames[:, :self.decoder_out_channels]

        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.mse_loss(target_video, recon_video, reduction = 'none')
            recon_loss = recon_loss[repeat(mask, 'b t -> b c t', c = self.decoder_out_channels)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(target_video, recon_video)

        self.last_recon_loss = recon_loss
        if self.use_continuous_bottleneck:
            self.last_kl_loss = self.vq.last_kl_loss
            if self.kl_warmup_steps > 0:
                kl_scale = min(float(step) / float(self.kl_warmup_steps), 1.0)
            else:
                kl_scale = 1.0
            recon_loss = recon_loss + (self.kl_weight * kl_scale * self.last_kl_loss)
        else:
            self.last_kl_loss = torch.zeros((), device=recon_loss.device, dtype=recon_loss.dtype)

        return recon_loss, num_unique_indices
        

    def inference(
        self,
        video,
        step = 0,
        mask = None,
        return_only_codebook_ids=False,
        user_action_token_num=None
    ):
        
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device

        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        rest_frames_tokens = self.to_patch_emb_first_frame(rest_frames)
        tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)


        shape = tokens.shape
        *_, h, w, _ = shape

        first_tokens, last_tokens = self.encode(tokens)

        # quantize
        first_tokens, first_packed_fhw_shape = pack([first_tokens], 'b * d')
        last_tokens, last_packed_fhw_shape = pack([last_tokens], 'b * d')

        if user_action_token_num is not None:
            tokens, indices = self.vq.inference(first_tokens, last_tokens, user_action_token_num=user_action_token_num)
        else:
            tokens, indices = self.vq.inference(first_tokens, last_tokens)

        
    
        if return_only_codebook_ids:
            return indices

        if math.sqrt(self.code_seq_len) % 1 == 0: # "code_seq_len should be square number"
            action_h = int(math.sqrt(self.code_seq_len))
            action_w = int(math.sqrt(self.code_seq_len))
        elif self.code_seq_len == 2:
            action_h = 2
            action_w = 1
        else:
            print("code_seq_len should be square number or defined as 2")
            return
        

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = action_h, w = action_w)
        concat_tokens = first_frame_tokens #.detach() #+ tokens
        recon_video = self.decode(concat_tokens, actions=tokens)
        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w')
        video = rest_frames 
        
        return returned_recon

