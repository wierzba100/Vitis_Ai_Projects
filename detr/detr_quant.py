import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from pytorch_nndct.apis import torch_quantizer
from tqdm import tqdm
import timm
import math
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import math
import numpy as np
from einops import rearrange, repeat
import torch.nn.functional as F

sys.setrecursionlimit(20000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_spatial_position_embedding(pos_emb_dim, feat_map, temperature=10000):
    assert pos_emb_dim % 4 == 0
    _, _, H, W = feat_map.shape
    y = torch.arange(H, device=feat_map.device, dtype=torch.float32)
    x = torch.arange(W, device=feat_map.device, dtype=torch.float32)
    y = y / (H - 1) * 2 * math.pi
    x = x / (W - 1) * 2 * math.pi
    y_embed = y[:, None].repeat(1, W)
    x_embed = x[None, :].repeat(H, 1)
    dim_t = torch.arange(pos_emb_dim // 4, device=feat_map.device, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pos_emb_dim)
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=3).flatten(2)
    pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=3).flatten(2)
    pos = torch.cat([pos_y, pos_x], dim=2)
    pos = pos.view(-1, pos_emb_dim)
    return pos

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement
    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = self.fn(x, **kwargs)
        if isinstance(res, tuple):
            out, attn = res
            return out + x, attn
        else:
            return res + x

class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        res = self.fn(x, **kwargs)
        if isinstance(res, tuple):
            out, attn = res
            out = self.norm(out)
            return out, attn
        else:
            return self.norm(res)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)
    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')
    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return mask

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, cpb_hidden=512):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.tauu = nn.Parameter(torch.tensor(0.01), requires_grad=True)
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement, upper_lower=False, left_right=True), requires_grad=False)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if self.relative_pos_embedding:
            self.register_buffer('relative_indices', get_relative_distances(window_size))
            self.cpb_mlp = nn.Sequential(
                nn.Linear(2, cpb_hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(cpb_hidden, heads, bias=True),
            )
            self.tau = nn.Parameter(torch.ones(heads) * 1.0)
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))
        self.to_out = nn.Linear(inner_dim, dim)
    def forward(self, x, pad_mask=None):
        if self.shifted:
            x = self.cyclic_shift(x)
        b, n_h, n_w, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size
        nw = nw_h * nw_w
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        dots = torch.einsum('b h w i d, b h w j d -> b h w i j', q, k)
        if self.relative_pos_embedding:
            tau = self.tau.clamp(min=0.01)
            dots = dots / tau.view(1, -1, 1, 1, 1)
        if self.relative_pos_embedding:
            rel = self.relative_indices.to(dots.device).float()
            rel_c = torch.sign(rel) * torch.log1p(rel.abs())
            M2 = self.window_size ** 2
            coords_flat = rel_c.view(M2 * M2, 2)
            cpb_flat = self.cpb_mlp(coords_flat)
            cpb = cpb_flat.view(M2, M2, self.heads)
            cpb = cpb.permute(2, 0, 1)
            dots = dots + cpb.unsqueeze(0).unsqueeze(2)
        else:
            dots = dots + self.pos_embedding
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask.to(dots.device)
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask.to(dots.device)
        if pad_mask is not None:
            if pad_mask.dim() == 2:
                pad_mask_batch = pad_mask.unsqueeze(0).expand(b, -1, -1)
            else:
                pad_mask_batch = pad_mask
            pad_mask_expand = pad_mask_batch.view(b, 1, nw, 1, self.window_size * self.window_size)
            dots = dots + pad_mask_expand.to(dots.device)
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        out = self.to_out(out)
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out, attn

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PostNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PostNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))
        self._shifted_flag = shifted
    def forward(self, x, pad_mask=None):
        attn_list = []
        attn_res = self.attention_block(x, pad_mask=pad_mask)
        if isinstance(attn_res, tuple):
            x, attn = attn_res
            attn_list.append(attn)
        else:
            x = attn_res
        mlp_res = self.mlp_block(x)
        x = mlp_res
        if len(attn_list) > 0:
            return x, (attn_list[0], self._shifted_flag)
        else:
            return x

def closest_factors(n):
    r = int(math.sqrt(n))
    for h in range(r, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n

def windows_attn_to_global(attn, H_pad, W_pad, window_size, shifted):
    b, heads, nw, win2, _ = attn.shape
    win = window_size
    assert win * win == win2
    nw_h = H_pad // win
    nw_w = W_pad // win
    displacement = win // 2 if shifted else 0
    device = attn.device
    global_attn = attn.new_zeros((b, heads, H_pad * W_pad, H_pad * W_pad))
    lr = torch.arange(0, win, device=device)
    lc = torch.arange(0, win, device=device)
    rr, cc = torch.meshgrid(lr, lc, indexing='ij')
    for w_idx in range(nw):
        base_r = (w_idx // nw_w) * win
        base_c = (w_idx % nw_w) * win
        if shifted:
            base_r = (base_r + displacement) % H_pad
            base_c = (base_c + displacement) % W_pad
        grid_r = (base_r + rr) % H_pad
        grid_c = (base_c + cc) % W_pad
        local_rows = grid_r.reshape(-1)
        local_cols = grid_c.reshape(-1)
        local_to_global = local_rows * W_pad + local_cols
        gi = local_to_global
        gj = local_to_global
        global_attn[:, :, gi[:, None], gj[None, :]] += attn[:, :, w_idx, :, :]
    return global_attn

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim=None,
                 dropout_prob=0.0, window_size=7, head_dim=None, relative_pos_embedding=True,
                 expand_to_global=True, use_abs_pos=False):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_model = d_model
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.expand_to_global = expand_to_global
        self.use_abs_pos = use_abs_pos
        if head_dim is None:
            assert d_model % num_heads == 0
            head_dim = d_model // num_heads
        self.head_dim = head_dim
        blocks = []
        for i in range(num_layers):
            shifted = (i % 2 == 1)
            blocks.append(SwinBlock(dim=d_model,
                                    heads=num_heads,
                                    head_dim=self.head_dim,
                                    mlp_dim=d_model * 4,
                                    shifted=shifted,
                                    window_size=window_size,
                                    relative_pos_embedding=relative_pos_embedding))
        self.blocks = nn.ModuleList(blocks)
        self.output_norm = nn.LayerNorm(d_model)
    def forward(self, x, spatial_position_embedding, feat_hw=None):
        B, N, D = x.shape
        assert D == self.d_model
        if feat_hw is not None:
            H, W = feat_hw
        else:
            H, W = closest_factors(N)
        assert H * W == N
        pad_h = (math.ceil(H / self.window_size) * self.window_size) - H
        pad_w = (math.ceil(W / self.window_size) * self.window_size) - W
        H_pad = H + pad_h
        W_pad = W + pad_w
        x_map = x.view(B, H, W, D)
        pos_map = spatial_position_embedding.view(H, W, D)
        valid = torch.ones(H, W, device=x.device, dtype=torch.bool)
        if pad_h > 0 or pad_w > 0:
            x_map = x_map.permute(0, 3, 1, 2)
            x_map = F.pad(x_map, (0, pad_w, 0, pad_h))
            x_map = x_map.permute(0, 2, 3, 1)
            pos_map = pos_map.permute(2, 0, 1)
            pos_map = F.pad(pos_map, (0, pad_w, 0, pad_h))
            pos_map = pos_map.permute(1, 2, 0)
            valid = F.pad(valid.float().unsqueeze(0).unsqueeze(0), (0, pad_w, 0, pad_h))
            valid = valid.squeeze(0).squeeze(0).bool()
        if self.use_abs_pos:
            x_map = x_map + pos_map.view(1, H_pad, W_pad, D)
        valid_padded = valid
        nw_h = H_pad // self.window_size
        nw_w = W_pad // self.window_size
        nw = nw_h * nw_w
        valid_windows = rearrange(valid_padded, '(nw_h w_h) (nw_w w_w) -> (nw_h nw_w) (w_h w_w)',
                                  w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        pad_mask_single = (~valid_windows).float() * -1e9
        pad_mask_batch = pad_mask_single.unsqueeze(0).expand(B, -1, -1)
        attn_weights_raw = []
        attn_weights_global = []
        for blk in self.blocks:
            res = blk(x_map, pad_mask=pad_mask_batch)
            if isinstance(res, tuple):
                x_map, (attn, shifted_flag) = res
                attn_weights_raw.append(attn)
                if self.expand_to_global:
                    global_attn = windows_attn_to_global(attn, H_pad, W_pad, self.window_size, shifted_flag)
                    valid_idx = valid_padded.view(-1).nonzero(as_tuple=False).squeeze(1)
                    cropped = global_attn[:, :, valid_idx][:, :, :, valid_idx]
                    attn_weights_global.append(cropped)
            else:
                x_map = res
        if pad_h > 0 or pad_w > 0:
            x_map = x_map[:, :H, :W, :]
        out = x_map.reshape(B, N, D)
        out = self.output_norm(out)
        if self.expand_to_global and len(attn_weights_global) > 0:
            return out, torch.stack(attn_weights_global)
        elif len(attn_weights_raw) > 0:
            return out, torch.stack(attn_weights_raw)
        else:
            return out, torch.empty(0, device=out.device)

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_model, ff_inner_dim,
                 dropout_prob=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.self_attention_s = nn.ModuleList(
            [nn.MultiheadAttention(d_model, num_heads, dropout=self.dropout_prob, batch_first=True)
             for _ in range(num_layers)]
        )
        self.cross_attention_s = nn.ModuleList(
            [nn.MultiheadAttention(d_model, num_heads, dropout=self.dropout_prob, batch_first=True)
             for _ in range(num_layers)]
        )
        self.ffn_s = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(d_model, ff_inner_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_prob),
                nn.Linear(ff_inner_dim, d_model),
            ) for _ in range(num_layers)]
        )
        self.self_attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.cross_attn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.self_attn_dropouts = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(num_layers)])
        self.cross_attn_dropouts = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(num_layers)])
        self.ffn_dropouts = nn.ModuleList([nn.Dropout(self.dropout_prob) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(d_model)
    def forward(self, query_objects, encoder_output, query_embedding, spatial_position_embedding):
        out = query_objects
        decoder_outputs = []
        decoder_cross_attn_weights = []
        for i in range(self.num_layers):
            in_attn = self.self_attn_norms[i](out)
            q = in_attn + query_embedding
            k = in_attn + query_embedding
            v = in_attn
            out_attn, _ = self.self_attention_s[i](query=q, key=k, value=v)
            out_attn = self.self_attn_dropouts[i](out_attn)
            out = out + out_attn
            in_attn = self.cross_attn_norms[i](out)
            q = in_attn + query_embedding
            k = encoder_output + spatial_position_embedding
            v = encoder_output
            out_attn, decoder_cross_attn = self.cross_attention_s[i](query=q, key=k, value=v)
            decoder_cross_attn_weights.append(decoder_cross_attn)
            out_attn = self.cross_attn_dropouts[i](out_attn)
            out = out + out_attn
            in_ff = self.ffn_norms[i](out)
            out_ff = self.ffn_s[i](in_ff)
            out_ff = self.ffn_dropouts[i](out_ff)
            out = out + out_ff
            decoder_outputs.append(self.output_norm(out))
        output = torch.stack(decoder_outputs)
        return output, torch.stack(decoder_cross_attn_weights)

class DETR(nn.Module):
    def __init__(self, config, num_classes, bg_class_idx):
        super().__init__()
        self.backbone_channels = config['backbone_channels']
        self.d_model = config['d_model']
        self.num_queries = config['num_queries']
        self.num_classes = num_classes
        self.num_decoder_layers = config['decoder_layers']
        self.cls_cost_weight = config['cls_cost_weight']
        self.l1_cost_weight = config['l1_cost_weight']
        self.giou_cost_weight = config['giou_cost_weight']
        self.bg_cls_weight = config['bg_class_weight']
        self.nms_threshold = config['nms_threshold']
        self.bg_class_idx = bg_class_idx
        valid_bg_idx = (self.bg_class_idx == 0 or self.bg_class_idx == (self.num_classes-1))
        assert valid_bg_idx
        self.backbone = timm.create_model('tf_efficientnet_lite0', pretrained=True, features_only=True)
        last_backbone_ch = self.backbone.feature_info.channels()[-1]
        self.backbone_channels = last_backbone_ch
        if config.get('freeze_backbone', False):
            for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone_proj = nn.Conv2d(last_backbone_ch, self.d_model, kernel_size=1)
        self.encoder = TransformerEncoder(num_layers=config['encoder_layers'],
                                          num_heads=config['encoder_attn_heads'],
                                          d_model=config['d_model'],
                                          ff_inner_dim=config['ff_inner_dim'],
                                          dropout_prob=config['dropout_prob'],
                                          window_size=config.get('encoder_window_size', 7),
                                          head_dim=config.get('encoder_head_dim', None),
                                          relative_pos_embedding=config.get('encoder_relative_pos_embedding', True),
                                          expand_to_global=config.get('encoder_expand_to_global', True),
                                          use_abs_pos=config.get('encoder_use_abs_pos', True))
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.d_model))
        self.decoder = TransformerDecoder(num_layers=config['decoder_layers'],
                                          num_heads=config['decoder_attn_heads'],
                                          d_model=config['d_model'],
                                          ff_inner_dim=config['ff_inner_dim'],
                                          dropout_prob=config['dropout_prob'])
        self.class_mlp = nn.Linear(self.d_model, self.num_classes)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 4),
        )
    def forward(self, x, targets=None, score_thresh=0, use_nms=False):
        features = self.backbone(x)
        conv_out = features[-1]
        last_ch = self.backbone.feature_info.channels()[-1]
        if conv_out.dim() == 4 and conv_out.shape[-1] == last_ch and conv_out.shape[1] != last_ch:
            conv_out = conv_out.permute(0, 3, 1, 2)
        conv_out = self.backbone_proj(conv_out)
        batch_size, d_model, feat_h, feat_w = conv_out.shape
        spatial_pos_embed = get_spatial_position_embedding(self.d_model, conv_out)
        conv_out = conv_out.reshape(batch_size, d_model, feat_h * feat_w).transpose(1, 2)
        enc_output, enc_attn_weights = self.encoder(conv_out, spatial_pos_embed, feat_hw=(feat_h, feat_w))
        if enc_attn_weights is not None and enc_attn_weights.numel() != 0:
            if enc_attn_weights.dim() == 5:
                enc_attn_weights = enc_attn_weights.mean(dim=2)
        query_objects = torch.zeros_like(self.query_embed.unsqueeze(0).repeat((batch_size, 1, 1)))
        decoder_outputs = self.decoder(query_objects,
                                       enc_output,
                                       self.query_embed.unsqueeze(0).repeat((batch_size, 1, 1)),
                                       spatial_pos_embed)
        query_objects, decoder_attn_weights = decoder_outputs
        cls_output = self.class_mlp(query_objects)
        bbox_output = self.bbox_mlp(query_objects).sigmoid()
        losses = defaultdict(list)
        detections = []
        detr_output = {}
        if targets is not None:
            num_decoder_layers = self.num_decoder_layers
            for decoder_idx in range(num_decoder_layers):
                cls_idx_output = cls_output[decoder_idx]
                bbox_idx_output = bbox_output[decoder_idx]
                with torch.no_grad():
                    class_prob = cls_idx_output.reshape((-1, self.num_classes)).softmax(dim=-1)
                    pred_boxes = bbox_idx_output.reshape((-1, 4))
                    target_labels = torch.cat([target["labels"] for target in targets])
                    target_boxes = torch.cat([target["boxes"] for target in targets])
                    cost_classification = -class_prob[:, target_labels]
                    pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(pred_boxes,'cxcywh','xyxy')
                    cost_localization_l1 = torch.cdist(pred_boxes_x1y1x2y2,target_boxes,p=1)
                    cost_localization_giou = -torchvision.ops.generalized_box_iou(pred_boxes_x1y1x2y2,target_boxes)
                    total_cost = (self.l1_cost_weight * cost_localization_l1
                                  + self.cls_cost_weight * cost_classification
                                  + self.giou_cost_weight * cost_localization_giou)
                    total_cost = total_cost.reshape(batch_size,self.num_queries,-1).cpu()
                    num_targets_per_image = [len(target["labels"]) for target in targets]
                    total_cost_per_batch_image = total_cost.split(num_targets_per_image,dim=-1)
                    match_indices = []
                    for batch_idx in range(batch_size):
                        batch_idx_assignments = linear_sum_assignment(total_cost_per_batch_image[batch_idx][batch_idx])
                        batch_idx_pred, batch_idx_target = batch_idx_assignments
                        match_indices.append((torch.as_tensor(batch_idx_pred,dtype=torch.int64),
                                              torch.as_tensor(batch_idx_target,dtype=torch.int64)))
                pred_batch_idxs = torch.cat([torch.ones_like(pred_idx) * i for i, (pred_idx, _) in enumerate(match_indices)])
                pred_query_idx = torch.cat([pred_idx for (pred_idx, _) in match_indices])
                valid_obj_target_cls = torch.cat([target["labels"][target_obj_idx] for target, (_, target_obj_idx) in zip(targets, match_indices)])
                target_classes = torch.full(cls_idx_output.shape[:2],fill_value=self.bg_class_idx,dtype=torch.int64,device=cls_idx_output.device)
                target_classes[(pred_batch_idxs, pred_query_idx)] = valid_obj_target_cls
                cls_weights = torch.ones(self.num_classes)
                cls_weights[self.bg_class_idx] = self.bg_cls_weight
                loss_cls = torch.nn.functional.cross_entropy(cls_idx_output.reshape(-1, self.num_classes),
                                                             target_classes.reshape(-1),
                                                             cls_weights.to(cls_idx_output.device))
                matched_pred_boxes = bbox_idx_output[pred_batch_idxs, pred_query_idx]
                target_boxes = torch.cat([target['boxes'][target_obj_idx] for target, (_, target_obj_idx) in zip(targets, match_indices)],dim=0)
                matched_pred_boxes_x1y1x2y2 = torchvision.ops.box_convert(matched_pred_boxes,'cxcywh','xyxy')
                loss_bbox = torch.nn.functional.l1_loss(matched_pred_boxes_x1y1x2y2,target_boxes,reduction='none')
                loss_bbox = loss_bbox.sum() / matched_pred_boxes.shape[0]
                loss_giou = torchvision.ops.generalized_box_iou_loss(matched_pred_boxes_x1y1x2y2,target_boxes)
                loss_giou = loss_giou.sum() / matched_pred_boxes.shape[0]
                losses['classification'].append(loss_cls * self.cls_cost_weight)
                losses['bbox_regression'].append(loss_bbox * self.l1_cost_weight+ loss_giou * self.giou_cost_weight)
            detr_output['loss'] = losses
        else:
            cls_output = cls_output[-1]
            bbox_output = bbox_output[-1]
            prob = torch.nn.functional.softmax(cls_output, -1)
            if self.bg_class_idx == 0:
                scores, labels = prob[..., 1:].max(-1)
                labels = labels+1
            else:
                scores, labels = prob[..., :-1].max(-1)
            boxes = torchvision.ops.box_convert(bbox_output,'cxcywh','xyxy')
            for batch_idx in range(boxes.shape[0]):
                scores_idx = scores[batch_idx]
                labels_idx = labels[batch_idx]
                boxes_idx = boxes[batch_idx]
                keep_idxs = scores_idx >= score_thresh
                scores_idx = scores_idx[keep_idxs]
                boxes_idx = boxes_idx[keep_idxs]
                labels_idx = labels_idx[keep_idxs]
                if use_nms:
                    keep_idxs = torchvision.ops.batched_nms(boxes_idx,scores_idx,labels_idx,iou_threshold=self.nms_threshold)
                    scores_idx = scores_idx[keep_idxs]
                    boxes_idx = boxes_idx[keep_idxs]
                    labels_idx = labels_idx[keep_idxs]
                detections.append({"boxes": boxes_idx,"scores": scores_idx,"labels": labels_idx})
            detr_output['detections'] = detections
            detr_output['enc_attn'] = enc_attn_weights
            detr_output['dec_attn'] = decoder_attn_weights
        return detr_output

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default="./data")
parser.add_argument('--model_dir', default="./")
parser.add_argument('--config_file', default=None)
parser.add_argument('--subset_len', default=None, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--quant_mode', default='calib', choices=['float', 'calib', 'test'])
parser.add_argument('--fast_finetune', dest='fast_finetune', action='store_true')
parser.add_argument('--deploy', dest='deploy', action='store_true')
parser.add_argument('--inspect', dest='inspect', action='store_true')
parser.add_argument('--target', dest='target', nargs="?", const="")
args, _ = parser.parse_known_args()

def load_data(train=False, data_dir="./data", batch_size=4, subset_len=None, **kwargs):
    transform = transforms.Compose([transforms.Resize((640, 640)),transforms.ToTensor()])
    dataset = torchvision.datasets.CocoDetection(root=data_dir,
        annFile=os.path.join(data_dir, "_annotations.coco.json"),transform=transform)
    if subset_len:
        dataset = Subset(dataset, random.sample(range(len(dataset)), subset_len))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), **kwargs)
    return loader

def evaluate(model, val_loader):
    model.eval()
    model = model.to(device)
    total = 0
    correct = 0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, total=len(val_loader)):
            images = torch.stack(images).to(device)
            outputs = model(images)
            detections = outputs['detections']
            for det, target in zip(detections, targets):
                if len(det["labels"]) == 0: 
                    continue
                pred_label = det["labels"][0].item()
                true_labels = [t["category_id"] for t in target]
                if pred_label in true_labels:
                    correct += 1
                total += 1
    acc = 100.0 * correct / max(total, 1)
    return acc

def quantization():
    quant_mode = args.quant_mode
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    config_file = args.config_file
    target = args.target
    config = {
        "backbone_channels": 1280,
        "d_model": 256,
        "num_queries": 100,
        "decoder_layers": 6,
        "encoder_layers": 6,
        "encoder_attn_heads": 8,
        "decoder_attn_heads": 8,
        "ff_inner_dim": 2048,
        "dropout_prob": 0.1,
        "cls_cost_weight": 1.0,
        "l1_cost_weight": 5.0,
        "giou_cost_weight": 2.0,
        "bg_class_weight": 0.1,
        "nms_threshold": 0.5
    }
    num_classes = 92
    bg_class_idx = 0
    state_dict = torch.load("./best_finetune_my_dataset_weight.pth", map_location="cpu")
    model = DETR(config, num_classes=num_classes, bg_class_idx=bg_class_idx).cpu()
    model.load_state_dict(state_dict, strict=False)
    dummy_input = torch.randn([batch_size, 3, 640, 640])
    if quant_mode == 'float':
        quant_model = model
    else:
        quantizer = torch_quantizer(quant_mode, model, (dummy_input,), device=device,
                                    quant_config_file=config_file, target=target)
        quant_model = quantizer.quant_model
    val_loader = load_data(train=False, data_dir=args.data_dir, batch_size=batch_size, subset_len=subset_len)
    if args.fast_finetune and quant_mode == 'calib':
        ft_loader = load_data(train=False, data_dir=args.data_dir, batch_size=batch_size, subset_len=512)
        quantizer.fast_finetune(evaluate, (quant_model, ft_loader))
    elif args.fast_finetune and quant_mode == 'test':
        quantizer.load_ft_param()
    acc = evaluate(quant_model, val_loader)
    print(f"Top-1 detection accuracy proxy: {acc:.2f}%")
    if quant_mode == 'test' and deploy:
        quantizer.export_xmodel(deploy_check=True)

if __name__ == '__main__':
    quantization()
