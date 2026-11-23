"""DM0.5 RoboFATE architecture and the two homogeneous-batch objectives."""
import copy
import math
import types

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .data import collate_contexts


def fixed_positions(length, width):
    position = torch.arange(length, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    result = torch.zeros(length, width)
    result[:, 0::2] = torch.sin(position * frequency)
    result[:, 1::2] = torch.cos(position * frequency)
    return result


class MoFFN(nn.Module):
    def __init__(self, ffn, action_width, risk_width, experts=8, top_k=2, beta=0.5):
        super().__init__()
        self.shared = copy.deepcopy(ffn)
        self.experts = nn.ModuleList(copy.deepcopy(ffn) for _ in range(experts))
        self.router_x = nn.Linear(action_width, experts, bias=False)
        self.router_c = nn.Linear(risk_width, experts, bias=False)
        nn.init.normal_(self.router_x.weight, std=0.01)
        nn.init.normal_(self.router_c.weight, std=0.01)
        self.top_k, self.beta = top_k, beta
        self.original, self.risk, self.balance = False, None, None

    def forward(self, x):
        shared = self.shared(x)
        if self.original:
            self.balance = None
            return shared
        if self.risk is None or self.risk.shape[0] != x.shape[0]:
            raise ValueError("Each action sample requires its own intervention risk")
        logits = (self.router_x(x) + self.router_c(self.risk)[:, None, :]).float()
        selected, indices = logits.topk(self.top_k, dim=-1)
        weights = selected.softmax(-1).to(x.dtype).reshape(-1, self.top_k)
        indices = indices.reshape(-1, self.top_k)
        flat = x.reshape(-1, x.shape[-1])
        routed = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            positions, choices = torch.where(indices == index)
            if positions.numel():
                outputs = expert(flat[positions]) * weights[positions, choices, None]
                routed = routed.index_add(0, positions, outputs)
        probability = logits.softmax(-1).mean(dim=(0, 1))
        frequency = F.one_hot(indices, len(self.experts)).float().sum(1).mean(0)
        self.balance = len(self.experts) * (frequency.detach() * probability).sum()
        return self.beta * shared + (1 - self.beta) * routed.reshape_as(x)


class QueryBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(4))
        self.self_attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.cross_attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, queries, context, padding):
        normalized = self.norms[0](queries)
        queries = queries + self.self_attn(normalized, normalized, normalized, need_weights=False)[0]
        context = self.norms[2](context)
        queries = queries + self.cross_attn(self.norms[1](queries), context, context,
                                            key_padding_mask=padding, need_weights=False)[0]
        return queries + self.ffn(self.norms[3](queries))


class PerceiverController(nn.Module):
    def __init__(self, width=2560, queries=64, heads=20, hidden=512, window=8,
                 blocks=2, max_tokens=1536):
        super().__init__()
        self.width, self.window = width, window
        self.queries = nn.Parameter(torch.randn(1, queries, width) * 0.02)
        self.source = nn.Embedding(5, width)
        self.register_buffer("position", fixed_positions(max_tokens, width))
        self.input_norm = nn.LayerNorm(width)
        self.output_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(QueryBlock(width, heads) for _ in range(blocks))
        self.pool = nn.Parameter(torch.randn(width) * 0.02)
        self.summary_norm = nn.LayerNorm(width)
        self.register_buffer("relative_time", fixed_positions(window, width))
        self.gru = nn.GRU(width, hidden, batch_first=True)
        self.trigger = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, 1))
        self.risk = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, width))

    def summarize(self, embeddings, sources, padding=None):
        batch, length, _ = embeddings.shape
        if length > self.position.shape[0]:
            raise ValueError("Controller context exceeds its positional encoding")
        context = self.input_norm(embeddings + self.source(sources) + self.position[:length][None])
        queries = self.queries.expand(batch, -1, -1)
        for block in self.blocks:
            if self.training:
                queries = checkpoint(block, queries, context, padding, use_reentrant=False)
            else:
                queries = block(queries, context, padding)
        queries = self.output_norm(queries)
        scores = torch.einsum("bmd,d->bm", queries, self.pool) / math.sqrt(self.width)
        return torch.einsum("bm,bmd->bd", scores.softmax(-1), queries)

    def predict(self, summaries):
        if summaries.ndim != 3 or summaries.shape[1] != self.window:
            raise ValueError("GRU input must contain exactly eight query summaries")
        sequence = self.summary_norm(summaries) + self.relative_time[None]
        # cuDNN recurrence stays FP32, including inside outer BF16 autocast.
        with torch.autocast(device_type=sequence.device.type, enabled=False):
            _, hidden = self.gru(sequence.float())
        hidden = hidden[-1]
        return self.trigger(hidden).squeeze(-1), self.risk(hidden)

    def forward(self, contexts):
        if len(contexts) != self.window:
            raise ValueError("Expected one context batch per causal policy query")
        return self.predict(torch.stack([self.summarize(*context) for context in contexts], dim=1))


def project_history_with_grad(model, pixels):
    features = model.vision_tower(pixel_values=pixels, return_dict=True).last_hidden_state
    projected = model.multi_modal_projector(features)
    return projected.to(model.get_input_embeddings().weight.dtype)


def install_visual_dispatch(visual):
    from transformers.modeling_outputs import BaseModelOutput
    raw_forward = visual.forward

    def forward(self, pixel_values, **kwargs):
        if pixel_values.ndim == 3:
            if pixel_values.shape[-1] != self.config.hidden_size:
                raise ValueError("Cached SigLIP feature width differs from the encoder")
            return BaseModelOutput(last_hidden_state=pixel_values)
        if pixel_values.ndim != 4:
            raise ValueError("Expected images or frozen SigLIP feature tensors")
        return raw_forward(pixel_values=pixel_values, **kwargs)

    visual.forward = types.MethodType(forward, visual)


def masked_mse(velocity, target, mask):
    denominator = mask.sum(dim=(1, 2))
    if not bool((denominator > 0).all()):
        raise ValueError("An action window has no supervised elements")
    return ((velocity.float() - target.float()).square() * mask).sum(dim=(1, 2)).div(denominator).mean()


def flow_inputs(action, mask):
    noise = torch.randn_like(action)
    time = torch.distributions.Beta(1.5, 1.0).sample((action.shape[0],)).to(action.device)
    time = time * 0.999 + 0.001
    mixed = (1 - time[:, None, None]) * action + time[:, None, None] * noise
    return torch.where(mask, mixed, noise), noise - action, time


class RoboFATE(nn.Module):
    def __init__(self, native_model, model_config):
        super().__init__()
        self.dm = native_model
        self.cfg = dict(model_config)
        self.dm.float()
        embedding = self.dm.model.language_model.embed_tokens.weight
        head = self.dm.model.vlm.lm_head.weight
        if embedding is not head and not torch.equal(embedding, head):
            raise ValueError("Native language embedding and output head must agree before tying")
        self.dm.model.vlm.lm_head.weight = embedding
        self.dm.config.force_fp32_action_path = False
        self.dm.model.config.force_fp32_action_path = False
        self.dm.model.action_expert.force_fp32_action_path = False
        if len(self.dm.model.action_expert.layers) != self.cfg["action_layers"]:
            raise ValueError("DM0.5 action depth differs from Table 6")
        self.moae = copy.deepcopy(self.dm.model.action_expert)
        self.mo_in = copy.deepcopy(self.dm.model.action_in_proj)
        self.mo_out = copy.deepcopy(self.dm.model.action_out_proj)
        self.mo_time_in = copy.deepcopy(self.dm.model.time_mlp_in)
        self.mo_time_out = copy.deepcopy(self.dm.model.time_mlp_out)
        for index, layer in enumerate(self.moae.layers, start=1):
            if index % 2 == 0:
                layer.mlp = MoFFN(layer.mlp, self.cfg["action_width"], self.cfg["risk_width"],
                                  self.cfg["experts"], self.cfg["top_k"], self.cfg["shared_coefficient"])
        self.controller = PerceiverController(
            width=self.cfg["summary_width"], queries=self.cfg["queries"],
            heads=self.cfg["attention_heads"], hidden=self.cfg["gru_hidden"],
            window=self.cfg["window"], blocks=self.cfg["qformer_blocks"],
            max_tokens=self.cfg["max_context_tokens"])
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        for parameter in self.dm.model.visual.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, checkpoint_path, config):
        from opendm.model.dm05 import dm05_arch as arch
        native = arch.DM05ForConditionalGeneration.from_pretrained(
            checkpoint_path, dtype=torch.float32, local_files_only=True)
        native.set_attention_implementation(llm_attn_implementation="sdpa",
                                            vision_attn_implementation="sdpa",
                                            action_attn_implementation="sdpa")
        model = cls(native, config["model"])
        install_visual_dispatch(model.dm.model.visual)
        arch._project_history_image_features = project_history_with_grad
        model.dm.enable_gradient_checkpointing(vlm_gradient_checkpointing=True,
                                               ae_gradient_checkpointing=True)
        model.moae.gradient_checkpointing_enable()
        model.dm.model.visual.to(dtype=torch.bfloat16)
        return model

    def train(self, mode=True):
        super().train(mode)
        self.dm.model.visual.eval()
        return self

    @property
    def pad_token_id(self):
        return self.dm.model.language_model.padding_idx

    def moe_layers(self):
        return [layer.mlp for layer in self.moae.layers if isinstance(layer.mlp, MoFFN)]

    def controller_inputs(self, context):
        from opendm.model.dm05.dm05_utils import HISTORY_PAD_TOKEN_ID
        ids, attention = context["input_ids"], context["attention_mask"].bool()
        if bool((ids == HISTORY_PAD_TOKEN_ID).any()) or context.get("history_mask") is not None:
            raise ValueError("Perceiver inputs contain only the current query")
        image_id = self.dm.config.vlm_config.image_token_index
        visual = self.dm.model.visual(pixel_values=context["pixel_values"], return_dict=True).last_hidden_state
        projected = self.dm.model.vlm.model.multi_modal_projector(visual)
        image_tokens = self.cfg["visual_tokens_per_view"]
        if projected.shape[1] != image_tokens or projected.shape[0] != ids.shape[0] * 3:
            raise ValueError("A current query must supply head, left-wrist, and right-wrist images")
        image_embeddings = projected.reshape(ids.shape[0], 3 * image_tokens, -1)
        rows, source_rows = [], []
        for i in range(ids.shape[0]):
            keep = attention[i] & (ids[i] != image_id)
            text = self.dm.model.embed_language_tokens(ids[i, keep])
            sources = torch.full((len(text),), 3, dtype=torch.long, device=ids.device)
            state = context["state_token_mask"][i, keep]
            if not bool(state.any()):
                raise ValueError("Native robot-state token span is missing")
            sources[state] = 4
            rows.append(torch.cat([image_embeddings[i].to(text.dtype), text]))
            source_rows.append(torch.cat([torch.arange(3, device=ids.device).repeat_interleave(image_tokens), sources]))
        length = max(len(row) for row in rows)
        embeddings = torch.stack([F.pad(row, (0, 0, 0, length - len(row))) for row in rows])
        sources = torch.stack([F.pad(row, (0, length - len(row))) for row in source_rows])
        padding = torch.stack([torch.arange(length, device=ids.device) >= len(row) for row in rows])
        return embeddings, sources, padding

    def summarize(self, context):
        return self.controller.summarize(*self.controller_inputs(context))

    def decision(self, histories):
        if not histories or any(len(history) != self.cfg["window"] for history in histories):
            raise ValueError("Decision histories must contain eight policy queries per sample")
        contexts = [collate_contexts([history[t] for history in histories], self.pad_token_id)
                    for t in range(self.cfg["window"])]
        return self.controller([self.controller_inputs(context) for context in contexts])

    def prefix(self, context):
        from opendm.model.dm05.dm05_utils import VLADynamicCache
        keys = ("input_ids", "attention_mask", "pixel_values", "token_type_ids",
                "history_pixel_values", "history_mask")
        return self.dm._compute_prefix_cache(**{k: context[k] for k in keys if k in context},
                                             cache_cls=VLADynamicCache)

    def velocity(self, context, cache, prefix_len, x, time, moae=True):
        from opendm.model.dm05.dm05_utils import HISTORY_PAD_TOKEN_ID, make_suffix_attn_mask, posemb_sincos
        module = self.moae if moae else self.dm.model.action_expert
        in_proj = self.mo_in if moae else self.dm.model.action_in_proj
        out_proj = self.mo_out if moae else self.dm.model.action_out_proj
        time_in = self.mo_time_in if moae else self.dm.model.time_mlp_in
        time_out = self.mo_time_out if moae else self.dm.model.time_mlp_out
        embeddings = in_proj(x)
        condition = F.silu(time_out(F.silu(time_in(posemb_sincos(
            time, self.cfg["action_width"], max_period=4.0).to(x.dtype)))))
        ids = context["input_ids"]
        mask = make_suffix_attn_mask(input_ids=ids, prefix_len=prefix_len, suffix_len=x.shape[1],
                                    batch_size=x.shape[0], device=x.device, dtype=embeddings.dtype,
                                    pad_token_id=self.pad_token_id,
                                    invisible_prefix_token_ids=(HISTORY_PAD_TOKEN_ID,))
        positions = self.dm._build_suffix_position_ids(
            prefix_len, x.shape[1], x.device, input_ids=ids, pad_token_id=self.pad_token_id,
            invisible_prefix_token_ids=(HISTORY_PAD_TOKEN_ID,))
        keys, values = self.dm._extract_prefix_cache_tensors(cache)
        hidden = module(suffix_embeds=embeddings, attention_mask=mask, position_ids=positions,
                        prefix_cache_keys=keys, prefix_cache_values=values, adarms_cond=condition)
        return out_proj(hidden).float()

    def forward(self, samples, decisions, original=False):
        context = collate_contexts([s["context"] for s in samples], self.pad_token_id)
        action = torch.stack([s["action"] for s in samples])
        mask = torch.stack([s["mask"] for s in samples]).bool()
        if original:
            if decisions:
                raise ValueError("Original-data updates have no decision examples")
            risk, bce = None, action.new_zeros(())
        else:
            _, risk = self.decision([s["history"] for s in samples])
            labels = action.new_tensor([d["label"] for d in decisions])
            if len(decisions) != len(samples) or labels.unique().numel() != 1:
                raise ValueError("Decision microbatch must have one class and four histories")
            logits, _ = self.decision([d["history"] for d in decisions])
            bce = F.binary_cross_entropy_with_logits(logits.float(), labels)
        for layer in self.moe_layers():
            layer.original, layer.risk = original, risk
        cache, length = self.prefix(context)
        x, target, time = flow_inputs(action, mask)
        fm = masked_mse(self.velocity(context, cache, length, x, time, True), target, mask)
        bae = masked_mse(self.velocity(context, cache, length, x, time, False), target, mask)
        balance = action.new_zeros(()) if original else torch.stack([m.balance for m in self.moe_layers()]).mean()
        return {"loss": fm + bae + bce + 0.001 * balance, "moae": fm, "bae": bae,
                "bce": bce, "balance": balance}

    @torch.no_grad()
    def generate(self, context, risk=None, seed=31001, steps=10, decoder="moae", original=False):
        if decoder not in ("bae", "moae"):
            raise ValueError("Unknown action decoder")
        for layer in self.moe_layers():
            layer.original, layer.risk = original, risk
        cache, length = self.prefix(context)
        generator = torch.Generator(device=context["input_ids"].device).manual_seed(seed)
        x = torch.randn((context["input_ids"].shape[0], self.cfg["horizon"], self.cfg["action_dim"]),
                        device=context["input_ids"].device, generator=generator)
        for i in range(steps):
            time = x.new_full((x.shape[0],), 1 - i / steps)
            x = x - self.velocity(context, cache, length, x, time, decoder == "moae") / steps
        return x

    @staticmethod
    def original_update_allowed(name):
        return name.startswith("dm.") or (name.startswith("moae.") and ".mlp.shared." in name)

    def discard_inactive_gradients(self, original):
        if original:
            for name, parameter in self.named_parameters():
                if not self.original_update_allowed(name):
                    parameter.grad = None
