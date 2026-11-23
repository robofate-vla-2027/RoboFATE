"""Query-level one-time takeover; environment execution remains at the host API."""
from collections import deque

import torch


class TakeoverPolicy:
    def __init__(self, model, q01, q99, seed=0):
        self.model = model.eval()
        self.q01 = torch.as_tensor(q01, dtype=torch.float32)
        self.q99 = torch.as_tensor(q99, dtype=torch.float32)
        if self.q01.shape != (14,) or self.q99.shape != (14,) or not bool((self.q99 > self.q01).all()):
            raise ValueError("Action normalization requires fourteen ordered quantile bounds")
        self.seed = seed
        self.reset()

    def reset(self):
        self.summaries = deque(maxlen=8)
        self.pending = deque()
        self.fixed_risk = None
        self.last_applied_count = None
        self.takeover = False
        self.ready = False
        self.stopped = False

    @torch.no_grad()
    def query(self, applied_count, current_context, action_context, remaining_actions):
        self.ready = False
        self.pending.clear()
        if self.stopped:
            raise RuntimeError("Reset the episode after a stop or failed query")
        device_type = action_context.get("input_ids", torch.empty(0)).device.type
        try:
            with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                return self._query(applied_count, current_context, action_context, remaining_actions)
        except Exception:
            self.stop()
            raise

    def _query(self, applied_count, current_context, action_context, remaining_actions):
        if type(applied_count) is not int or applied_count < 0 or applied_count % 25:
            raise ValueError("Queries occur after multiples of 25 applied commands")
        if self.last_applied_count is None:
            if applied_count != 0:
                raise ValueError("Prime a restored start with its eight causal query summaries")
        elif applied_count != self.last_applied_count + 25:
            raise ValueError("Policy queries must preserve the native causal cadence")
        if type(remaining_actions) is not int or remaining_actions <= 0:
            raise ValueError("No command is generated after the native action budget expires")
        self.ready = False
        self.pending.clear()
        if not self.takeover:
            self.summaries.append(self.model.summarize(current_context))
            if len(self.summaries) == 8:
                logit, risk = self.model.controller.predict(torch.stack(list(self.summaries), dim=1))
                if not bool(torch.isfinite(logit).all() and torch.isfinite(risk).all()):
                    raise FloatingPointError("Controller output is nonfinite")
                probability = float(logit.float().sigmoid().item())
                if probability >= 0.5:
                    self.fixed_risk = risk.detach().clone()
                    self.takeover = True
        decoder = "moae" if self.takeover else "bae"
        action = self.model.generate(action_context, self.fixed_risk,
                                     seed=self.seed + applied_count // 25, steps=10, decoder=decoder)
        if action.shape != (1, 50, 32) or not bool(torch.isfinite(action).all()):
            raise ValueError("Decoder must return a finite 50-by-32 action chunk")
        lo, hi = self.q01.to(action.device), self.q99.to(action.device)
        commands = (action[0, :, :14].float() + 1) * (hi - lo + 1e-6) / 2 + lo
        self.pending.extend(commands[:min(25, remaining_actions)].unbind(0))
        self.last_applied_count = applied_count
        self.ready = True
        return {"decoder": decoder, "takeover": self.takeover, "query_index": applied_count // 25 + 1,
                "commands_available": len(self.pending)}

    @torch.no_grad()
    def prime(self, histories, applied_count):
        from .data import query_history
        query_history(applied_count)
        if self.last_applied_count is not None or self.takeover or len(histories) != 7:
            raise ValueError("Prime supplies the seven preceding queries to a fresh policy")
        if self.stopped:
            raise RuntimeError("Reset the episode before priming a stopped policy")
        device_type = histories[0].get("input_ids", torch.empty(0)).device.type
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            self.summaries.extend(self.model.summarize(context) for context in histories)
        self.last_applied_count = applied_count - 25

    def next_command(self):
        if not self.ready or not self.pending:
            raise RuntimeError("A completed policy query is required before consuming commands")
        return self.pending.popleft()

    def stop(self):
        self.pending.clear()
        self.ready = False
        self.stopped = True
