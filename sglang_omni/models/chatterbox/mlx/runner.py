# SPDX-License-Identifier: Apache-2.0
"""SGLang MLX runner adapter for Chatterbox-Turbo T3 speech-token decoding."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx


class ChatterboxT3MlxModelRunner:
    """Customize prompt prefill with the T3 conditioning prefix; generic MLX
    cache/decode stays upstream."""

    def _load_model(self) -> None:
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            resolve_model_directory,
        )

        from .config import ChatterboxT3MlxConfig
        from .loader import load_t3_weights
        from .model import ChatterboxT3MlxModel

        model_dir = resolve_model_directory(self.model_path, revision=self.revision)
        self.model = ChatterboxT3MlxModel(ChatterboxT3MlxConfig())
        load_t3_weights(self.model, model_dir)

        self._builtin_speaker_emb = None
        self._builtin_cond_speech_tokens = None
        conds_path = os.path.join(model_dir, "conds.pt")
        if os.path.exists(conds_path):
            from chatterbox.tts_turbo import Conditionals

            conds = Conditionals.load(conds_path, map_location="cpu")
            self._builtin_speaker_emb = mx.array(
                conds.t3.speaker_emb.detach().float().cpu().numpy()
            )
            self._builtin_cond_speech_tokens = mx.array(
                conds.t3.cond_prompt_speech_tokens.detach().cpu().numpy().astype("int32")
            )

    def _request_prompt(self, req: Any) -> tuple[mx.array, mx.array, mx.array]:
        text_tokens = getattr(req, "_chatterbox_text_tokens", None)
        if text_tokens is None:
            raise ValueError("Chatterbox MLX request is missing text tokens")
        text_tokens = mx.array([list(text_tokens)], dtype=mx.int32)

        speaker = getattr(req, "_chatterbox_speaker_emb", None)
        cond_tokens = getattr(req, "_chatterbox_cond_speech_tokens", None)
        if speaker is None:
            speaker = self._builtin_speaker_emb
            cond_tokens = self._builtin_cond_speech_tokens
            if speaker is None:
                raise ValueError(
                    "Chatterbox MLX request has no speaker and no builtin voice"
                )
        else:
            if hasattr(speaker, "detach"):
                speaker = speaker.detach().cpu().float().numpy()
            speaker = mx.array(speaker)
            cond_tokens = mx.array([list(cond_tokens)], dtype=mx.int32)
        return speaker, cond_tokens, text_tokens

    def prefill_start(
        self,
        req_id: str,
        new_token_ids: list[int],
        full_token_ids: list[int],
        prefix_slot_ids: list[int],
        new_slot_ids: list[int],
        req_pool_idx: int,
        req: Any | None = None,
        needs_logits: bool = True,
        logit_edit_row: mx.array | None = None,
        logprob_spec: Any = None,
    ):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxPendingPrefill
        from sglang.srt.hardware_backend.mlx.sampling import MlxSamplingParams

        del new_token_ids, new_slot_ids
        if req is None:
            raise ValueError("Chatterbox MLX prefill requires its scheduler request")
        if prefix_slot_ids:
            raise NotImplementedError("Chatterbox MLX does not support radix prefixes yet")

        if self._enable_sampling:
            self._req_sampling[req_id] = MlxSamplingParams.from_req(
                req, deterministic_seeding=self._deterministic_seeding
            )

        speaker, cond_tokens, text_tokens = self._request_prompt(req)
        embeddings = self.model._build_inputs_embeds(speaker, cond_tokens, text_tokens)
        cache = self._acquire_cache()
        logits = self.model._forward_last_logits(embeddings, cache=cache)
        lazy_token, lazy_logprobs = self._select_tokens_with_logprobs(
            logits[:, -1, :], [req_id], [cache], logit_edit_row, logprob_spec
        )
        del needs_logits
        return MlxPendingPrefill(
            lazy_token=lazy_token,
            cache=cache,
            req_id=req_id,
            full_token_ids=list(full_token_ids),
            req_pool_idx=req_pool_idx,
            synced_offset=0,
            lazy_logprobs=lazy_logprobs,
        )


def make_chatterbox_t3_mlx_runner_class():
    """Build the runner after SGLang's MLX backend has been imported."""
    from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

    class ChatterboxT3MlxRunner(ChatterboxT3MlxModelRunner, MlxModelRunner):
        pass

    return ChatterboxT3MlxRunner


__all__ = ["ChatterboxT3MlxModelRunner", "make_chatterbox_t3_mlx_runner_class"]
