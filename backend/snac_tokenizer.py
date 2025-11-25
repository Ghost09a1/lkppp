"""
SNAC decoder utilities for Orpheus-style audio tokens.

Matches Orpheus 3B TTS layouts:
- uses hubertsiuzdak/snac_24khz (24 kHz SNAC codec)
- expects flat audio tokens in the form <custom_token_X>
  and groups them in 7s across 3 code levels (Orpheus layout)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from snac import SNAC

# Regex to pull Orpheus audio tokens out of text, e.g. "<custom_token_1234>"
AUDIO_TOKENS_REGEX = re.compile(r"<custom_token_(\d+)>")


def load_snac_model(
    model_id: str = "hubertsiuzdak/snac_24khz",
    local_path: Optional[str] = None,
    device: str = "cpu",
) -> SNAC:
    """
    Load the SNAC codec.

    - If local_path is set, load from there (e.g. unpacked HF repo).
    - Else load by HF model ID (once cached, can be used offline).
    """
    if local_path:
        model_path = Path(local_path)
        snac_model = SNAC.from_pretrained(model_path)
    else:
        snac_model = SNAC.from_pretrained(model_id)

    snac_model = snac_model.eval().to(device)
    return snac_model


def extract_audio_token_ids(text: str) -> List[int]:
    """Return the numeric IDs from <custom_token_X> sequences in the text."""
    return [int(m.group(1)) for m in AUDIO_TOKENS_REGEX.finditer(text)]


def unpack_snac_from_7(flat_ids: Sequence[int]) -> List[torch.Tensor]:
    """
    Orpheus / Parasail layout:
    - Input: flat list with N * 7 token IDs
    - Output: 3 levels (codes_0, codes_1, codes_2) as [1, T] tensors.
    """
    ids = torch.tensor(flat_ids, dtype=torch.int32).reshape(-1, 7)

    # Level 0: first column
    codes_0 = ids[:, 0].unsqueeze(0)

    # Level 1: columns 1 and 4, interleaved
    codes_1 = torch.stack((ids[:, 1], ids[:, 4]), dim=1).reshape(-1).unsqueeze(0)

    # Level 2: columns 2, 3, 5, 6, interleaved
    codes_2 = (
        torch.stack((ids[:, 2], ids[:, 3], ids[:, 5], ids[:, 6]), dim=1)
        .reshape(-1)
        .unsqueeze(0)
    )

    return [codes_0, codes_1, codes_2]


def decode_audio_from_ids(
    flat_ids: Sequence[int],
    snac_model: SNAC,
    device: str = "cpu",
) -> np.ndarray:
    """
    Decode flat Orpheus audio IDs (7-wide groups) into WAV samples.
    Returns a 1D numpy array (float32), mono, 24 kHz.
    """
    if len(flat_ids) == 0:
        raise ValueError("No audio token ids provided")

    if len(flat_ids) % 7 != 0:
        raise ValueError(
            f"Expected multiple of 7 audio tokens, got {len(flat_ids)} "
            f"(this will break the SNAC grouping)."
        )

    levels = unpack_snac_from_7(flat_ids)
    codes = [level.to(device) for level in levels]

    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)

    # audio_hat: [B, 1, T] -> flatten to [T]
    wav = audio_hat[0].detach().cpu().numpy().reshape(-1).astype("float32")
    return wav


def decode_audio_from_text(
    text_with_tokens: str,
    snac_model: SNAC,
    device: str = "cpu",
) -> np.ndarray:
    """
    Convenience: extract <custom_token_...> from text and decode to WAV samples.
    """
    ids = extract_audio_token_ids(text_with_tokens)
    return decode_audio_from_ids(ids, snac_model=snac_model, device=device)
