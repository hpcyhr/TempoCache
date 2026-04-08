"""
tempocache/runtime/router.py

Routing logic for TempoCache.

Decision tree (evaluated in order):
  1. d_inter >= TAU_REUSE_HARD       → invalidate cache
  2. cache not valid                  → Full
  3. cache_age >= MAX_REUSE           → Full
  4. d_inter <= TAU_REUSE  AND
     var_intra <= TAU_STABLE          → Reuse
  5. var_intra <= TAU_COLLAPSE        → Collapse
  6. otherwise                        → Full

Semantics preserved from the validated MVP.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from torch import Tensor

from tempocache.config.default_config import RouterConfig
from tempocache.runtime.cache_state import CacheState
from tempocache.runtime.signature_extractor import compute_inter_window_distance


def route(
    sig_cur: Tensor,
    sig_prev: Optional[Tensor],
    cache_state: CacheState,
    var_intra: float,
    config: RouterConfig,
) -> Tuple[str, CacheState, Dict]:
    """Select execution mode and update cache validity flags.

    Returns (mode, cache_state, diagnostics).
    """
    cache_valid_before = cache_state.valid
    cache_age_before = cache_state.age

    # --- inter-window distance ---
    if sig_prev is not None:
        d_inter = compute_inter_window_distance(sig_cur, sig_prev).item()
    else:
        d_inter = float("inf")

    # --- decision tree ---
    if d_inter >= config.tau_reuse_hard:
        cache_state.valid = False

    if not cache_state.valid:
        mode = "full"
    elif cache_state.age >= config.max_reuse:
        mode = "full"
    elif d_inter <= config.tau_reuse and var_intra <= config.tau_stable:
        mode = "reuse"
    elif var_intra <= config.tau_collapse:
        mode = "collapse"
    else:
        mode = "full"

    diag: Dict[str, object] = {
        "mode": mode,
        "d_inter": d_inter,
        "var_intra": var_intra,
        "cache_valid_before": cache_valid_before,
        "cache_valid_after": cache_state.valid,
        "cache_age_before": cache_age_before,
        "signature_norm": sig_cur.abs().sum().item(),
    }
    return mode, cache_state, diag