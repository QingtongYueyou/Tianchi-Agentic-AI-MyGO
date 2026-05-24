"""Export sb3-contrib MaskablePPO model to numpy .npz for deployment without SB3."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

_logger = logging.getLogger("agent.sb3_export")

# Keys in the exported .npz follow this naming convention:
#   shared_{idx}_weight / shared_{idx}_bias   -- policy MLP layers
#   action_net_weight / action_net_bias        -- action head
#   value_net_weight / value_net_bias          -- value head
#   _state_dim / _action_dim                   -- metadata scalars


def export_maskable_ppo_to_numpy(model_path: str, output_path: str) -> bool:
    """Export an sb3-contrib MaskablePPO .zip to a numpy .npz file.

    Returns True on success, False on failure.
    """
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        _logger.error("sb3-contrib is required for export. Install via: pip install sb3-contrib")
        return False

    model_path = str(model_path)
    output_path = str(output_path)

    try:
        model = MaskablePPO.load(model_path)
    except Exception as exc:
        _logger.error("Failed to load MaskablePPO from %s: %s", model_path, exc)
        return False

    state_dict = model.policy.state_dict()
    weights: dict[str, np.ndarray] = {}

    # Extract shared policy MLP layers from mlp_extractor.policy_net
    # sb3 MlpPolicy keys: mlp_extractor.policy_net.{idx}.weight/bias
    # We also track activation type for the numpy wrapper.
    policy_net_keys = sorted(
        [k for k in state_dict if k.startswith("mlp_extractor.policy_net.")],
        key=lambda k: (int(k.split(".")[2]), k),
    )
    layer_idx = 0
    for key in policy_net_keys:
        if ".weight" in key:
            w = state_dict[key].cpu().numpy()
            b_key = key.replace(".weight", ".bias")
            b = state_dict[b_key].cpu().numpy() if b_key in state_dict else np.zeros(w.shape[0])
            weights[f"shared_{layer_idx}_weight"] = w
            weights[f"shared_{layer_idx}_bias"] = b
            layer_idx += 1

    # Activation type: sb3 default MlpPolicy uses Tanh
    weights["_activation"] = np.array(["tanh"])

    # Action head
    if "action_net.weight" in state_dict:
        weights["action_net_weight"] = state_dict["action_net.weight"].cpu().numpy()
        weights["action_net_bias"] = state_dict["action_net.bias"].cpu().numpy()

    # Value head
    if "value_net.weight" in state_dict:
        weights["value_net_weight"] = state_dict["value_net.weight"].cpu().numpy()
        weights["value_net_bias"] = state_dict["value_net.bias"].cpu().numpy()

    # Metadata
    obs_shape = model.observation_space.shape
    weights["_state_dim"] = np.array([int(np.prod(obs_shape))])
    weights["_action_dim"] = np.array([int(model.action_space.n)])
    weights["_num_shared_layers"] = np.array([layer_idx])

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **weights)
        _logger.info(
            "Exported MaskablePPO to %s (%d layers, state_dim=%d, action_dim=%d)",
            output_path,
            layer_idx,
            weights["_state_dim"][0],
            weights["_action_dim"][0],
        )
        return True
    except Exception as exc:
        _logger.error("Failed to save .npz to %s: %s", output_path, exc)
        return False


def export_maskable_ppo_cli() -> None:
    """CLI entry point for MaskablePPO export."""
    parser = argparse.ArgumentParser(description="Export MaskablePPO .zip to .npz")
    parser.add_argument("--model", required=True, help="Path to MaskablePPO .zip model")
    parser.add_argument("--output", required=True, help="Output .npz path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ok = export_maskable_ppo_to_numpy(args.model, args.output)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    export_maskable_ppo_cli()
