"""Pure-numpy inference for exported MaskablePPO policy with action masking."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from agent.rl_models import _ACTION_DIM, _STATE_DIM

_logger = logging.getLogger("agent.maskable_policy_numpy")


class MaskablePolicyNumpy:
    """Numpy-only inference for an exported MaskablePPO policy.

    Supports action masking: invalid actions get zero probability after softmax.
    """

    def __init__(self) -> None:
        self._weights: dict[str, np.ndarray] = {}
        self._loaded = False
        self._num_layers = 0
        self._activation = "tanh"

    def load(self, path: str) -> None:
        path = str(path)
        if not path.endswith(".npz"):
            path = path + ".npz"
        try:
            data = np.load(path, allow_pickle=False)
            self._weights = {k: data[k] for k in data.files}

            state_dim = int(self._weights.get("_state_dim", np.array([_STATE_DIM]))[0])
            action_dim = int(self._weights.get("_action_dim", np.array([_ACTION_DIM]))[0])
            if state_dim != _STATE_DIM or action_dim != _ACTION_DIM:
                _logger.warning(
                    "MaskablePolicyNumpy ignored incompatible weights from %s: "
                    "state_dim=%s action_dim=%s expected=(%s,%s)",
                    path, state_dim, action_dim, _STATE_DIM, _ACTION_DIM,
                )
                self._weights = {}
                self._loaded = False
                return

            self._num_layers = int(self._weights.get("_num_shared_layers", np.array([2]))[0])
            act_arr = self._weights.get("_activation")
            if act_arr is not None:
                self._activation = str(act_arr[0])
            self._loaded = True
            _logger.info("MaskablePolicyNumpy loaded from %s (activation=%s)", path, self._activation)
        except Exception as exc:
            _logger.warning("MaskablePolicyNumpy load failed: %s", exc)
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def forward(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward pass returning (action_probs, value).

        Returns uniform probs and 0.0 value if not loaded.
        """
        if not self._loaded:
            return np.ones(_ACTION_DIM) / _ACTION_DIM, 0.0

        x = np.array(state, dtype=np.float32).flatten()

        try:
            # Shared policy MLP
            for i in range(self._num_layers):
                x = self._linear(x, f"shared_{i}_weight", f"shared_{i}_bias")
                x = self._tanh(x) if self._activation == "tanh" else self._relu(x)

            # Action head -> logits
            logits = self._linear(x, "action_net_weight", "action_net_bias")
            logits = logits - logits.max()
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum()

            # Value head
            value = self._linear(x, "value_net_weight", "value_net_bias")
            value = float(value.flatten()[0]) if hasattr(value, "flatten") else float(value)

            return probs, value
        except Exception as exc:
            _logger.debug("MaskablePolicyNumpy.forward failed: %s", exc)
            return np.ones(_ACTION_DIM) / _ACTION_DIM, 0.0

    def get_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> int:
        """Select an action, respecting the action mask if provided.

        Args:
            state: 79-dim observation vector.
            action_mask: Boolean array of shape (action_dim,). True = valid.
            deterministic: If True, return argmax; otherwise sample.
        """
        probs, _ = self.forward(state)

        if action_mask is not None:
            mask = np.array(action_mask, dtype=bool)
            if mask.shape[0] != probs.shape[0]:
                _logger.debug("Action mask shape %s != probs shape %s, ignoring mask", mask.shape, probs.shape)
                mask = np.ones(probs.shape[0], dtype=bool)
            masked_probs = probs * mask
            total = masked_probs.sum()
            if total > 0:
                probs = masked_probs / total
            else:
                # All masked: fall back to uniform over mask
                if mask.any():
                    probs = mask.astype(np.float64) / mask.sum()
                else:
                    probs = np.ones_like(probs) / len(probs)

        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def get_action_probs(
        self,
        state: np.ndarray,
        action_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return masked action probabilities (useful for ranked action selection)."""
        probs, _ = self.forward(state)
        if action_mask is not None:
            mask = np.array(action_mask, dtype=bool)
            if mask.shape[0] == probs.shape[0]:
                masked_probs = probs * mask
                total = masked_probs.sum()
                if total > 0:
                    probs = masked_probs / total
        return probs

    # --- internals ---

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def _linear(self, x: np.ndarray, weight_key: str, bias_key: str) -> np.ndarray:
        W = self._weights.get(weight_key)
        b = self._weights.get(bias_key)
        if W is None:
            return x
        out = x @ W.T
        if b is not None:
            out = out + b
        return out
