"""Parameter space definitions for VAD optimization.

Defines search spaces for each VAD backend, separated into:
- backend_params: Backend-specific parameters (threshold, mode, hop_size, etc.)
- vad_config: Segment detection parameters (min_speech_ms, min_silence_ms, etc.)
"""

from __future__ import annotations

from typing import Any

import optuna

from livecap_core.vad.config import VADConfig


# Parameter space definitions
# Each VAD has "backend" and "vad_config" sub-dictionaries
#
# IMPORTANT: threshold is in vad_config, NOT backend_params!
# The state machine uses VADConfig.threshold for speech detection
# (see livecap_core/vad/state_machine.py:114)
# Backend threshold parameters are NOT used for actual detection.
PARAM_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "silero": {
        "backend": {
            # Silero backend threshold is NOT used for detection
            # (kept for compatibility but not optimized)
        },
        "vad_config": {
            # threshold IS used by state machine for speech detection
            "threshold": {"type": "float", "low": 0.2, "high": 0.8},
            "neg_threshold": {"type": "float", "low": 0.1, "high": 0.5},
            "min_speech_ms": {"type": "int", "low": 100, "high": 500, "step": 50},
            "min_silence_ms": {"type": "int", "low": 30, "high": 300, "step": 10},
            "speech_pad_ms": {"type": "int", "low": 30, "high": 200, "step": 10},
        },
    },
    "tenvad": {
        "backend": {
            "hop_size": {"type": "categorical", "choices": [160, 256]},
            # TenVAD backend threshold is NOT used for detection
        },
        "vad_config": {
            # threshold IS used by state machine for speech detection
            "threshold": {"type": "float", "low": 0.2, "high": 0.8},
            "neg_threshold": {"type": "float", "low": 0.1, "high": 0.5},
            "min_speech_ms": {"type": "int", "low": 100, "high": 500, "step": 50},
            "min_silence_ms": {"type": "int", "low": 30, "high": 300, "step": 10},
            "speech_pad_ms": {"type": "int", "low": 30, "high": 200, "step": 10},
        },
    },
    "webrtc": {
        "backend": {
            # WebRTC modes: 0=normal, 1=low bitrate, 2=aggressive, 3=very aggressive
            "mode": {"type": "categorical", "choices": [0, 1, 2, 3]},
            "frame_duration_ms": {"type": "categorical", "choices": [10, 20, 30]},
        },
        "vad_config": {
            # NOTE: threshold is excluded because WebRTC outputs binary (0/1)
            # The state machine threshold has no effect on WebRTC detection
            "min_speech_ms": {"type": "int", "low": 100, "high": 500, "step": 50},
            "min_silence_ms": {"type": "int", "low": 30, "high": 300, "step": 10},
            "speech_pad_ms": {"type": "int", "low": 30, "high": 200, "step": 10},
        },
    },
}


def _suggest_param(trial: optuna.Trial, name: str, config: dict[str, Any]) -> Any:
    """Suggest a single parameter value based on its configuration.

    Args:
        trial: Optuna trial object
        name: Parameter name
        config: Parameter configuration dict with 'type' and bounds

    Returns:
        Suggested parameter value
    """
    param_type = config["type"]

    if param_type == "float":
        return trial.suggest_float(name, config["low"], config["high"])
    elif param_type == "int":
        step = config.get("step", 1)
        return trial.suggest_int(name, config["low"], config["high"], step=step)
    elif param_type == "categorical":
        return trial.suggest_categorical(name, config["choices"])
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def _suggest_group(
    trial: optuna.Trial,
    group_config: dict[str, dict[str, Any]],
    prefix: str = "",
) -> dict[str, Any]:
    """Suggest parameters for a group (backend or vad_config).

    Args:
        trial: Optuna trial object
        group_config: Dictionary of parameter configurations
        prefix: Optional prefix for parameter names (for namespacing)

    Returns:
        Dictionary of parameter names to suggested values
    """
    params = {}
    for param_name, param_config in group_config.items():
        full_name = f"{prefix}{param_name}" if prefix else param_name
        params[param_name] = _suggest_param(trial, full_name, param_config)
    return params


def suggest_params(
    trial: optuna.Trial,
    vad_type: str,
) -> tuple[dict[str, Any], VADConfig | None]:
    """Suggest parameters for a VAD type.

    Args:
        trial: Optuna trial object
        vad_type: VAD type ("silero", "tenvad", "webrtc")

    Returns:
        Tuple of (backend_params, vad_config)
        - backend_params: Dict of backend-specific parameters
        - vad_config: VADConfig instance or None if no vad_config params

    Raises:
        ValueError: If vad_type is not in PARAM_SPACES

    Example:
        study = optuna.create_study()
        trial = study.ask()
        backend_params, vad_config = suggest_params(trial, "silero")
        vad = create_vad("silero", backend_params, vad_config)
    """
    if vad_type not in PARAM_SPACES:
        available = ", ".join(PARAM_SPACES.keys())
        raise ValueError(f"Unknown VAD type: {vad_type}. Available: {available}")

    space = PARAM_SPACES[vad_type]

    # Suggest backend parameters
    backend_params = _suggest_group(trial, space.get("backend", {}), prefix="backend_")

    # Suggest VADConfig parameters
    vad_config_params = _suggest_group(
        trial, space.get("vad_config", {}), prefix="vad_config_"
    )

    # Create VADConfig if there are parameters
    vad_config = VADConfig(**vad_config_params) if vad_config_params else None

    return backend_params, vad_config


def get_supported_vad_types() -> list[str]:
    """Get list of VAD types that support Bayesian optimization.

    Returns:
        List of VAD type names

    Note:
        JaVAD is not included because it only has 1 parameter (model)
        and should use Grid Search instead.
    """
    return list(PARAM_SPACES.keys())


__all__ = [
    "PARAM_SPACES",
    "suggest_params",
    "get_supported_vad_types",
]
