"""Utilities for normalising application config to the core schema."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, MutableMapping, Sequence

from livecap_core.config.defaults import get_default_config, merge_config
from livecap_core.config.validator import ConfigValidator


CoreConfigDict = Dict[str, Any]

# Core が扱うトップレベルセクション（Phase 1 合意済み）
_CORE_SECTIONS = (
    "audio",
    "multi_source",
    "silence_detection",
    "transcription",
    "translation",
    "engines",
    "logging",
    "queue",
    "debug",
    "file_mode",
)

_TRANSLATION_KEY_MAP = {
    "enable_translation": "enabled",
    "translation_service": "service",
}


def build_core_config(raw_config: Mapping[str, Any] | None) -> CoreConfigDict:
    """Normalize GUI/raw configuration into the Core schema.

    - 不要な GUI 専用キーを除去
    - 翻訳設定のキー（enable_translation → enabled など）を正規化
    - `multi_source.sources` の dict / Sequence 両対応をハンドリング
    - CoreConfig TypedDict に沿って `ConfigValidator` を適用
    """

    source = raw_config or {}
    defaults = get_default_config()
    core_config: CoreConfigDict = {}

    for section in _CORE_SECTIONS:
        default_section = deepcopy(defaults[section])
        if section == "translation":
            normalized = _normalize_translation(source.get("translation"))
            core_config[section] = (
                merge_config(default_section, normalized) if normalized else default_section
            )
        elif section == "multi_source":
            normalized = _normalize_multi_source(source.get("multi_source"))
            core_config[section] = (
                merge_config(default_section, normalized) if normalized else default_section
            )
        else:
            override = source.get(section)
            if isinstance(default_section, dict):
                override_mapping = override if isinstance(override, Mapping) else {}
                core_config[section] = merge_config(default_section, override_mapping)
            else:
                core_config[section] = override if override is not None else default_section

    ConfigValidator.validate_or_raise(core_config)
    return core_config


def _normalize_translation(raw_translation: Any) -> Dict[str, Any]:
    if not isinstance(raw_translation, Mapping):
        return {}

    normalized: Dict[str, Any] = {}
    for old_key, new_key in _TRANSLATION_KEY_MAP.items():
        if old_key in raw_translation:
            normalized[new_key] = raw_translation[old_key]
    if "enabled" in raw_translation:
        normalized["enabled"] = raw_translation["enabled"]
    if "service" in raw_translation:
        normalized["service"] = raw_translation["service"]
    if "target_language" in raw_translation:
        normalized["target_language"] = raw_translation["target_language"]

    for nested_key in ("performance", "riva_settings"):
        if nested_key in raw_translation and isinstance(raw_translation[nested_key], Mapping):
            normalized[nested_key] = dict(raw_translation[nested_key])

    return normalized


def _normalize_multi_source(raw_multi_source: Any) -> Dict[str, Any]:
    if not isinstance(raw_multi_source, Mapping):
        return {}

    normalized: Dict[str, Any] = {}
    for key, value in raw_multi_source.items():
        if key == "sources":
            normalized["sources"] = _normalize_sources(value)
        else:
            normalized[key] = value
    return normalized


def _normalize_sources(raw_sources: Any) -> MutableMapping[str, Dict[str, Any]]:
    normalized: MutableMapping[str, Dict[str, Any]] = {}

    if isinstance(raw_sources, Mapping):
        for source_id, data in raw_sources.items():
            normalized[str(source_id)] = _normalize_single_source(str(source_id), data)
    elif isinstance(raw_sources, Sequence):
        for entry in raw_sources:
            if isinstance(entry, Mapping):
                source_id = str(entry.get("id") or f"source{len(normalized) + 1}")
                data = dict(entry)
                data.pop("id", None)
                normalized[source_id] = _normalize_single_source(source_id, data)
    return normalized


def _normalize_single_source(source_id: str, data: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if isinstance(data, Mapping):
        result.update(data)
    result.setdefault("id", source_id)
    return result
