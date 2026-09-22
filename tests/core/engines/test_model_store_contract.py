"""ModelRoot 契約の docs と実装が一致していること (Issue #456)。

``docs/architecture/model-store-contract.md`` は「どこに何が置かれるか」の SSoT で、
明示例外 (wheel 同梱の VAD 資産) の表は ``model_store.MODEL_STORE_EXEMPT_ASSETS`` と
同じ集合でなければならない。片方だけ増やすと「契約の対象 / 例外」の一覧が嘘になる。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from livecap_cli.engines.model_store import MANIFEST_NAME, MODEL_STORE_EXEMPT_ASSETS

DOC = Path(__file__).resolve().parents[3] / "docs" / "architecture" / "model-store-contract.md"


def test_contract_doc_exists_and_names_the_manifest():
    text = DOC.read_text(encoding="utf-8")
    assert MANIFEST_NAME in text
    assert "validate_repo_dir" in text and "publish_dir" in text and "fetch_repo_dir" in text


@pytest.mark.parametrize("asset", sorted(MODEL_STORE_EXEMPT_ASSETS))
def test_every_exempt_asset_is_documented(asset):
    text = DOC.read_text(encoding="utf-8")
    label = {"silero_vad": "Silero VAD", "ten_vad": "TenVAD"}[asset]
    assert label in text, f"{asset} が docs の明示例外表に無い"


def test_doc_lists_no_undeclared_exemptions():
    """docs の例外表 (§5) にある VAD 行が実装の集合と同じであること。"""
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## 5. 明示例外")[1].split("## 6.")[0]
    documented = {name for name, label in {"silero_vad": "Silero VAD", "ten_vad": "TenVAD"}.items() if label in section}
    assert documented == set(MODEL_STORE_EXEMPT_ASSETS)


@pytest.mark.parametrize("asset", sorted(MODEL_STORE_EXEMPT_ASSETS))
def test_exempt_assets_are_install_owned(asset):
    """例外は wheel 同梱であること: package が入っていれば site-packages 配下に実体がある。"""
    spec = importlib.util.find_spec(asset)
    if spec is None or spec.origin is None:
        pytest.skip(f"{asset} 未導入")
    origin = Path(spec.origin).resolve()
    assert "site-packages" in origin.parts or "dist-packages" in origin.parts, origin


def _repo_ids_used_by_production() -> set[str]:
    from livecap_cli.engines.metadata import EngineMetadata
    from livecap_cli.engines.reazonspeech_engine import ReazonSpeechEngine
    from livecap_cli.engines.whispers2t_engine import MODEL_REPOS
    from livecap_cli.translation.impl.riva_instruct import RivaInstructTranslator
    from livecap_cli.translation.lang_codes import get_opus_mt_model_name

    used = set(MODEL_REPOS.values()) | {ReazonSpeechEngine.HF_REPO_ID, RivaInstructTranslator.MODEL_NAME}
    used |= {get_opus_mt_model_name("ja", "en"), get_opus_mt_model_name("en", "ja")}
    for info in EngineMetadata.get_all().values():
        model_name = info.default_params.get("model_name")
        if isinstance(model_name, str) and "/" in model_name:
            used.add(model_name)
    assert {"Qwen/Qwen3-ASR-0.6B", "mistralai/Voxtral-Mini-3B-2507", "nvidia/parakeet-tdt-0.6b-v2", "nvidia/canary-1b-flash"} <= used, "metadata から拾えていない"
    return used


def test_known_model_repos_cover_every_engine_and_translator_repo_id():
    """``scan_external_caches`` (``livecap-cli info``) は engine を import せず ``KNOWN_MODEL_REPOS`` で
    「cli が使う repo」を判定する。engine / translator が使う実 repo id が増えたらここが落ちる (#453)。"""
    import fnmatch

    from livecap_cli.engines.legacy_model_layouts import KNOWN_MODEL_REPOS

    used = _repo_ids_used_by_production()
    uncovered = sorted(r for r in used if not any(fnmatch.fnmatchcase(r, k.pattern) for k in KNOWN_MODEL_REPOS))
    assert uncovered == [], f"KNOWN_MODEL_REPOS に無い repo: {uncovered}"


def test_known_repo_destinations_match_the_engines_real_destination_paths():
    """``adopted`` は ``KnownRepo.destinations`` が ``models_root`` 相対の**実際の正本 path** と
    一致していて初めて意味を持つ (PR #463 レビュー MEDIUM: 1 repo に 2 正本の ReazonSpeech)。
    engine / translator に実際の path を聞いて突き合わせる。"""
    from livecap_cli.engines.canary_engine import CanaryEngine
    from livecap_cli.engines.legacy_model_layouts import _known_repo
    from livecap_cli.engines.parakeet_engine import ParakeetEngine
    from livecap_cli.engines.qwen3asr_engine import Qwen3ASREngine
    from livecap_cli.engines.reazonspeech_engine import ReazonSpeechEngine
    from livecap_cli.engines.voxtral_engine import VoxtralEngine
    from livecap_cli.engines.whispers2t_engine import WhisperS2TEngine

    root = Path("R")
    actual: dict[str, set[str]] = {}

    def record(repo_id: str, path: Path) -> None:
        actual.setdefault(repo_id, set()).add(path.relative_to(root).as_posix())

    for engine in (
        Qwen3ASREngine(device="cpu"),
        WhisperS2TEngine(device="cpu", model_size="base", language="en"),
        VoxtralEngine(device="cpu", language="en"),
        ReazonSpeechEngine(device="cpu", use_int8=False),
        ReazonSpeechEngine(device="cpu", use_int8=True),
    ):
        record(engine._repo_dir_spec().repo_id, engine._get_local_model_path(root))
    for engine, repo_id in (
        (ParakeetEngine(device="cpu"), "nvidia/parakeet-tdt-0.6b-v2"),
        (ParakeetEngine(device="cpu", engine_name="parakeet_ja"), "nvidia/parakeet-tdt_ctc-0.6b-ja"),
        (CanaryEngine(device="cpu", language="en"), "nvidia/canary-1b-flash"),
    ):
        record(repo_id, engine._get_local_model_path(root))

    with patch("livecap_cli.resources.get_model_manager") as manager:
        manager.return_value.get_models_dir.return_value = root
        manager.return_value.models_root = root
        from livecap_cli.translation.impl.opus_mt import OpusMTTranslator
        from livecap_cli.translation.impl.riva_instruct import RivaInstructTranslator

        opus = OpusMTTranslator(source_lang="ja", target_lang="en")
        record(opus.model_name, opus.model_dir)
        riva = RivaInstructTranslator(device="cpu")
        record(riva.MODEL_NAME, riva.model_dir)

    assert set(actual) <= _repo_ids_used_by_production()
    for repo_id, paths in sorted(actual.items()):
        known = _known_repo(repo_id)
        assert known is not None, repo_id
        declared = set(known.resolve(repo_id))
        assert paths <= declared, f"{repo_id}: 宣言に無い正本 {sorted(paths - declared)}"
        # ReazonSpeech は variant ごとに別 engine instance なので、1 instance では片方しか出ない。
        # 宣言側に余分があってはならない (消してよいと表示してしまう) ので、repo 単位で突き合わせる
        if repo_id == ReazonSpeechEngine.HF_REPO_ID:
            assert declared == paths, f"{repo_id}: 宣言と実際が不一致 {sorted(declared ^ paths)}"
