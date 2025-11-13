import pytest

from livecap_core import i18n


@pytest.fixture
def reset_i18n():
    with i18n.i18n.preserve_state() as manager:
        manager.clear_translator()
        manager.clear_fallbacks()
        yield manager


def test_translate_uses_fallback(reset_i18n):
    i18n.register_fallbacks({"example.key": "Example"})

    assert i18n.translate("example.key") == "Example"


def test_translate_uses_default_when_no_fallback(reset_i18n):
    assert i18n.translate("missing.key", default="Fallback") == "Fallback"


def test_registered_translator_overrides_fallback(reset_i18n):
    def fake_translator(key: str, **kwargs) -> str:
        return f"translated:{key}"

    i18n.register_fallbacks({"example.key": "Example"})
    i18n.register_translator(fake_translator, name="fake-translator", extras=("translation",))

    assert i18n.translate("example.key") == "translated:example.key"


def test_diagnostics_exposes_registration(reset_i18n):
    diagnostics = i18n.diagnose()
    assert diagnostics.translator.registered is False

    def fake_translator(key: str, **kwargs) -> str:
        return "ok"

    i18n.register_fallbacks({"sample": "value"})
    i18n.register_translator(fake_translator, name="fake", extras=("translation",), metadata={"provider": "test"})

    diagnostics = i18n.diagnose()
    assert diagnostics.translator.registered is True
    assert diagnostics.translator.name == "fake"
    assert diagnostics.translator.extras == ("translation",)
    assert diagnostics.translator.metadata.get("provider") == "test"
    assert diagnostics.fallback_count == 1
    assert diagnostics.fallback_keys_sample == ("sample",)
