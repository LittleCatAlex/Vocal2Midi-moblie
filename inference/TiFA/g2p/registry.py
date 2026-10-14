from enum import Enum, auto
from typing import Callable


class Language(Enum):
    """Special converter language markers."""

    ANY = auto()


_preprocessor_registry: dict[str, type] = {}
_converter_registry: dict[str, type] = {}


def parse_language(language: str | Language | None) -> tuple[str, ...] | Language | None:
    if language is None or language is Language.ANY:
        return language
    return tuple(t.strip() for t in language.split(","))


def preprocessor(*, id: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        if id in _preprocessor_registry:
            raise ValueError(f"Preprocessor '{id}' is already registered.")
        _preprocessor_registry[id] = cls
        return cls
    return decorator


def converter(*, id: str, language: str | Language | None = None) -> Callable[[type], type]:
    tags = parse_language(language)

    def decorator(cls: type) -> type:
        if id in _converter_registry:
            raise ValueError(f"Converter '{id}' is already registered.")
        _converter_registry[id] = cls
        cls.language = tags
        return cls
    return decorator


def _ensure_loaded():
    if not _converter_registry:
        try:
            import inference.TiFA.g2p.converters  # noqa: F401
        except Exception:
            pass
    if not _preprocessor_registry:
        try:
            import inference.TiFA.g2p.preprocessors  # noqa: F401
        except Exception:
            pass


def get_preprocessor(id: str) -> type:
    _ensure_loaded()
    if id not in _preprocessor_registry:
        try:
            import inference.TiFA.g2p.preprocessors.simple  # noqa: F401
        except Exception:
            pass
    if id not in _preprocessor_registry:
        raise KeyError(f"G2P preprocessor '{id}' not found. Available: {list(_preprocessor_registry.keys())}")
    return _preprocessor_registry[id]


def get_converter(id: str) -> type:
    _ensure_loaded()
    if id not in _converter_registry:
        # Fallback direct submodule imports
        if id in ("chinese-pinyin", "yue-jyutping", "pinyin-lax"):
            try:
                import inference.TiFA.g2p.converters.chinese  # noqa: F401
            except Exception:
                pass
        elif id in ("japanese-kana", "japanese-mecab"):
            try:
                import inference.TiFA.g2p.converters.japanese  # noqa: F401
            except Exception:
                pass
        elif id == "lstm":
            try:
                import inference.TiFA.g2p.converters.lstm  # noqa: F401
            except Exception:
                pass
        elif id == "dictionary":
            try:
                import inference.TiFA.g2p.converters.dictionary  # noqa: F401
            except Exception:
                pass
        elif id in ("passthrough", "characters"):
            try:
                import inference.TiFA.g2p.converters.simple  # noqa: F401
            except Exception:
                pass
        elif id == "japanese-onnx":
            try:
                import inference.TiFA.japanese_onnx  # noqa: F401
            except Exception:
                pass
        elif id == "japanese-lexicon":
            try:
                import inference.TiFA.japanese_lexicon  # noqa: F401
            except Exception:
                pass

    if id not in _converter_registry:
        raise KeyError(f"G2P converter '{id}' not found. Available: {list(_converter_registry.keys())}")
    return _converter_registry[id]


def list_preprocessors() -> list[str]:
    _ensure_loaded()
    return list(_preprocessor_registry.keys())


def list_converters() -> list[str]:
    _ensure_loaded()
    return list(_converter_registry.keys())
