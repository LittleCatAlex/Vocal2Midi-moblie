"""Build the default TiFA G2P pipeline for Vocal2Midi (zh/ja/en).

Mirrors the upstream configs/g2p.yaml converter chain without the pydantic
config machinery of g2p.api: preprocessors (filter-punctuation,
strip-whitespace, lowercase) then chinese-pinyin, japanese-mecab (optional),
lstm (en), dictionary (zh), dictionary (ja), passthrough.
"""
from __future__ import annotations

import logging
from pathlib import Path

import inference.TiFA.g2p.converters  # noqa: F401 - populates the registry
import inference.TiFA.g2p.preprocessors  # noqa: F401
from inference.ja_onnx_g2p import DEFAULT_MODEL_DIR as JA_G2P_MODEL_DIR
from inference.TiFA.g2p.pipeline import G2PPipeline
from inference.TiFA.g2p.registry import get_converter, get_preprocessor

logger = logging.getLogger(__name__)


def build_g2p_pipeline(
    model_dir: str | Path,
    japanese_g2p_engine: str = "kashi-g2p-onnx",
    kashi_g2p_model_dir: str | Path | None = None,
    language: str | None = None,
) -> G2PPipeline:
    """Construct the converter chain rooted at the TiFA model directory."""
    root = Path(model_dir)
    dictionaries = root / "dictionaries"

    converters = []

    # 1. Chinese pinyin
    try:
        pinyin_dict = dictionaries / "ds-zh-pinyin-lite.txt"
        if pinyin_dict.is_file():
            converters.append(get_converter("chinese-pinyin")(dict_path=str(pinyin_dict)))
    except Exception as exc:
        logger.warning(f"Failed to initialize chinese-pinyin converter: {exc}")

    # 2. Cantonese jyutping
    try:
        jyutping_dict_path = dictionaries / "jyutping_dict.txt"
        if jyutping_dict_path.is_file():
            converters.append(get_converter("yue-jyutping")(dict_path=str(jyutping_dict_path)))
    except Exception as exc:
        logger.warning(f"Failed to initialize yue-jyutping converter: {exc}")

    # 3. Japanese converters
    ja_engine = str(japanese_g2p_engine or "kashi-g2p-onnx").strip().lower()
    kashi_path = Path(kashi_g2p_model_dir) if kashi_g2p_model_dir else JA_G2P_MODEL_DIR

    if ja_engine in {"kashi-g2p-onnx", "ja_g2p", "ja_g2p_onnx"} and (kashi_path / "model.onnx").is_file():
        try:
            import inference.TiFA.japanese_onnx  # noqa: F401 - populates the registry
            converters.append(get_converter("japanese-onnx")(
                dict_path=str(dictionaries / "japanese_dict_full.txt"),
                model_dir=str(kashi_path),
            ))
        except Exception as exc:
            logger.warning(f"Japanese ONNX converter unavailable: {exc}")
    elif ja_engine == "pyopenjtalk":
        try:
            import inference.TiFA.japanese_pyopenjtalk  # noqa: F401 - populates the registry
            converters.append(get_converter("japanese-pyopenjtalk")(
                dict_path=str(dictionaries / "japanese_dict_full.txt"),
            ))
        except Exception as e:
            logger.warning(f"Japanese pyopenjtalk converter unavailable: {e}")

    lexicon_path = root / "dictionaries" / "ja_lexicon.txz"
    if lexicon_path.is_file():
        try:
            import inference.TiFA.japanese_lexicon  # noqa: F401 - populates the registry
            converters.append(get_converter("japanese-lexicon")(
                lexicon_path=str(lexicon_path),
                dict_path=str(dictionaries / "japanese_dict_full.txt"),
            ))
        except Exception as exc:
            logger.warning(f"Japanese lexicon converter unavailable: {exc}")

    try:
        mecab = get_converter("japanese-mecab")(
            dict_path=str(dictionaries / "japanese_dict_full.txt"),
            nbest=32,
            double_written_sokuon=False,
        )
        mecab.convert("あ")
        converters.append(mecab)
    except Exception as e:
        logger.debug(f"Japanese MeCab converter unavailable, falling back to kana: {e}")
        try:
            kana = get_converter("japanese-kana")(
                dict_path=str(dictionaries / "japanese_dict_full.txt"),
                double_written_sokuon=False,
            )
            converters.append(kana)
        except Exception as exc:
            logger.warning(f"Japanese kana converter unavailable: {exc}")

    # 4. English LSTM converter
    try:
        lstm_model_dir = root / "assets" / "LstmG2p-Eng"
        lstm_dict = dictionaries / "ds_cmudict-07b.txt"
        if lstm_model_dir.is_dir() and lstm_dict.is_file():
            lstm = get_converter("lstm")(
                dict_path=str(lstm_dict),
                model_path=str(lstm_model_dir),
                beam_size=16,
            )
            lstm.language = ("en",)
            converters.append(lstm)
    except Exception as exc:
        logger.warning(f"LSTM English converter unavailable: {exc}")

    # 5. Dictionaries
    try:
        zh_dict = dictionaries / "ds-zh-pinyin-lite.txt"
        if zh_dict.is_file():
            zh_dictionary = get_converter("dictionary")(dict_path=str(zh_dict))
            zh_dictionary.language = ("zh",)
            lax_pinyin = get_converter("pinyin-lax")(dict_path=str(zh_dict))
            converters.extend([zh_dictionary, lax_pinyin])
    except Exception as exc:
        logger.warning(f"Chinese dictionary converter unavailable: {exc}")

    try:
        ja_dict = dictionaries / "japanese_dict_full.txt"
        if ja_dict.is_file():
            ja_dictionary = get_converter("dictionary")(dict_path=str(ja_dict))
            ja_dictionary.language = ("ja",)
            converters.append(ja_dictionary)
    except Exception as exc:
        logger.warning(f"Japanese dictionary converter unavailable: {exc}")

    try:
        jyutping_dict_path = dictionaries / "jyutping_dict.txt"
        if jyutping_dict_path.is_file():
            yue_dictionary = get_converter("dictionary")(dict_path=str(jyutping_dict_path))
            yue_dictionary.language = ("yue",)
            converters.append(yue_dictionary)
    except Exception as exc:
        logger.warning(f"Cantonese dictionary converter unavailable: {exc}")

    # 6. Passthrough fallback
    try:
        converters.append(get_converter("passthrough")())
    except Exception as exc:
        logger.error(f"Failed to initialize passthrough converter: {exc}")

    # Preprocessors
    preprocessors = []
    for pid in ("filter-punctuation", "strip-whitespace", "lowercase"):
        try:
            preprocessors.append(get_preprocessor(pid)())
        except Exception as exc:
            logger.warning(f"Failed to load preprocessor '{pid}': {exc}")

    return G2PPipeline(
        preprocessors=preprocessors,
        converters=converters,
    )
