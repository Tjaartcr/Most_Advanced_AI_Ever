



"""
af_to_en.py - patched

Simple Afrikaans -> English translator module.

Changes applied:
- Adds a backward-compat shim for huggingface_hub.cached_download -> hf_hub_download.
- Lazy-loads Transformers pipeline inside _get_translator to avoid import-time HF network calls.
- Robust device selection (respects explicit device argument).
- Defensive error handling so import failures raise clear exceptions.

Usage (from other modules):
    from af_to_en import translate, translate_batch
    text_en = translate("Goeie more, hoe gaan dit?")
    texts_en = translate_batch(["Goeie more", "Dankie"]) 
"""
from typing import List, Union, Optional
import logging

_LOG = logging.getLogger(__name__)

# === Backward compatibility shim for huggingface_hub.cached_download ===
try:
    from huggingface_hub import cached_download  # type: ignore
except Exception:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception:
        hf_hub_download = None

    def cached_download(*args, **kwargs):
        if hf_hub_download is None:
            raise ImportError("hf_hub_download not available to emulate cached_download")
        if len(args) >= 2 and 'repo_id' not in kwargs and 'filename' not in kwargs:
            repo_id, filename, *rest = args
            return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
        return hf_hub_download(*args, **kwargs)

    try:
        import importlib
        _hf_mod = importlib.import_module("huggingface_hub")
        setattr(_hf_mod, "cached_download", cached_download)
    except Exception:
        pass


# model id (Helsinki-NLP)
_MODEL_NAME = "Helsinki-NLP/opus-mt-af-en"

# module-level pipeline singleton (lazy)
_translator = None


def _get_translator(device: Union[int, None] = None):
    """Lazy-load and cache the HF translation pipeline.

    device semantics:
      - device is None: use GPU 0 if available, else CPU (-1)
      - device is int: use that device index (use -1 for CPU)
    """
    global _translator
    if _translator is not None:
        return _translator

    try:
        import torch
        from transformers import pipeline
    except Exception as e:
        _LOG.exception("transformers or torch not available: %s", e)
        raise

    # Respect explicit device, otherwise use GPU 0 when available
    device_arg = 0 if (device is None and torch.cuda.is_available()) else (device if device is not None else -1)

    try:
        _translator = pipeline(
            "translation",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            device=device_arg,
            truncation=True,
        )
    except Exception as e:
        _LOG.exception("Failed to create translation pipeline: %s", e)
        raise

    return _translator


def translate(text: str, device: Union[int, None] = None) -> str:
    """Translate a single Afrikaans string to English. Returns the translated string."""
    if not text:
        return ""
    pipe = _get_translator(device)
    try:
        out = pipe(text)
    except TypeError:
        # Some pipeline versions may require different call signature
        out = pipe(text, truncation=True)
    except Exception as e:
        _LOG.exception("Translation call failed: %s", e)
        return text

    if isinstance(out, list) and out:
        return out[0].get("translation_text", "")
    if isinstance(out, dict):
        return out.get("translation_text", "")
    return str(out)


def translate_batch(texts: List[str], device: Union[int, None] = None) -> List[str]:
    """Translate a list of Afrikaans strings to English."""
    if not texts:
        return []
    pipe = _get_translator(device)
    try:
        outs = pipe(texts)
    except TypeError:
        outs = pipe(texts, truncation=True)
    except Exception as e:
        _LOG.exception("Batch translation failed: %s", e)
        # fallback: return originals to avoid crash
        return texts

    results = []
    for o in outs:
        if isinstance(o, dict):
            results.append(o.get("translation_text", ""))
        else:
            results.append(str(o))
    return results


# optional CLI for quick manual test
if __name__ == "__main__":
    examples = [
        "Goeie more, hoe gaan dit met jou?",
        "Waar is die biblioteek?"
    ]
    print("AF -> EN examples:")
    for s in examples:
        print("AF:", s)
        print("EN:", translate(s))
        print("---")





### af_to_en.py
##"""
##Simple Afrikaans -> English translator module.
##
##Usage (from other modules):
##    from af_to_en import translate, translate_batch
##    text_en = translate("Goeie more, hoe gaan dit?")
##    texts_en = translate_batch(["Goeie more", "Dankie"])
##
##This module lazy-loads a Hugging Face MarianMT model the first time you call translate(...).
##"""
##from typing import List, Union
##import torch
##from transformers import pipeline
##
### model id (Helsinki-NLP)
##_MODEL_NAME = "Helsinki-NLP/opus-mt-af-en"
##
### module-level pipeline singleton (lazy)
##_translator = None
##
##def _get_translator(device: Union[int, None] = None):
##    global _translator
##    if _translator is None:
##        # device: None => auto (GPU if available)
##        if device is None:
##            device_arg = 0 if torch.cuda.is_available() else -1
##        else:
##            device_arg = device
##        _translator = pipeline(
##            "translation",
##            model=_MODEL_NAME,
##            tokenizer=_MODEL_NAME,
##            device=device_arg,
##            truncation=True,
##        )
##    return _translator
##
##def translate(text: str, device: Union[int, None] = None) -> str:
##    """
##    Translate a single Afrikaans string to English.
##    Returns the translated string.
##    """
##    if not text:
##        return ""
##    pipe = _get_translator(device)
##    out = pipe(text)
##    # pipeline returns list[dict] for single string too
##    return out[0].get("translation_text", "")
##
##def translate_batch(texts: List[str], device: Union[int, None] = None) -> List[str]:
##    """
##    Translate a list of Afrikaans strings to English.
##    """
##    if not texts:
##        return []
##    pipe = _get_translator(device)
##    outs = pipe(texts)
##    return [o.get("translation_text", "") for o in outs]
##
### optional CLI for quick manual test
##if __name__ == "__main__":
##    examples = [
##        "Goeie more, hoe gaan dit met jou?",
##        "Waar is die biblioteek?"
##    ]
##    print("AF -> EN examples:")
##    for s in examples:
##        print("AF:", s)
##        print("EN:", translate(s))
##        print("---")
