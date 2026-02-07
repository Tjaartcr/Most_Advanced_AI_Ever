





##from langdetect import detect
##from typing import Tuple, Optional
##
##def detect_language_flags(text: str) -> Tuple[Optional[str], bool, bool]:
##    """
##    Returns: (lang_code or None, is_english, is_afrikaans)
##    """
##    try:
##        lang = detect(text)
##    except Exception:
##        return None, False, False
##
##    is_en = lang == "en"
##    is_af = lang == "af"
##    return lang, is_en, is_af
##




from langdetect import detect

english_spoken = False
afrikaans_spoken = False

def detect_language(text: str):
    global english_spoken, afrikaans_spoken
    english_spoken = False
    afrikaans_spoken = False
    try:
        lang = detect(text)
        if lang == "af":
            afrikaans_spoken = True
            return "af"
        elif lang == "en":
            english_spoken = True
            return "en"
        else:
            return lang
    except Exception:
        return None

def is_afrikaans():
    return afrikaans_spoken

def is_english():
    return english_spoken
