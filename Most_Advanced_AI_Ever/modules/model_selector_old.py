import os

MODEL_DIR = "C://Users//ITF//.ollama//models//manifests//registry.ollama.ai//library"

# Load model names once
try:
    MODEL_NAMES = os.listdir(MODEL_DIR)
except FileNotFoundError:
    MODEL_NAMES = []
    print(f"[ERROR] Model directory not found: {MODEL_DIR}")


EXCLUSIONS = {
    "thinkbot": [
        "deepseek-r1", "code", "llava", "moondream", "vision", "-embed-", "chatqa"
    ],
    "chatbot": [
        "deepseek-r1", "codegemma", "deepseek-coder", "deepseek-r1:8b", "dolphin-phi", "gemma",
        "granite3.1-moe", "granite3.2-vision", "granite3.3", "llama2", 
        "llama3.2", "llava", "llava-phi3", "mistral", "moondream", "mxbai-embed-large",
        "nomic-embed-text", "qwen2.5-coder", "stablelm2", "starcoder2", "yi-coder"
    ],
    "vision": [
        "phi", "qwen", "stablelm2", "mistral", "granite3.3", "deepseek-r1", "dolphin-phi", "gemma",
        "granite3.1-moe", "code", "-embed-", "llama"
    ],
    "coding": [
        "llava", "granite3.2-vision", "moondream", "phi", "qwen", "granite3.3", "dolphin-phi",
        "gemma", "granite3.1-moe", "-embed-", "llama", "chatqa", "deepseek-r1"
    ],
}

ADDITIONS = {
    "thinkbot": ["deepseek-r1:8b", "deepseek-r1:1b"],
    "chatbot": ["llama3-chatqa"],
    "vision": ["llava-phi3"],
    "coding": ["deepseek-r1:8b"]
}


def get_models_by_type(model_type: str):
    """Return filtered model list for a given model type."""
    if model_type not in EXCLUSIONS:
        raise ValueError(f"Invalid model type: {model_type}. Must be one of {list(EXCLUSIONS.keys())}")

    excluded_words = EXCLUSIONS[model_type]
    models = []

    for model in MODEL_NAMES:
        if not any(word in model.lower() for word in excluded_words):
            models.append(model)

    # Add manual inclusions if needed
    models += ADDITIONS.get(model_type, [])

    return models
