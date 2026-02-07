






# hf_compat.py
"""
Compatibility shims for offline or older Torch/HuggingFace setups.
Load this before any import of `transformers` or `torch`.
"""

import importlib

# --- cached_download patch (for older huggingface_hub) ---
try:
    from huggingface_hub import cached_download  # type: ignore
except Exception:
    try:
        from huggingface_hub import hf_hub_download
    except Exception:
        hf_hub_download = None

    def cached_download(*args, **kwargs):
        if hf_hub_download is None:
            raise ImportError("hf_hub_download not available to emulate cached_download")
        if len(args) >= 2 and 'repo_id' not in kwargs and 'filename' not in kwargs:
            repo_id, filename, *rest = args
            return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
        return hf_hub_download(*args, **kwargs)

    _hf_mod = importlib.import_module("huggingface_hub")
    setattr(_hf_mod, "cached_download", cached_download)

# --- pytree patch for older torch versions ---
try:
    import torch
    pytree = getattr(torch.utils, "_pytree", None)
    if pytree and not hasattr(pytree, "register_pytree_node") and hasattr(pytree, "_register_pytree_node"):

        # Wrap the old function to safely accept new keyword args
        def _register_pytree_node_adapter(*args, **kwargs):
            # Drop new kwargs that older PyTorch doesn't recognize
            kwargs.pop("serialized_type_name", None)
            kwargs.pop("serialized_value_fn", None)
            return pytree._register_pytree_node(*args, **kwargs)

        pytree.register_pytree_node = _register_pytree_node_adapter

except Exception:
    pass




####
####
##### hf_compat.py  — combine all your compatibility shims here
####
##### --- cached_download patch (optional, if you already added it, keep as-is) ---
####try:
####    from huggingface_hub import cached_download  # type: ignore
####except Exception:
####    try:
####        from huggingface_hub import hf_hub_download
####    except Exception:
####        hf_hub_download = None
####    def cached_download(*args, **kwargs):
####        if hf_hub_download is None:
####            raise ImportError("hf_hub_download not available to emulate cached_download")
####        if len(args) >= 2 and 'repo_id' not in kwargs and 'filename' not in kwargs:
####            repo_id, filename, *rest = args
####            return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
####        return hf_hub_download(*args, **kwargs)
####    import importlib
####    _hf_mod = importlib.import_module("huggingface_hub")
####    setattr(_hf_mod, "cached_download", cached_download)
####
####
##### --- pytree patch for older torch versions ---
####try:
####    import torch
####    pytree = getattr(torch.utils, "_pytree", None)
####    if pytree and not hasattr(pytree, "register_pytree_node") and hasattr(pytree, "_register_pytree_node"):
####        pytree.register_pytree_node = pytree._register_pytree_node
####except Exception:
####    pass



### hf_compat.py  — place this in project root and import it at top of main.py
### Provides cached_download if missing by delegating to hf_hub_download.
##
##try:
##    # If cached_download already exists, nothing to do
##    from huggingface_hub import cached_download  # type: ignore
##except Exception:
##    try:
##        from huggingface_hub import hf_hub_download  # type: ignore
##    except Exception:
##        hf_hub_download = None
##
##    def cached_download(*args, **kwargs):
##        if hf_hub_download is None:
##            raise ImportError("hf_hub_download not available to emulate cached_download")
##        # Common legacy pattern: cached_download(repo_id, filename, ...)
##        if len(args) >= 2 and 'repo_id' not in kwargs and 'filename' not in kwargs:
##            repo_id, filename, *rest = args
##            return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
##        return hf_hub_download(*args, **kwargs)
##
##    # Inject into huggingface_hub namespace so imports succeed
##    try:
##        import importlib
##        _hf_mod = importlib.import_module("huggingface_hub")
##        setattr(_hf_mod, "cached_download", cached_download)
##    except Exception:
##        # best-effort; if huggingface_hub isn't importable yet, import will succeed later and shim will be available
##        pass
