### save as dump_env.py
##import os
##
##output_file = "env_dump.txt"
##
##with open(output_file, "w", encoding="utf-8") as f:
##    for k, v in os.environ.items():
##        f.write(f"{k}={v}\n")
##
##print(f"[INFO] Environment variables dumped to {output_file}")



# env_dump.py
import os, json, shutil, sys, platform

keys = sorted(k for k in os.environ.keys() if k.startswith("OLLAMA") or k.startswith("CUDA") or k in ("PATH","PROCESSOR_IDENTIFIER"))
info = {k: os.environ.get(k) for k in keys}
info["_python_executable"] = sys.executable
info["_cwd"] = os.getcwd()
info["_platform"] = platform.platform()
info["_ollama_path"] = shutil.which("ollama")
print(json.dumps(info, indent=2, ensure_ascii=False))
