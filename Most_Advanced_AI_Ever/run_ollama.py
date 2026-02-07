
import os, subprocess, time

env = os.environ.copy()
env.pop("OLLAMA_NO_GPU", None)
env["OLLAMA_GPU_LAYERS"] = "15"
env["OLLAMA_CTX"] = "2048"
env["OLLAMA_BATCH_SIZE"] = "128"

proc = subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# optional: wait for server to come up (simple)
start = time.time()
while time.time() - start < 30:
    line = proc.stdout.readline()
    if not line:
        time.sleep(0.1)
        continue
    print(line, end='')
    if "Listening on" in line:
        print("Ollama serve started")
        break
else:
    raise RuntimeError("Ollama serve did not start in time; check logs.")



