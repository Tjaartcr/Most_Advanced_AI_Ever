
import asyncio
import edge_tts

async def main():
    communicate = edge_tts.Communicate("Hello world", "en-US-GuyNeural")
    await communicate.save("output.mp3")

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())











##
### edge_tts_trace.py
### Run with same venv: python edge_tts_trace.py
##import asyncio, logging, time, os, traceback, tempfile
##import edge_tts
##
### Enable very verbose logging for aiohttp and edge-tts
##logging.basicConfig(level=logging.DEBUG,
##                    format="%(asctime)s %(levelname)8s %(name)s: %(message)s")
### reduce overly chatty libs if you want later:
### logging.getLogger("chardet.charsetprober").setLevel(logging.INFO)
##
##async def run_once(voice, text, out):
##    print("\n--- Attempting voice:", voice, "->", out, "---\n")
##    try:
##        # edge-tts internal logs + aiohttp logs will appear due to logging.basicConfig
##        comm = edge_tts.Communicate(text, voice=voice)
##        # add a timeout wrapper so hangs are visible
##        await asyncio.wait_for(comm.save(out), timeout=40.0)
##        if os.path.exists(out):
##            size = os.path.getsize(out)
##        else:
##            size = 0
##        print(f"[RESULT] file exists: {os.path.exists(out)} size: {size}")
##    except Exception as e:
##        print("[EXCEPTION]", type(e).__name__, e)
##        traceback.print_exc()
##        # show any partial file
##        try:
##            if os.path.exists(out):
##                print("Partial file size:", os.path.getsize(out))
##        except Exception:
##            pass
##
##def main():
##    text = "Edge tts verbose trace test. If this fails we will have full logs."
##    tmp = tempfile.gettempdir()
##    # try the voice you want and a known-working fallback list
##    voices = ["af-ZA-WillemNeural", "en-US-GuyNeural", "en-GB-LibbyNeural"]
##    for v in voices:
##        out = os.path.join(tmp, f"edge_tts_trace_{v.replace('/','_')}_{int(time.time())}.mp3")
##        try:
##            asyncio.run(run_once(v, text, out))
##        except Exception as e:
##            print("Top-level run error:", type(e).__name__, e)
##            traceback.print_exc()
##
##if __name__ == "__main__":
##    # ensure proxy env vars unset unless you intend to use them
##    os.environ.pop("HTTP_PROXY", None)
##    os.environ.pop("HTTPS_PROXY", None)
##    os.environ.pop("http_proxy", None)
##    os.environ.pop("https_proxy", None)
##
##    print("python:", os.path.realpath(os.sys.executable))
##    main()
##
##
















### edge_tts_network_test.py
### Run with the same python: python edge_tts_network_test.py
##import asyncio, ssl, socket, sys, traceback, time
##import aiohttp
##
##HOSTS = [
##    ("www.bing.com", 443),
##    ("speech.platform.bing.com", 443),
##    ("speech.microsoft.com", 443),
##]
##
##print("Python:", sys.executable, sys.version.splitlines()[0])
##print("Testing TCP connect and TLS get_server_certificate for targets...\n")
##
##for host, port in HOSTS:
##    print(f"--- {host}:{port} ---")
##    try:
##        # TCP connect
##        s = socket.create_connection((host, port), timeout=6)
##        s.close()
##        print("TCP: OK")
##    except Exception as e:
##        print("TCP: FAILED ->", repr(e))
##
##    try:
##        pem = ssl.get_server_certificate((host, port))
##        if pem and pem.startswith("-----BEGIN CERTIFICATE-----"):
##            print("TLS: get_server_certificate OK (certificate retrieved)")
##        else:
##            print("TLS: get_server_certificate returned unexpected content")
##    except Exception as e:
##        print("TLS: FAILED ->", repr(e))
##
##    print()
##
### aiohttp GET for a public website and an example speech host
##async def aio_check():
##    timeout = aiohttp.ClientTimeout(total=15)
##    async with aiohttp.ClientSession(timeout=timeout) as sess:
##        for url in ("https://www.bing.com/", "https://speech.platform.bing.com/"):
##            print("HTTP GET ->", url)
##            try:
##                async with sess.get(url) as r:
##                    text = await r.text(errors="ignore")
##                    print(" Status:", r.status, "Content-Length:", r.headers.get("Content-Length"), "Downloaded:", len(text))
##            except Exception as e:
##                print(" HTTP failed ->", type(e).__name__, e)
##            print()
##try:
##    asyncio.run(aio_check())
##except Exception:
##    traceback.print_exc()
##










##import asyncio
##import edge_tts
##
##async def test():
##    communicate = edge_tts.Communicate("Hello world", "en-US-GuyNeural")
##    await communicate.save("output.mp3")
##
##asyncio.run(test())
##






####"D:\Python_Env\New_Virtual_Env\Alfred_Offline_GUI_Env\Alfred_Offline_Venv\Alfred_cuda_support\Scripts\python.exe" - <<'PY'
##import asyncio, edge_tts, tempfile, os, traceback, time
##async def tts(text, voice):
##    out = os.path.join(tempfile.gettempdir(), f"edge_tts_test_{voice.replace('/','_')}_{int(time.time())}.mp3")
##    try:
##        c = edge_tts.Communicate(text, voice=voice)
##        await c.save(out)
##        print("Wrote:", out, "size:", os.path.getsize(out))
##    except Exception as e:
##        print("Exception for voice", voice, ":", type(e).__name__, e)
##        traceback.print_exc()
##        if os.path.exists(out):
##            print("Partial file size:", os.path.getsize(out))
##    return
##
##voices = ["en-US-GuyNeural", "en-GB-LibbyNeural", "en-US-JennyNeural", "af-ZA-WillemNeural"]
##for v in voices:
##    print("Trying voice:", v)
##    asyncio.run(tts("This is a quick edge-tts test.", v))
##








####"D:\Python_Env\New_Virtual_Env\Alfred_Offline_GUI_Env\Alfred_Offline_Venv\Alfred_cuda_support\Scripts\python.exe" - <<'PY'
##import asyncio, edge_tts, sys, os, tempfile, traceback
##async def test(voice):
##    out = os.path.join(tempfile.gettempdir(), "edge_tts_test.mp3")
##    try:
##        c = edge_tts.Communicate("This is a quick edge-tts test.", voice=voice)
##        await c.save(out)
##        print("Wrote:", out, "size:", os.path.getsize(out))
##    except Exception as e:
##        print("Exception:", type(e).__name__, e)
##        traceback.print_exc()
##        # Attempt to read any partial file
##        if os.path.exists(out):
##            print("Partial file size:", os.path.getsize(out))
##    return
##
### Try a common voice first — change if you'd like:
##voice = "en-US-GuyNeural"
##asyncio.run(test(voice))
