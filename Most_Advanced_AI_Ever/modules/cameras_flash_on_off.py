##
##import urllib.request
##import urllib.error
##import socket
##from urllib.parse import urlparse
##
##def _normalize_host_port(addr: str):
##    """
##    Return (scheme, host, port) where host is IP and port may be None.
##    Accepts:
##      - "192.168.1.10"
##      - "192.168.1.10:81"
##      - "http://192.168.1.10:81"
##      - "http://192.168.1.10"
##    """
##    if not addr:
##        return None, None, None, "No address provided"
##    a = addr.strip()
##    if not a.startswith("http://") and not a.startswith("https://"):
##        a = "http://" + a
##    parsed = urlparse(a)
##    scheme = parsed.scheme or "http"
##    host = parsed.hostname
##    port = parsed.port  # None if not specified
##    if not host:
##        return None, None, None, f"Could not parse host from '{addr}'"
##    return scheme, host, port, None
##
##def _try_http_flash(addr: str, on: bool, timeout: float = 3.0, retries: int = 1):
##    """
##    Improved HTTP attempt:
##    - Tries the original host:port/path first
##    - On 404 or failure it will try the same host on port 80 (no :81)
##    - Prints each attempted URL and returns detailed info
##    Returns (success:bool, message:str)
##    """
##    if not addr:
##        return False, "No IP provided for HTTP control"
##
##    path = "/flash/on" if on else "/flash/off"
##    scheme, host, port, err = _normalize_host_port(addr)
##    if err:
##        return False, err
##
##    tried_results = []
##    candidates = []
##
##    # Candidate 1: use the original parsed port (if any) or assume none
##    if port:
##        candidates.append((scheme, host, port))
##    else:
##        # if no port specified, first try host without port (implies 80)
##        candidates.append((scheme, host, None))
##
##    # Always add explicit port 80 as an alternative (if different from original)
##    # (avoid duplicate)
##    if not (port == 80 or port is None):
##        candidates.append((scheme, host, 80))
##    elif port == 80:
##        # port already 80, but ensure a host-without-port candidate exists
##        candidates.append((scheme, host, None))
##
##    # Deduplicate while preserving order
##    seen = set()
##    cand_final = []
##    for s, h, p in candidates:
##        key = f"{s}://{h}:{p if p else 80}"
##        if key not in seen:
##            seen.add(key)
##            cand_final.append((s, h, p))
##
##    last_error = None
##    for attempt_num, (s, h, p) in enumerate(cand_final, start=1):
##        # build url
##        if p:
##            url = f"{s}://{h}:{p}{path}"
##        else:
##            url = f"{s}://{h}{path}"
##        try:
##            print(f"[flash] HTTP request -> {url} (candidate {attempt_num}/{len(cand_final)})")
##            with urllib.request.urlopen(url, timeout=timeout) as resp:
##                body = resp.read().decode(errors="ignore")
##                msg = f"HTTP OK ({resp.status}) from {url} - {body[:200]}"
##                print(f"[flash] Success: {msg}")
##                return True, msg
##        except urllib.error.HTTPError as e:
##            # if 404, the server answered but didn't find the endpoint on that port
##            note = f"HTTPError {e.code} from {url}: {e.reason}"
##            print(f"[flash] HTTPError: {note}")
##            tried_results.append(note)
##            last_error = note
##            # try next candidate (e.g., port 80)
##            continue
##        except urllib.error.URLError as e:
##            note = f"URLError from {url}: {e.reason}"
##            print(f"[flash] URLError: {note}")
##            tried_results.append(note)
##            last_error = note
##            # try next candidate (maybe different port)
##            continue
##        except socket.timeout:
##            note = f"Timeout from {url}"
##            print(f"[flash] Timeout: {note}")
##            tried_results.append(note)
##            last_error = note
##            continue
##        except Exception as e:
##            note = f"Unknown error from {url}: {e}"
##            print(f"[flash] Unknown error: {note}")
##            tried_results.append(note)
##            last_error = note
##            continue
##
##    # If we get here, none of the candidates succeeded
##    detail = "; ".join(tried_results) if tried_results else (last_error or "No attempts made")
##    return False, f"All HTTP attempts failed for {addr}. Details: {detail}"




import threading
import socket
import urllib.request
import urllib.error
from urllib.parse import urlparse

def _normalize_addr(addr: str):
    """
    Normalize an address that may be:
      - "192.168.1.10"
      - "192.168.1.10:81/stream"
      - "http://192.168.1.10"
      - "http://192.168.1.10:81/stream"
    Returns (scheme, host, error) where host includes optional port (e.g. '192.168.1.10:81').
    """
    if not addr:
        return None, None, None, "No address provided"
    addr = addr.strip()
    parse_target = addr if addr.startswith(("http://", "https://")) else "http://" + addr
    parsed = urlparse(parse_target)
    scheme = parsed.scheme or "http"
    host = parsed.hostname
    port = parsed.port
    if not host:
        return None, None, None, f"Could not parse host from '{addr}'"
    return scheme, host, port, None


##    if not addr:
##        return None, None, "No address provided"
##    addr = addr.strip()
##    # Ensure we have a scheme so urlparse populates netloc
##    parse_target = addr if addr.startswith("http://") or addr.startswith("https://") else "http://" + addr
##    parsed = urlparse(parse_target)
##    host = parsed.netloc  # netloc includes host:port if present
##    scheme = parsed.scheme or "http"
##    if not host:
##        return None, None, f"Could not parse host from '{addr}'"
##    # remove any trailing slashes
##    host = host.rstrip("/")
##    return scheme, host, None




def _try_http_flash(addr: str, on: bool, timeout: float = 3.0):
    """
    Send flash ON/OFF command via HTTP to ESP32-CAM.
    Accepts forms like:
      - "192.168.1.10"
      - "192.168.1.10:81"
      - "http://192.168.1.10:81/stream"
      - "192.168.1.10/stream"
    Returns (success: bool, message: str)
    """

    """
    Send flash ON/OFF command via HTTP to ESP32-CAM.
    Always strips the port so only the host/IP is used.
    """
    if not addr:
        return False, "No IP provided for HTTP control"

    addr = addr.strip()

    # Ensure scheme so urlparse fills fields
    parse_target = addr if addr.startswith(("http://", "https://")) else "http://" + addr
    parsed = urlparse(parse_target)

    # Force only the hostname (no port)
    host = parsed.hostname
    if not host:
        return False, f"Could not parse host from '{addr}'"

    path = "/flash/on" if on else "/flash/off"
    url = f"http://{host}{path}"   # ✅ no port included

    try:
        print(f"[flash] HTTP request -> {url}")
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode(errors="ignore")
            msg = f"HTTP OK ({resp.status}) from {url} - {body[:200]}"
            print(f"[flash] Success: {msg}")
            return True, msg
    except urllib.error.HTTPError as e:
        note = f"HTTPError {e.code} from {url}: {e.reason}"
        print(f"[flash] HTTPError: {note}")
        return False, note
    except urllib.error.URLError as e:
        note = f"URLError from {url}: {e.reason}"
        print(f"[flash] URLError: {note}")
        return False, note
    except socket.timeout:
        note = f"Timeout from {url}"
        print(f"[flash] Timeout: {note}")
        return False, note
    except Exception as e:
        note = f"Unknown error from {url}: {e}"
        print(f"[flash] Unknown error: {note}")
        return False, note






##    if not addr:
##        return False, "No IP provided for HTTP control"
##
##    addr = addr.strip()
##
##    # Ensure scheme so urlparse fills netloc
##    parse_target = addr if addr.startswith(("http://", "https://")) else "http://" + addr
##    parsed = urlparse(parse_target)
##
##    # Use netloc (host[:port]) so we preserve the port if present
##    hostport = parsed.netloc
##    if not hostport:
##        return False, f"Could not parse host from '{addr}'"
##
##    # Remove any trailing path elements like /stream
##    # Some callers might pass "192.168.1.10:81/stream" — we only want host:port
##    # netloc already excludes path so hostport is OK. But if someone passed "192.168.1.10/stream" without scheme,
##    # urlparse returned netloc="" and path="/stream", so try to extract from path in that rare case:
##    if hostport == "" and parsed.path:
##        # try to get first path segment if it looks like host
##        maybe = parsed.path.lstrip('/')
##        if ":" in maybe or maybe.replace('.', '').isdigit():
##            hostport = maybe.split('/')[0]
##
##    path = "/flash/on" if on else "/flash/off"
##    url = f"http://{hostport}{path}"
##
##    try:
##        print(f"[flash] HTTP request -> {url}")
##        with urllib.request.urlopen(url, timeout=timeout) as resp:
##            body = resp.read().decode(errors="ignore")
##            msg = f"HTTP OK ({resp.status}) from {url} - {body[:200]}"
##            print(f"[flash] Success: {msg}")
##            return True, msg
##    except urllib.error.HTTPError as e:
##        note = f"HTTPError {e.code} from {url}: {e.reason}"
##        print(f"[flash] HTTPError: {note}")
##        return False, note
##    except urllib.error.URLError as e:
##        note = f"URLError from {url}: {e.reason}"
##        print(f"[flash] URLError: {note}")
##        return False, note
##    except socket.timeout:
##        note = f"Timeout from {url}"
##        print(f"[flash] Timeout: {note}")
##        return False, note
##    except Exception as e:
##        note = f"Unknown error from {url}: {e}"
##        print(f"[flash] Unknown error: {note}")
##        return False, note
##
##



##def _try_http_flash(addr: str, on: bool, timeout: float = 3.0):
##    """
##    Send flash ON/OFF command via HTTP to ESP32-CAM (auto-strip /stream and :81, use port 80)
##    Returns (success: bool, message: str)
##    """
##    if not addr:
##        return False, "No IP provided for HTTP control"
##
##    # Normalize address
##    addr = addr.strip()
##    # Remove /stream suffix if present
##    if addr.endswith("/stream"):
##        addr = addr[:-7]
##    # Remove :81 port if present
##    if ":81" in addr:
##        addr = addr.replace(":81", "")
##
##    # Ensure scheme
##    if not addr.startswith("http://") and not addr.startswith("https://"):
##        addr = "http://" + addr
##
##    # Parse
##    parsed = urlparse(addr)
##    host = parsed.hostname # + ":81"
##    if not host:
##        return False, f"Could not parse host from '{addr}'"
##
##    path = "/flash/on" if on else "/flash/off"
##    url = f"http://{host}/flash{'/on' if on else '/off'}"
##
##    try:
##        print(f"[flash] HTTP request -> {url}")
##        with urllib.request.urlopen(url, timeout=timeout) as resp:
##            body = resp.read().decode(errors="ignore")
##            msg = f"HTTP OK ({resp.status}) from {url} - {body[:200]}"
##            print(f"[flash] Success: {msg}")
##            return True, msg
##    except urllib.error.HTTPError as e:
##        note = f"HTTPError {e.code} from {url}: {e.reason}"
##        print(f"[flash] HTTPError: {note}")
##        return False, note
##    except urllib.error.URLError as e:
##        note = f"URLError from {url}: {e.reason}"
##        print(f"[flash] URLError: {note}")
##        return False, note
##    except socket.timeout:
##        note = f"Timeout from {url}"
##        print(f"[flash] Timeout: {note}")
##        return False, note
##    except Exception as e:
##        note = f"Unknown error from {url}: {e}"
##        print(f"[flash] Unknown error: {note}")
##        return False, note



##def _try_http_flash(addr: str, on: bool, timeout: float = 3.0, retries: int = 1):
##    """
##    Improved HTTP attempt:
##    - Tries the original host:port/path first
##    - On 404 or failure it will try the same host on port 80 (no :81)
##    - Prints each attempted URL and returns detailed info
##    Returns (success:bool, message:str)
##    """
##    if not addr:
##        return False, "No IP provided for HTTP control"
##
##    path = "/flash/on" if on else "/flash/off"
##    scheme, host, port, err = _normalize_addr(addr)
##    if err:
##        return False, err
##
##    tried_results = []
##    candidates = []
##
##    # Candidate 1: use the original parsed port (if any) or assume none
##    if port:
##        candidates.append((scheme, host, port))
##    else:
##        # if no port specified, first try host without port (implies 80)
##        candidates.append((scheme, host, None))
##
##    # Always add explicit port 80 as an alternative (if different from original)
##    # (avoid duplicate)
##    if not (port == 80 or port is None):
##        candidates.append((scheme, host, 80))
##    elif port == 80:
##        # port already 80, but ensure a host-without-port candidate exists
##        candidates.append((scheme, host, None))
##
##    # Deduplicate while preserving order
##    seen = set()
##    cand_final = []
##    for s, h, p in candidates:
##        key = f"{s}://{h}:{p if p else 80}"
##        if key not in seen:
##            seen.add(key)
##            cand_final.append((s, h, p))
##
##    last_error = None
##    for attempt_num, (s, h, p) in enumerate(cand_final, start=1):
##        # build url
##        if p:
##            url = f"{s}://{h}:{p}{path}"
##        else:
##            url = f"{s}://{h}{path}"
##        try:
##            print(f"[flash] HTTP request -> {url} (candidate {attempt_num}/{len(cand_final)})")
##            with urllib.request.urlopen(url, timeout=timeout) as resp:
##                body = resp.read().decode(errors="ignore")
##                msg = f"HTTP OK ({resp.status}) from {url} - {body[:200]}"
##                print(f"[flash] Success: {msg}")
##                return True, msg
##        except urllib.error.HTTPError as e:
##            # if 404, the server answered but didn't find the endpoint on that port
##            note = f"HTTPError {e.code} from {url}: {e.reason}"
##            print(f"[flash] HTTPError: {note}")
##            tried_results.append(note)
##            last_error = note
##            # try next candidate (e.g., port 80)
##            continue
##        except urllib.error.URLError as e:
##            note = f"URLError from {url}: {e.reason}"
##            print(f"[flash] URLError: {note}")
##            tried_results.append(note)
##            last_error = note
##            # try next candidate (maybe different port)
##            continue
##        except socket.timeout:
##            note = f"Timeout from {url}"
##            print(f"[flash] Timeout: {note}")
##            tried_results.append(note)
##            last_error = note
##            continue
##        except Exception as e:
##            note = f"Unknown error from {url}: {e}"
##            print(f"[flash] Unknown error: {note}")
##            tried_results.append(note)
##            last_error = note
##            continue
##
##    # If we get here, none of the candidates succeeded
##    detail = "; ".join(tried_results) if tried_results else (last_error or "No attempts made")
##    return False, f"All HTTP attempts failed for {addr}. Details: {detail}"


def _try_arduino_flash(on: bool, arduino_obj=None):
    """
    Fallback: try to send a command to an arduino/serial bridge object.
    Pass arduino_obj explicitly or rely on global `arduino`.
    Returns (success:bool, message:str).
    """
    if arduino_obj is None:
        arduino_obj = globals().get("arduino")
    if arduino_obj is None:
        return False, "arduino object not available (pass arduino_obj or ensure imported)"
    cmd = "FLASH_ON" if on else "FLASH_OFF"
    tried = []
    candidates = ["write", "send", "send_command", "sendSerial", "serial_write", "command"]
    for name in candidates:
        if hasattr(arduino_obj, name):
            try:
                method = getattr(arduino_obj, name)
                if name == "write":
                    try:
                        method((cmd + "\n").encode())
                    except Exception:
                        method(cmd)
                else:
                    method(cmd)
                return True, f"Sent via arduino.{name}"
            except Exception as e:
                tried.append(f"{name} failed: {e}")
    # try serial.write if available
    if hasattr(arduino_obj, "serial"):
        try:
            s = getattr(arduino_obj, "serial")
            if hasattr(s, "write"):
                s.write((cmd + "\n").encode())
                return True, "Sent via arduino.serial.write"
        except Exception as e:
            tried.append(f"serial.write failed: {e}")
    return False, "No usable arduino send method found. Tried: " + ", ".join(tried)

# Generic worker used by the wrappers below
def _flash_worker(addr: str, on: bool, arduino_obj=None):
    # Normalize and attempt HTTP if addr provided
    if addr:
        ok, info = _try_http_flash(addr, on)
        if ok:
            print(f"[flash:{'on' if on else 'off'}] HTTP success:", info)
            return True, info
        print(f"[flash:{'on' if on else 'off'}] HTTP failed:", info)
    # Arduino fallback
    ok, info = _try_arduino_flash(on, arduino_obj=arduino_obj)
    if ok:
        print(f"[flash:{'on' if on else 'off'}] Arduino success:", info)
        return True, info
    print(f"[flash:{'on' if on else 'off'}] Arduino failed:", info)
    return False, info

# Public wrapper functions (left/right kept for compatibility)
def set_flash_on_left(ip: str = None, arduino_obj=None, use_thread: bool = False):
    """
    Turn left flash ON. ip may include http://, :port, /stream, etc.
    """
    if ip:
        # strip common "/stream" suffix if present (optional)
##        ip = ip.replace("/stream", "")
        print(f"Camera received IP left ON : {ip}")
    if use_thread:
        t = threading.Thread(target=_flash_worker, args=(ip, True, arduino_obj), daemon=True)
        t.start()
        return True, "thread started"
    else:
        return _flash_worker(ip, True, arduino_obj)

def set_flash_off_left(ip: str = None, arduino_obj=None, use_thread: bool = False):
    if ip:
##        ip = ip.replace("/stream", "")
        print(f"Camera received IP left OFF : {ip}")
    if use_thread:
        t = threading.Thread(target=_flash_worker, args=(ip, False, arduino_obj), daemon=True)
        t.start()
        return True, "thread started"
    else:
        return _flash_worker(ip, False, arduino_obj)

def set_flash_on_right(ip: str = None, arduino_obj=None, use_thread: bool = False):
    if ip:
##        ip = ip.replace("/stream", "")
        print(f"Camera received IP right ON : {ip}")
    if use_thread:
        t = threading.Thread(target=_flash_worker, args=(ip, True, arduino_obj), daemon=True)
        t.start()
        return True, "thread started"
    else:
        return _flash_worker(ip, True, arduino_obj)

def set_flash_off_right(ip: str = None, arduino_obj=None, use_thread: bool = False):
    if ip:
##        ip = ip.replace("/stream", "")
        print(f"Camera received IP right OFF : {ip}")
    if use_thread:
        t = threading.Thread(target=_flash_worker, args=(ip, False, arduino_obj), daemon=True)
        t.start()
        return True, "thread started"
    else:
        return _flash_worker(ip, False, arduino_obj)





##import urllib.request
##import urllib.error
##import socket
##import threading
##
##def _try_http_flash(ip: str, on: bool, timeout: float = 3.0):
##    """Try to toggle flash via the camera's /flash endpoints."""
##    if not ip:
##        return False, "No IP provided for HTTP control"
##    path = "/flash/on" if on else "/flash/off"
##    url = f"http://{ip}{path}"
##    try:
##        with urllib.request.urlopen(url, timeout=timeout) as resp:
##            body = resp.read().decode(errors="ignore")
##            return True, f"HTTP OK ({resp.status}) - {body[:200]}"
##    except urllib.error.HTTPError as e:
##        return False, f"HTTPError {e.code}: {e.reason}"
##    except urllib.error.URLError as e:
##        return False, f"URLError: {e.reason}"
##    except socket.timeout:
##        return False, "HTTP request timed out"
##    except Exception as e:
##        return False, f"HTTP unknown error: {e}"
##
##def _try_arduino_flash(on: bool):
##    """Try to toggle flash via the imported `arduino` object (serial/command)."""
##    if 'arduino' not in globals() or arduino is None:
##        return False, "arduino object not available"
##    cmd = "FLASH_ON" if on else "FLASH_OFF"
##    tried = []
##    # Common candidate method names that user code might expose
##    candidates = ["write", "send", "send_command", "sendSerial", "serial_write", "command"]
##    for name in candidates:
##        if hasattr(arduino, name):
##            try:
##                method = getattr(arduino, name)
##                # try bytes first for 'write'
##                if name == "write":
##                    try:
##                        method((cmd + "\n").encode())
##                        return True, f"Sent via arduino.{name} (bytes)"
##                    except Exception:
##                        method(cmd)  # fallback to string
##                        return True, f"Sent via arduino.{name} (string)"
##                else:
##                    method(cmd)
##                    return True, f"Sent via arduino.{name}"
##            except Exception as e:
##                tried.append(f"{name} failed: {e}")
##                continue
##    # try a generic attribute 'serial' with write
##    if hasattr(arduino, "serial"):
##        try:
##            s = getattr(arduino, "serial")
##            if hasattr(s, "write"):
##                s.write((cmd + "\n").encode())
##                return True, "Sent via arduino.serial.write"
##        except Exception as e:
##            tried.append(f"serial.write failed: {e}")
##    return False, "No usable arduino send method found. Tried: " + ", ".join(tried)
##
##def set_flash_on_left(ip: str = None, use_thread: bool = False):
##    """
##    Turn flash ON.
##    If ip is provided, HTTP will be attempted first.
##    If use_thread True, the operation runs in a background thread (non-blocking).
##    Returns (success, info) OR if use_thread True returns (True, 'thread started').
##    """
##    ip = ip.replace(":81/stream","")
##    print(f"Camera received IP left ON : {ip}")
##
##    
##    def _work():
##        # Try HTTP first (if ip provided)
##        if ip:
##            ok, info = _try_http_flash(ip, True)
##            if ok:
##                print("[flash:on] HTTP success:", info)
##                return
##            print("[flash:on] HTTP failed:", info)
##        # Fallback to arduino serial/command
##        ok, info = _try_arduino_flash(True)
##        if ok:
##            print("[flash:on] Arduino success:", info)
##        else:
##            print("[flash:on] Arduino failed:", info)
##
##    if use_thread:
##        t = threading.Thread(target=_work, daemon=True)
##        t.start()
##        return True, "thread started"
##    else:
##        _work()
##        # we cannot reliably return the inner result (printed instead), so return True for completion
##        return True, "operation attempted (check logs for details)"
##
##def set_flash_off_left(ip: str = None, use_thread: bool = False):
##    """
##    Turn flash OFF. Same parameters/behaviour as set_flash_on.
##    """
##
##    ip = ip.replace(":81/stream","")
##    print(f"Camera received IP left OFF : {ip}")
##
##    def _work():
##        if ip:
##            ok, info = _try_http_flash(ip, False)
##            if ok:
##                print("[flash:off] HTTP success:", info)
##                return
##            print("[flash:off] HTTP failed:", info)
##        ok, info = _try_arduino_flash(False)
##        if ok:
##            print("[flash:off] Arduino success:", info)
##        else:
##            print("[flash:off] Arduino failed:", info)
##
##    if use_thread:
##        t = threading.Thread(target=_work, daemon=True)
##        t.start()
##        return True, "thread started"
##    else:
##        _work()
##        return True, "operation attempted (check logs for details)"
##
##
##
##def set_flash_on_right(ip: str = None, use_thread: bool = False):
##    """
##    Turn flash ON.
##    If ip is provided, HTTP will be attempted first.
##    If use_thread True, the operation runs in a background thread (non-blocking).
##    Returns (success, info) OR if use_thread True returns (True, 'thread started').
##    """
##
##    ip = ip.replace(":81/stream","")
##    print(f"Camera received IP right ON : {ip}")
##    
##    def _work():
##        # Try HTTP first (if ip provided)
##        if ip:
##            ok, info = _try_http_flash(ip, True)
##            if ok:
##                print("[flash:on] HTTP success:", info)
##                return
##            print("[flash:on] HTTP failed:", info)
##        # Fallback to arduino serial/command
##        ok, info = _try_arduino_flash(True)
##        if ok:
##            print("[flash:on] Arduino success:", info)
##        else:
##            print("[flash:on] Arduino failed:", info)
##
##    if use_thread:
##        t = threading.Thread(target=_work, daemon=True)
##        t.start()
##        return True, "thread started"
##    else:
##        _work()
##        # we cannot reliably return the inner result (printed instead), so return True for completion
##        return True, "operation attempted (check logs for details)"
##
##    
##
##def set_flash_off_right(ip: str = None, use_thread: bool = False):
##    """
##    Turn flash OFF. Same parameters/behaviour as set_flash_on.
##    """
##    ip = ip.replace(":81/stream","")
##    print(f"Camera received IP right OFF : {ip}")
##    
##    def _work():
##        if ip:
##            ok, info = _try_http_flash(ip, False)
##            if ok:
##                print("[flash:off] HTTP success:", info)
##                return
##            print("[flash:off] HTTP failed:", info)
##        ok, info = _try_arduino_flash(False)
##        if ok:
##            print("[flash:off] Arduino success:", info)
##        else:
##            print("[flash:off] Arduino failed:", info)
##
##    if use_thread:
##        t = threading.Thread(target=_work, daemon=True)
##        t.start()
##        return True, "thread started"
##    else:
##        _work()
##        return True, "operation attempted (check logs for details)"
