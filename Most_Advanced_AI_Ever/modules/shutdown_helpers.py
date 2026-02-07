# shutdown_helpers.py
import threading
import subprocess
import time
import os
import signal
import sys
from typing import Optional, Iterable

# Event that background loops should check regularly and exit when set.
app_shutdown_event = threading.Event()

# Track subprocess.Popen objects and started threads (only those you explicitly register).
_registered_subprocesses = set()  # holds subprocess.Popen instances
_registered_threads = {}          # name -> threading.Thread

_lock = threading.Lock()


# ---------- subprocess helpers ----------
def register_subprocess(proc: subprocess.Popen):
    """Register a subprocess.Popen object so the shutdown helper can close it later."""
    with _lock:
        _registered_subprocesses.add(proc)


def unregister_subprocess(proc: subprocess.Popen):
    with _lock:
        _registered_subprocesses.discard(proc)


def list_registered_subprocesses() -> Iterable[subprocess.Popen]:
    with _lock:
        return tuple(_registered_subprocesses)


def start_tracked_subprocess(args, **popen_kwargs) -> subprocess.Popen:
    """
    Convenience wrapper to start a subprocess and register it.
    Example: proc = start_tracked_subprocess(["ollama","serve"], cwd=work_dir)
    """
    # Make sure stdout/stderr are not blocking if caller expects them
    proc = subprocess.Popen(args, **popen_kwargs)
    register_subprocess(proc)
    return proc


# ---------- thread helpers ----------
def register_thread(t: threading.Thread, name: Optional[str] = None):
    """Register a thread object (so we can join it on shutdown)."""
    with _lock:
        _registered_threads[name or t.name] = t


def unregister_thread(t: threading.Thread):
    with _lock:
        for k, v in list(_registered_threads.items()):
            if v is t:
                _registered_threads.pop(k, None)


def list_registered_threads():
    with _lock:
        return dict(_registered_threads)


# ---------- shutdown actions ----------
def _terminate_proc_graceful(proc: subprocess.Popen, timeout: float = 3.0) -> bool:
    """Try terminate(), wait, then kill() if necessary. Returns True if process ended."""
    try:
        if proc.poll() is not None:
            return True
        # Try polite termination
        try:
            proc.terminate()
        except Exception:
            pass
        # wait a bit
        try:
            proc.wait(timeout=timeout)
            return True
        except Exception:
            pass
        # escalate
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=1.0)
            return True
        except Exception:
            return (proc.poll() is not None)
    except Exception:
        return False


def initiate_shutdown(grace_period: float = 2.0, wait_thread_timeout: float = 3.0):
    """
    Initiate app shutdown:
     - set the app_shutdown_event for threads to see,
     - politely terminate tracked subprocesses (terminate() then kill()),
     - join tracked threads (best-effort).
    """
    # set the event for loops to notice
    app_shutdown_event.set()

    # 1) Terminate subprocesses
    procs = list_registered_subprocesses()
    if procs:
        # Attempt graceful termination
        for p in procs:
            try:
                _terminate_proc_graceful(p, timeout=grace_period)
            except Exception:
                pass

    # 2) Join threads (best-effort)
    threads = list_registered_threads()
    if threads:
        now = time.time()
        for name, t in threads.items():
            if not t.is_alive():
                continue
            try:
                # don't join current thread or join forever
                if t is threading.current_thread():
                    continue
                t.join(timeout=wait_thread_timeout)
            except Exception:
                pass


def close_all_console_windows_graceful_or_force() -> bool:
    """
    Close tracked subprocess windows (safer than indiscriminately killing all CMD windows).
    Returns True if it attempted to close any process.
    """
    procs = list_registered_subprocesses()
    if not procs:
        return False

    any_attempted = False
    for p in procs:
        try:
            pid = getattr(p, "pid", None)
            if pid is None:
                # try to terminate using Popen handle
                _terminate_proc_graceful(p, timeout=1.0)
                any_attempted = True
            else:
                any_attempted = True
                if os.name == "nt":
                    # taskkill /PID <pid> /T /F will terminate tree - but we try polite terminate first
                    try:
                        _terminate_proc_graceful(p, timeout=1.0)
                    except Exception:
                        pass
                    try:
                        # best-effort final kill by PID using taskkill (Windows)
                        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    except Exception:
                        pass
                else:
                    # POSIX: send SIGTERM then SIGKILL
                    try:
                        _terminate_proc_graceful(p, timeout=1.0)
                    except Exception:
                        pass
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except Exception:
                        pass
        except Exception:
            pass

    return any_attempted


# ---------- tidy helpers ----------
def clear_all_registered():
    """Clear registrations (use when shutting down completely)."""
    with _lock:
        _registered_subprocesses.clear()
        _registered_threads.clear()
