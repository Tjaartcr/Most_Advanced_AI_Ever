# probe_esp32_cam.py
import urllib.request
import urllib.error
from urllib.parse import urljoin
import sys
import socket

def probe(host):
    ports = [80, 81, 8080]
    paths = ['', '/', '/stream', '/capture', '/jpg', '/camera', '/cam', '/mjpeg']
    timeout = 4.0

    results = []
    for port in ports:
        for p in paths:
            url = f"http://{host}:{port}{p}"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'probe/1.0'})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    status = r.status
                    ctype = r.headers.get('Content-Type', '')
                    body = r.read(512)  # read up to 512 bytes
                    snippet = body[:200]
                    results.append((url, 'OK', status, ctype, snippet))
                    print(f"[OK] {url} -> {status} {ctype} | {len(snippet)} bytes")
            except urllib.error.HTTPError as he:
                print(f"[HTTPError] {url} -> {he.code} {he.reason}")
                results.append((url, 'HTTPError', he.code, str(he.reason), b''))
            except urllib.error.URLError as ue:
                reason = ue.reason
                print(f"[URLError] {url} -> {reason}")
                results.append((url, 'URLError', str(reason), '', b''))
            except socket.timeout:
                print(f"[Timeout] {url}")
                results.append((url, 'Timeout', None, '', b''))
            except Exception as e:
                print(f"[Error] {url} -> {e}")
                results.append((url, 'Error', str(e), '', b''))
    return results

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python probe_esp32_cam.py <ip-or-host>")
        sys.exit(1)
    host = sys.argv[1]
    probe(host)
