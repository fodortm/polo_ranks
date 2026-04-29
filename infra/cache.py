import hashlib
import os


def build_file_fingerprint(files):
    payload = []
    for path in files:
        stat = os.stat(path)
        payload.append(f"{os.path.basename(path)}:{int(stat.st_mtime_ns)}:{stat.st_size}")
    encoded = "|".join(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
