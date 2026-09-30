"""Extract a recording package, recovering what it can from a truncated upload.

The browser recorder writes an uncompressed ("stored") zip with manifest.json
first. If the upload was cut off, the central directory at the end is missing
and zipfile refuses the archive. Walking the local file headers one by one
still recovers every entry that arrived complete (checked with its CRC-32).
"""
import os
import struct
import zipfile
import zlib
from typing import Optional, Set

LOCAL_HEADER = b"PK\x03\x04"


def extract(path: str, dest: str) -> Optional[Set[str]]:
    """Extract into dest. Returns None if the zip was intact, else the names recovered."""
    try:
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
        return None
    except zipfile.BadZipFile:
        pass

    data = open(path, "rb").read()
    pos, recovered = 0, set()
    while pos + 30 <= len(data) and data[pos:pos + 4] == LOCAL_HEADER:
        (_, _, _flags, method, _, _, crc, size_c, _size, n_name, n_extra) = struct.unpack("<IHHHHHIIIHH", data[pos:pos + 30])
        name = data[pos + 30:pos + 30 + n_name].decode("utf-8", "replace")
        start = pos + 30 + n_name + n_extra
        end = start + size_c
        if method != 0 or end > len(data) or zlib.crc32(data[start:end]) != crc:
            break  # this entry was cut off: nothing after it is complete
        root = os.path.abspath(dest)
        out = os.path.abspath(os.path.join(root, name))
        if not out.startswith(root + os.sep):
            break  # refuse path traversal ("../") in entry names
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(data[start:end])
        recovered.add(name)
        pos = end
    if "manifest.json" not in recovered:
        raise ValueError("The zip is damaged and manifest.json could not be recovered. Upload it again.")
    print(f"Warning: truncated zip ({len(data) / 1e6:.1f} MB). Recovered {len(recovered) - 1} complete videos.")
    return recovered
