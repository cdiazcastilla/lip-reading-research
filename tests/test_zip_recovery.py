import json
import os
import zipfile

import pytest

from lipreading.vsr.evaluate import load_manifest
from lipreading.vsr.zip_recovery import extract


def _package(path, n_videos=3):
    manifest = {"clips": [{"id": f"c{i}", "archivo": f"c{i}.webm", "modo": "silencio",
                           "frase": "Un café", "contexto": "¿Qué quieres?"} for i in range(n_videos)]}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr("manifest.json", json.dumps(manifest))
        for i in range(n_videos):
            z.writestr(f"c{i}.webm", os.urandom(2000))


def test_intact_zip(tmp_path):
    _package(tmp_path / "p.zip")
    assert extract(str(tmp_path / "p.zip"), str(tmp_path / "out")) is None
    clips = load_manifest(str(tmp_path / "out"))
    assert clips[0] == {"id": "c0", "file": "c0.webm", "mode": "silent", "sentence": "Un café", "context": "¿Qué quieres?"}


def test_truncated_zip_recovers_complete_entries(tmp_path):
    _package(tmp_path / "p.zip")
    data = (tmp_path / "p.zip").read_bytes()
    (tmp_path / "cut.zip").write_bytes(data[: len(data) // 2])  # upload interrupted
    recovered = extract(str(tmp_path / "cut.zip"), str(tmp_path / "out"))
    assert "manifest.json" in recovered and "c0.webm" in recovered
    assert "c2.webm" not in recovered


def test_unrecoverable_zip(tmp_path):
    (tmp_path / "bad.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 10)
    with pytest.raises(ValueError):
        extract(str(tmp_path / "bad.zip"), str(tmp_path / "out"))
