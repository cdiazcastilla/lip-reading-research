"""Unzip the downloaded weights and put them where the model's .ini expects them:

  vsr/benchmarks/CMUMOSEAS/models/es/CMUMOSEAS_V_ES_WER44.5/model.pth (+ model.json)
  vsr/benchmarks/CMUMOSEAS/language_models/es/lm_es/model.pth (+ model.json)

Does not assume the zips' internal layout: it looks for model.pth + model.json.
Usage: python -m lipreading.vsr.place_weights vsr /tmp/vsr_es.zip /tmp/lm_es.zip
"""
import os
import shutil
import sys
import tempfile
import zipfile


def find_model(folder):
    for root, _, files in os.walk(folder):
        if "model.pth" in files and "model.json" in files:
            return root
    raise SystemExit(f"No model.pth + model.json found inside {folder}")


def place(zip_path, dest):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    src = find_model(tmp)
    os.makedirs(dest, exist_ok=True)
    for f in ("model.pth", "model.json"):
        shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  ok -> {dest}")


if __name__ == "__main__":
    vsr, model_zip, lm_zip = sys.argv[1:4]
    place(model_zip, os.path.join(vsr, "benchmarks/CMUMOSEAS/models/es/CMUMOSEAS_V_ES_WER44.5"))
    place(lm_zip, os.path.join(vsr, "benchmarks/CMUMOSEAS/language_models/es/lm_es"))
