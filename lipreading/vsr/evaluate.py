#!/usr/bin/env python3
"""Feasibility test: open-vocabulary lip reading in Spanish.

Takes a recording package (videos + manifest.json with the true sentence and
the conversational context of each clip) and measures four stages:

  1. Visual model alone    -> WER of the raw reading
  2. + LLM, no context     -> WER of the LLM's first option
  3. + LLM with context    -> WER using what the person had just been told
  4. Best of 3             -> is the right sentence among the 3 options shown
                              (in the app the user taps the right one)

Optional --rerank N: the LLM proposes N candidates and they are re-ordered by
viseme similarity to the visual reading (see lipreading/visemes.py).

Usage (on a GPU machine with the visual model installed, see README):
  python -m lipreading.vsr.evaluate --zip package.zip --vsr ./vsr
  python -m lipreading.vsr.evaluate --zip package.zip --vsr ./vsr --no-llm
  python -m lipreading.vsr.evaluate --zip package.zip --readings readings.json   # reuse readings, no GPU

Writes report.md, results.json, results.csv and readings.json to --out.
Videos are deleted when the run ends.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Dict, List

from . import llm
from .zip_recovery import extract
from ..metrics import summarize, wer
from ..visemes import rerank_by_visemes

# Ma, Petridis & Pantic (2022), CMU-MOSEAS Spanish visual-only model.
VSR_CONFIG = "configs/CMUMOSEAS_V_ES_WER44.5.ini"


def load_manifest(folder: str) -> List[Dict]:
    """Clips as dicts with id, file, mode, sentence, context.
    The DeeprVoice recorder writes Spanish keys; both spellings are accepted."""
    raw = json.load(open(os.path.join(folder, "manifest.json"), encoding="utf-8"))["clips"]
    pick = lambda c, en, es: c.get(en, c.get(es))  # noqa: E731
    return [{
        "id": c["id"],
        "file": pick(c, "file", "archivo"),
        "mode": {"silencio": "silent", "voz": "aloud"}.get(pick(c, "mode", "modo"), pick(c, "mode", "modo")),
        "sentence": pick(c, "sentence", "frase"),
        "context": pick(c, "context", "contexto"),
    } for c in raw]


def to_mp4(src: str, dst: str, fps: int = 30):
    """Browser webm/mp4 (variable frame rate) -> constant 30 fps mp4, no audio."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-r", str(fps), "-an",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", dst], check=True)


def patch_video_reader():
    """The model reads videos with torchvision.io.read_video, which recent
    torchvision builds (the ones shipped for new GPUs) removed or broke.
    Replace it with an OpenCV reader returning the same thing:
    (frames [T,H,W,3] uint8 RGB, empty audio, info)."""
    import cv2
    import numpy as np
    import torch
    import torchvision

    def read_video(filename, pts_unit="sec", **kwargs):
        cap = cv2.VideoCapture(str(filename))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not read any frame from {filename}")
        return torch.from_numpy(np.stack(frames)), torch.empty((1, 0)), {"video_fps": fps}

    torchvision.io.read_video = read_video


class VisualReader:
    """Wraps the inference pipeline of mpc001/Visual_Speech_Recognition_for_Multiple_Languages."""

    def __init__(self, vsr_dir: str, gpu: bool = True):
        import torch
        patch_video_reader()
        self.vsr_dir = os.path.abspath(vsr_dir)
        sys.path.insert(0, self.vsr_dir)
        cwd = os.getcwd()
        os.chdir(self.vsr_dir)  # paths in the .ini are relative to the repo
        try:
            from pipelines.pipeline import InferencePipeline
            device = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
            print(f"Loading model on {device}...")
            self.pipe = InferencePipeline(VSR_CONFIG, device=device, detector="mediapipe", face_track=True)
        finally:
            os.chdir(cwd)

    def read(self, video_mp4: str) -> str:
        cwd = os.getcwd()
        os.chdir(self.vsr_dir)
        try:
            return (self.pipe(os.path.abspath(video_mp4)) or "").strip()
        finally:
            os.chdir(cwd)


STAGES = [("Visual model alone", "wer_raw"), ("+ LLM, no context", "wer_llm"),
          ("+ LLM with context", "wer_llm_ctx"), ("Best of 3 (with context)", "wer_best3")]


def pct(x):
    return "—" if x is None else f"{x * 100:.0f} %"


def write_report(rows: List[Dict], out: str, meta: Dict):
    by_mode: Dict[str, List[Dict]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    lines = [
        "# Lip-reading test — open vocabulary (Spanish)", "",
        f"- Date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- Clips evaluated: {len(rows)}",
        "- Visual model: Ma et al. 2022, CMU-MOSEAS Spanish (published WER 44.5 %) — **non-commercial licence, evaluation only**",
        f"- LLM: {meta.get('llm') or 'disabled'}" + (f", re-ranked by visemes from {meta['rerank']} candidates" if meta.get("rerank") else ""),
        "",
        "WER = share of words read wrong (lower is better). *Understandable* = sentences with at most 25 % WER. "
        "*Best of 3* = the right sentence was among the 3 options the user would see.", "",
    ]
    for mode, rs in by_mode.items():
        lines += [f"## Mode: {mode} ({len(rs)} clips)", "", "| Stage | WER | Exact | Understandable |", "|---|---|---|---|"]
        for name, k in STAGES:
            vals = [r[k] for r in rs if r.get(k) is not None]
            if vals:
                s = summarize(vals)
                lines.append(f"| {name} | {pct(s['wer'])} | {pct(s['exact'])} | {pct(s['understandable'])} |")
        lines += ["", "<details><summary>Per sentence</summary>", "",
                  "| # | True sentence | Model reading | LLM with context (1st option) | WER raw → ctx |", "|---|---|---|---|---|"]
        for r in rs:
            first = (r.get("options_ctx") or [""])[0]
            lines.append(f"| {r['id']} | {r['sentence']} | {r['reading']} | {first} | {pct(r['wer_raw'])} → {pct(r.get('wer_llm_ctx'))} |")
        lines += ["", "</details>", ""]
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def llm_stage(reading: str, sentence: str, context: str, rerank: int = 0) -> Dict:
    """Runs the LLM with and without context and scores the options."""
    if rerank:
        opts = rerank_by_visemes(reading, llm.options(reading, n=rerank))[:3]
        opts_ctx = rerank_by_visemes(reading, llm.options(reading, context, n=rerank))[:3]
    else:
        opts, opts_ctx = llm.options(reading), llm.options(reading, context)
    return {
        "options": opts, "options_ctx": opts_ctx,
        "wer_llm": wer(sentence, opts[0]),
        "wer_llm_ctx": wer(sentence, opts_ctx[0]),
        "wer_best3": min(wer(sentence, o) for o in opts_ctx),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="recording package (.zip)")
    ap.add_argument("--vsr", default="./vsr", help="folder of the visual model repository")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-llm", action="store_true", help="visual model only")
    ap.add_argument("--rerank", type=int, default=0, metavar="N", help="ask the LLM for N candidates and re-rank them by visemes")
    ap.add_argument("--readings", help="JSON {clip id: reading} from a previous run (skips the visual model)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="lipread_")
    try:
        recovered = extract(args.zip, tmp)
        clips = load_manifest(tmp)
        if recovered is not None:
            missing = [c["id"] for c in clips if c["file"] not in recovered]
            clips = [c for c in clips if c["file"] in recovered]
            if missing:
                print(f"   {len(missing)} clips did not arrive complete: {', '.join(missing)}")
        print(f"{len(clips)} clips in the package")

        readings = json.load(open(args.readings, encoding="utf-8")) if args.readings else {}
        reader = VisualReader(args.vsr, gpu=not args.cpu) if len(readings) < len(clips) else None

        rows, shown_trace = [], False
        for i, c in enumerate(clips, 1):
            t0 = time.time()
            if c["id"] not in readings:
                mp4 = os.path.join(tmp, c["id"] + ".mp4")
                to_mp4(os.path.join(tmp, c["file"]), mp4)
                try:
                    readings[c["id"]] = reader.read(mp4)
                except Exception as e:
                    print(f"  [model] failed on {c['id']}: {type(e).__name__}: {e}")
                    if not shown_trace:
                        traceback.print_exc()
                        shown_trace = True
                    readings[c["id"]] = ""
            reading = readings[c["id"]]
            row = {**c, "reading": reading, "wer_raw": wer(c["sentence"], reading)}
            if not args.no_llm and reading:
                row.update(llm_stage(reading, c["sentence"], c["context"], args.rerank))
            rows.append(row)
            print(f"[{i}/{len(clips)}] {c['id']} ({time.time() - t0:.1f}s)\n   true:  {c['sentence']}\n"
                  f"   model: {reading}  (WER {row['wer_raw']:.0%})"
                  + (f"\n   LLM:   {row['options_ctx'][0]}  (WER {row['wer_llm_ctx']:.0%})" if "options_ctx" in row else ""))

        # Keep the readings so the LLM stage can be re-run without a GPU
        json.dump(readings, open(os.path.join(args.out, "readings.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        json.dump(rows, open(os.path.join(args.out, "results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        with open(os.path.join(args.out, "results.csv"), "w", newline="", encoding="utf-8") as fh:
            fields = ["id", "mode", "context", "sentence", "reading", "wer_raw", "wer_llm", "wer_llm_ctx", "wer_best3"]
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        write_report(rows, args.out, {"llm": None if args.no_llm else llm.MODEL, "rerank": args.rerank})
        print(f"\nDone: {args.out}/report.md")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # videos are never kept


if __name__ == "__main__":
    main()
