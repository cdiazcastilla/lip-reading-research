# Lip reading for people who have lost their voice

[![tests](https://github.com/cdiazcastilla/lip-reading-research/actions/workflows/tests.yml/badge.svg)](https://github.com/cdiazcastilla/lip-reading-research/actions/workflows/tests.yml)

Research code behind the lip-reading features of [DeeprVoice](https://deeprvoice.com), an assistive-communication app for people who can no longer speak (laryngectomy, tracheostomy, ALS). They mouth a sentence silently; the app says it out loud in their own cloned voice.

This repository contains two lines of work, in Python:

| | What it does | Status |
|---|---|---|
| [**Personal lip reader**](#1-personal-lip-reader) | Recognises the user's *own* phrases from face landmarks, trained on a few clips they record themselves | In production (browser version); NumPy port here |
| [**Open-vocabulary evaluation**](#2-open-vocabulary-lip-reading-evaluation) | Measures whether a state-of-the-art model + an LLM can read *any* sentence | Pilot 01 done: not usable yet, but the visual signal is there |

The production app runs in the browser (JavaScript). The personal reader here is a faithful NumPy port, so the method can be read, tested and reproduced. User recordings are biometric health data and are not published; the tests and demo use synthetic landmarks.

## 1. Personal lip reader

`lipreading/personal/`

Generic lip-reading models struggle with silent speech and with people whose articulation has changed. Instead, each user builds a small dataset of their own phrases ("I need water", "Call my daughter"), a few recordings each, and the reader learns their mouth.

```
MediaPipe face mesh (478 landmarks) per frame
  -> lips, chin and cheeks, expressed in the face's own frame
     (origin between the cheeks, unit = face width, rotation removed)
     => invariant to head position, in-plane rotation and camera distance
  -> PCA fitted on this user's frames: their personal "shape space"
  -> silence trimming (mouth opening + motion energy), smoothing, velocities
  -> resampling to a fixed length
  -> k-nearest-neighbour Dynamic Time Warping against the user's templates
  -> accept / ambiguous / reject
```

**Calibrated, not hard-coded.** Acceptance thresholds come from leave-one-out over the user's own dataset: every clip is classified against the rest, and the operating point maximises *accepted hits − 3 × accepted errors*. Saying the wrong sentence out loud is worse than saying nothing. The same pass reports an estimated accuracy per phrase, so the app can tell the user which phrase needs more recordings.

**One problem found along the way.** An early version clustered frames with K-Means into viseme classes. A variance analysis showed jaw displacement dominated the feature space and collapsed the clusters; per-dimension standardisation and per-user calibration fixed it, and the design later moved to per-user PCA + DTW.

```bash
pip install -r requirements.txt
python examples/synthetic_demo.py
```

```
Trained on 20 clips, 4 phrases
Leave-one-out accuracy: 100%
Calibrated threshold 0.184, margin 0.80
New clips: 40 | matched 40 | ambiguous 0 | rejected 0 | correct matches 40/40
Unrecorded phrase "Open the window": wrongly spoken 0/10 times
```

Synthetic clips are clean by design (random head pose, distance, speaking rate and landmark noise, but no real faces), so these numbers only show the pipeline works end to end; they are not an accuracy claim.

## 2. Open-vocabulary lip reading evaluation

`lipreading/vsr/` · results in [`experiments/pilot-01`](experiments/pilot-01)

A fixed phrase list is limiting. Can a model read *new* sentences? The evaluation pipeline:

1. Recovers the recording package, even from a truncated upload (walks the zip's local headers and keeps every entry whose CRC-32 checks out).
2. Runs the Spanish visual speech recognition model of [Ma, Petridis & Pantic (2022)](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) on GPU, with an OpenCV video reader patched in for recent torchvision builds.
3. Corrects the noisy reading with an LLM, with and without the conversational context (what the person had just been told).
4. Scores every stage with word error rate (WER), character error rate and a Spanish **viseme** similarity metric, and writes a Markdown report plus CSV/JSON.

**Pilot 01 (30 silent sentences):** the model alone reaches 86 % WER, the LLM with context 74 %, and only 13 % of sentences are understandable in the best of 3 options, far from the 60 % target. But at the level of mouth shapes the model's reading matches the true sentence **72 %**, and in 23/30 sentences the raw reading fits the lips better than the LLM's "correction". The vision works; the language step loses it. [Full write-up](experiments/pilot-01/README.md).

![Viseme similarity vs word accuracy](experiments/pilot-01/viseme_vs_wer.png)

### Running it

On a GPU machine (tested on RunPod with the PyTorch template):

```bash
bash setup_vsr.sh                                   # model code + Spanish weights
export LLM_API_KEY=...                              # any OpenAI-compatible API; Groq by default
python -m lipreading.vsr.evaluate --zip package.zip --vsr ./vsr
python -m lipreading.vsr.evaluate --zip package.zip --readings results/readings.json --rerank 10   # LLM stage only, no GPU
```

> **Licence note:** the Ma et al. model weights are for non-commercial research only. They are used here to measure feasibility, and are not part of the product.

## Repository layout

```
lipreading/
  metrics.py            WER, CER, normalisation, summaries
  visemes.py            Spanish viseme classes, similarity, re-ranking
  personal/
    reader.py           personal lip reader (descriptors, PCA, DTW, LOO calibration)
    landmarks.py        MediaPipe FaceLandmarker indices
    synthetic.py        synthetic landmark clips for tests and demos
  vsr/
    evaluate.py         open-vocabulary evaluation pipeline (CLI)
    llm.py              LLM correction (OpenAI-compatible API)
    zip_recovery.py     recovery of truncated recording packages
    place_weights.py    model weight installation
experiments/pilot-01/   data, analysis script, figure and write-up
examples/               synthetic demo
tests/                  pytest suite (runs in CI)
```

## Author

Carlos Diaz · Electronic & Telecommunications Engineer, AI engineer · [LinkedIn](https://linkedin.com/in/cdiazcastilla) · [deeprvoice.com](https://deeprvoice.com)

MIT licence for the code in this repository.
