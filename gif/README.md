# Demo GIF

A presentation-ready looping animation that demonstrates **every mechanic** of the
`ner_rnapii_3d` engine on one compact locus (600 sites × 250 bp = **150 kb**, single
chain). Unlike the shipped 10k configs — which enable RNA Pol II and lesions
*separately* — the demo config turns on all mechanics at once.

| file | what |
|------|------|
| `demo.gif`         | the animation — 1280×720, ~20 s, infinite loop (drop straight into a slide) |
| `demo_poster.png`  | a single crisp frame (static poster / thumbnail) |
| `config_demo.yaml` | the compact all-features config |
| `run/LEFPositions.h5` | the trajectory the GIF was rendered from |
| `make_gif.py`      | the renderer (reads the config + H5, writes the GIF) |

## What the animation shows

Three genome-browser tracks over a shared genomic axis:

- **LOOP EXTRUSION** — cohesin loops load, extrude, get **captured** at oriented
  convergent **CTCF** barriers (bright, square-capped arcs), **bypass** a deliberately
  leaky boundary (tall cross-domain arcs), and **turn over**. TAD domains are shaded.
- **GENES / RNA Pol II** — genes coloured by class (**housekeeping / cell-type /
  developmental**); **enhancers** light up **teal** when a cohesin loop brackets the
  enhancer–promoter pair (E–P contact arc); single-molecule **Pol II** traffic coloured
  by state (pre-init / paused / elongating / terminating / **stalled**) with comet
  trails; live **active / poised / silent** badge per gene.
- **DNA DAMAGE & REPAIR** — UV lesions as stars: **Type A = TC-NER**, **Type B = GG-NER**;
  hollow = pre-recognition, filled + halo = repair machinery bound. A **stalled Pol II
  sitting over a Type-A lesion is flagged `TC-NER`** (transcription-coupled repair).

A live stat readout (biological time, cohesins CTCF-anchored, genes active, Pol II
elongating/stalled, lesions in repair, transcripts completed) runs along the bottom.

## Regenerate

```bash
# 1. simulate  (writes run/LEFPositions.h5)
uv run ner-sim run gif/config_demo.yaml --output gif/run/
#    (or: .venv/bin/python -m ner_rnapii_3d run gif/config_demo.yaml --output gif/run/)

# 2. render the GIF
.venv/bin/python gif/make_gif.py --out gif/demo.gif --frames 300 --fps 15

# fast single-frame preview while tweaking the design:
.venv/bin/python gif/make_gif.py --frame 150 --png gif/preview.png
```

The config uses only the validated default plugin set, so the run stays on the Numba
engine. Rates are **boosted vs. the calibrated configs** so every event type is visible
in a short trajectory — this asset is for demonstration, not quantitative analysis.
