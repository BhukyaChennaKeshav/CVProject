# Geometry-Aware Document Forensics & Tampering Detection (refined proposal)

**Run it:** upload `DocForensics_Colab.ipynb` to Google Colab → *Runtime › Run all*. It needs no GPU and no manual dataset download.
The core run takes about 5–10 min on a Colab CPU. Every pipeline step is plotted inline.
`doc_forensics.py` is the same code as a plain script. `make_notebook.py` rebuilds the notebook from it.

## 1. Problem & scope
Fraud on structured documents (contracts, receipts, invoices) usually means editing a *few words or numbers*:
changing an amount, inserting a clause word, nudging a figure, or whiting out a line. Each edit leaves two kinds of evidence:

| Evidence | Why it appears | How we measure it |
|---|---|---|
| **Geometric** | Pasted or shifted text rarely sits exactly on the original baseline or at the original angle, height, and spacing | Projection profiles, robust baseline fits, Hough parallelism, gap statistics |
| **Photometric** | A pasted patch has a different noise, blur, ink, and compression history from the scanned page | Noise residual maps, ELA, edge sharpness, ink-core noise, stroke width |

The system is fully classical (OpenCV, NumPy, SciPy, scikit-image) and every decision is explainable.

## 2. What was refined vs. the original proposal
1. **Scale-adaptive morphology ("adaptive structuring-element matrix")** — the main refinement.
   Structured documents mix 10 px fine print, 20 px body text, and 45 px titles on one page, so any fixed kernel fails.
   * A **local character-height map** is built from connected-component heights (grid median → nearest-neighbour fill → smoothing), then quantised into ⅓-octave **scale bins**.
   * For each bin, the **word-gap ratio is learned from the page itself**. Glyph gaps on a line form two groups (gaps inside words and gaps between words). Otsu gives a first split, which is then moved to the density valley between the two peaks.
   * Each bin gets its own structuring element `rect(ratio·h, 0.1·h)`. The closing is **spatially variant**: each pixel takes the result from its own bin.
   * The same idea sizes the Sauvola window (≈2.2·h), the background-estimation kernel (3·h), and the rule-line opening lengths.
   * Ablation (word-segmentation F1 against renderer ground truth):

     | doc | fixed 15×3 | global-scale | **local-adaptive** |
     |---|---|---|---|
     | contract | 0.15 | 0.87 | **0.94** |
     | invoice | 0.56 | 0.75 | **0.85** |
     | receipt | 0.67 | 0.49 | **0.79** |
2. **Structured-document awareness:**
   * Table and rule lines are extracted first with length-adaptive opening plus `HoughLinesP`, then removed so they don't glue words together.
   * Row peers are found **by proximity across columns**, so a receipt amount or a table cell is checked against its own row.
   * Dashes and underscores are classed as graphics and excluded from scoring.
3. **Glyph-level baselines:** the baseline is the densest cluster of glyph-component bottoms, so descenders and "T" crossbars don't throw it off. The angle comes from a Theil–Sen fit with a per-word uncertainty (1.5 px over the fitted span).
4. **Forensics computed on the raw pixel grid** (noise residual, ELA, gradients) *before* deskew interpolation, then rotated into the analysis frame.
5. **One-sided, sliding-window noise map:** forged patches and whiteouts are *too clean*. Saturated white paper is excluded because it is clean for innocent reasons.
6. **Principled fusion and thresholds:** chi-style evidence fusion `S = √(Σ wᵢ zᵢ²) − √(Σ wᵢ)` over robust neighbourhood z-scores. Thresholds are **calibrated on genuine pages** to a target false-alarm rate (CFAR) rather than hand-tuned.
7. **Measurable evaluation:** a tamper generator with pixel-exact masks, plus the real forged-receipt dataset.

## 3. Datasets
| Dataset | Role | Access |
|---|---|---|
| Built-in synthetic contracts / receipts / invoices (mixed fonts and sizes, tables, signature rules) + print-and-scan simulation + **4 attacks** (splice-insert, copy-move, baseline shift, whiteout delete) with pixel masks | Development, ablation, quantitative evaluation | Generated in the notebook |
| **FUNSD** (scanned forms) / **CORD-v2** (receipts) | Real clean scans used as bases for the tamper generator | HuggingFace `nielsr/funsd`, `naver-clova-ix/cord-v2` (auto in Colab) |
| **Find-it-again** (L3i, ICDAR 2023): 988 SROIE receipts, 163 forged by people | Real-world evaluation | Set `USE_FINDIT=True` ([project page](https://l3i-share.univ-lr.fr/2023Finditagain/index.html)) |
| DocTamper (CVPR 2023, 170k images) | Optional large-scale follow-up | On request from the authors |

## 4. Pipeline (each step is plotted in the notebook)
**A. Low-level**
* **A1** Grayscale and character-height histogram
* **A2** Background estimation (closing) → illumination normalisation
* **A3** Local scale map and bins
* **A4** Scale-adaptive Sauvola binarisation
* **A5** Deskew by projection-profile sharpness
* **A6** Sobel and auto-Canny
* **A7** Noise residual, sliding noise σ, noise z-map, ELA

**B. Mid-level**
* **B1** Rule and table-grid extraction with HoughLinesP
* **B2** Learned gap ratios → adaptive kernel table
* **B3** Fixed vs. adaptive word segmentation
* **B4** Horizontal and vertical projection profiles → line bands and segments
* **B5** Per-word baselines
* **B6** Hough on baseline points → parallelism histogram

**C. Reasoning**
* **C1** Per-word feature table: baseline offset, angle, ascender excess, gap, stroke width, ink level, ink-core noise, background noise, ELA, sharpness
* **C2** Robust neighbourhood z-scores and a per-feature evidence bar chart
* **C3** Copy-move check: difference-variance of look-alike word pairs, normalised against the page
* **C4** Fused heat-map, flagged regions labelled with the dominant reason, final verdict

## 5. Current results (synthetic benchmark, local run, 12 forged/genuine pairs — small sample, expect variance)
* Image-level ROC-AUC **0.75**. Pixel-level heat-map AUC **0.87–0.97**. About **0.5 false-flagged words per genuine page**.
* Detection rate by attack:

  | Attack | Rate | Main evidence |
  |---|---|---|
  | Baseline shift | **1.00** | Geometry |
  | Whiteout delete | **0.75** | Noise map |
  | Splice-insert | 0.33 | Geometry + ink noise |
  | Copy-move | 0.20 | — |

* **Honest limitations:**
  * Copy-move with sub-pixel alignment plus JPEG re-saving is close to invisible to pixel statistics, and identical glyphs repeat legitimately. Geometry catches it only when it is misplaced.
  * Born-digital PDFs have no sensor noise, so the photometric cues weaken and the geometric ones remain.
  * Projection bands assume roughly horizontal text (after deskew), not curved photos.

## 6. Possible extensions
* Run the full Find-it-again evaluation and report AUC per forgery type.
* Add JPEG-grid (8×8 blockiness) misalignment maps for double-compression evidence.
* Replace the hand-weighted fusion with a logistic regression trained on the feature table. It stays interpretable.
