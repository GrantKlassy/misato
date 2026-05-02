# misato

> Indeed, she is *that bitch*.

A two-stage Rust pipeline that scans the four canonical Neon Genesis
Evangelion artbooks and surfaces the highest-quality images of **Misato
Katsuragi**, **Rei Ayanami**, and **Asuka Langley Soryu** by hair-class
color clustering.

Built end-to-end per `HUMAN.md`. No ML — just careful color math, connected
components, and inter-character competition.

## Pipeline

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│  812 page JPEGs │ ─▶ │ rank (full   │ ─▶ │ extract (top-N) │
│  4 artbooks     │    │ corpus, ↓1500)│    │ full-res crops  │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                    │
                              ▼                    ▼
                       rankings.{csv,json}   top_pages/, extracted/,
                                             index.html (contact sheet)
```

### Stage 1 — `rank`

For every page, downscaled to ≤ 1500 px on the longest side:

1. **Hair-class clustering** per character: a pixel is classed as matching
   the character's signature hair color iff (HSV hue is in band)
   ∧ (saturation/value in range) ∧ (CIELAB ΔE76 ≤ tolerance). HSV catches
   the right *family* of color; LAB ΔE keeps the match perceptually close.
2. **Connected components** on the binary mask, filtered by minimum area.
3. **Secondary-color coupling**: jacket / plugsuit / eyes pixels within 2×
   diagonal of each hair blob's centroid are counted — strong signal that
   the hair cluster belongs to a real character, not a background wash.
4. **Shape gate**: largest blob's density (area / bbox\_area) and aspect
   ratio modulate the score so coherent hair clusters beat sparse
   wash-outs.
5. **Inter-character competition**: every character is scored on the same
   page; if another character has a much stronger combined signature
   (primary + 0.7 × secondary mass), the runner-up is multiplicatively
   demoted. This keeps Asuka's plugsuit-red pages from cross-firing as
   Misato hits when violet shadows are present.

Score ~ `2.0 × log(1 + 1000·blob_ratio) × shape + log(mass) × 0.6 +
secondary × 0.3 + coupling × 0.6` then × competition factor (∈ [0.25, 1]).

### Stage 2 — `extract`

For the top-N ranked pages per character:

1. Re-detect hair blobs **at full resolution** (4–16 MP per page).
2. Hierarchically merge blobs into character groups (union-find on
   3×-expanded bbox overlap).
3. Crop a padded bbox around each group (3.5× expansion, with a floor of
   30% of the page's shorter side) and save the JPEG.
4. Save a full-res copy of the source page in `top_pages/` for posterity
   ("highest quality images of Misato for the world").

### Stage 3 — contact sheet

After extraction, a self-contained `index.html` is written per character:
a dark-themed grid of all top-N pages with rank, certainty, source book,
and image stats. Click a thumbnail to open the full-res page.

## Build & run

```bash
cargo build --release

# rank all pages, write rankings.{csv,json} per character
./target/release/misato rank

# extract top-40 ranked pages per character (loads rankings.json)
./target/release/misato extract --top-n 40

# both back-to-back
./target/release/misato all --top-n 40

# only one character
./target/release/misato --character misato all --top-n 60

# point at different art roots
./target/release/misato --art-root /path/to/book1 --art-root /path/to/book2 rank
```

## Performance

- **Rank**: 812 pages × 3 characters in **~28s wall-clock** on 16 cores
  (rayon-parallel, downscaled to 1500 px max dim).
- **Extract**: top-40 per character (120 pages full-res) in **~20s**.

## Output layout

```
output/
├── misato/
│   ├── rankings.csv           # all 812 pages, ranked
│   ├── rankings.json          # same, machine-readable
│   ├── index.html             # contact sheet — open in a browser
│   ├── top_pages/             # full-res copies of top-N source pages
│   │   └── rank001_score15.283_..._p020_i049.jpg
│   └── extracted/             # padded bbox crops around hair clusters
│       └── rank001_score15.283_..._crop00.jpg
├── rei/   (same structure)
└── asuka/ (same structure)
```

## Character configs

Hand-tuned per `HUMAN.md`. See `src/config.rs`. Hair is the primary
signature; jacket / plugsuit / eyes are secondaries.

| Character | Primary (hair)        | Secondaries                                    |
| --------- | --------------------- | ---------------------------------------------- |
| Misato    | violet `#5B4B7A`      | jacket red `#B91C2C`                           |
| Rei       | pale blue `#B8D4E3`   | eyes `#D72638`, A10 orange `#FF6B1A`           |
| Asuka     | auburn `#C44827`      | plugsuit red `#C8102E`                         |

## Detection quality (observed)

- **Asuka**: excellent. Top hits are obvious Asuka plates; auburn is rare
  in the broader artbook palette so false positives are minimal.
- **Rei**: excellent. Pale blue + red eyes is a near-unique combination.
- **Misato**: fair. Violet (#5B4B7A) is also Eva-01's mech color and shows
  up as cool shadow tone on red plugsuits, kimono fabric, and twilight
  skies. Real Misato hits (e.g., the iconic wink-with-Shinji on
  `Die Sterne/p020_i049.jpg`) rank in the top 10, but false positives from
  Eva-01 and Asuka plugsuit pages also appear there. Browse the contact
  sheet — about 3–4 of every 10 top Misato hits are real.

## Module map

| File                   | What it does                                   |
| ---------------------- | ---------------------------------------------- |
| `src/main.rs`          | clap CLI: `rank` / `extract` / `all`           |
| `src/color.rs`         | RGB ↔ HSV ↔ CIELAB conversions, ΔE76, hue band |
| `src/config.rs`        | Character + ColorTarget definitions            |
| `src/classify.rs`      | Per-pixel match (HSV ∧ LAB), mask building     |
| `src/connected.rs`     | 4-connected component labeling, blob shape     |
| `src/io.rs`            | Page enumeration across artbook roots          |
| `src/rank.rs`          | Stage 1 — page certainty scoring (parallel)    |
| `src/extract.rs`       | Stage 2 — full-res crops + page copies         |
| `src/contact.rs`       | Stage 3 — static HTML contact sheets           |

## Tests

```bash
cargo test --release
```

Color-math unit tests cover hex parsing, HSV roundtrip on Misato hair,
hue-wraparound on Misato jacket / Asuka plugsuit reds, LAB endpoints, and
hue-band membership.

---

**Misato's image must be maintained.** This pipeline is one small step.
