# Parameter Files — Classic vs Vector Blocking

## Overview

Every dataset `SXX` has two parameter files:

| File | Blocking method |
|---|---|
| `SXX-parms.txt` | **Classic DWM** — exact token string matching |
| `SXX-vector-parms.txt` | **Vector DWM** — approximate nearest-neighbour matching |

All other parameters (`mu`, `beta`, `comparator`, `sigma`, etc.) are identical
between the two files so that results are directly comparable.

---

## How Classic Blocking Works (`SXX-parms.txt`)

Classic blocking is handled by `DWM42_BuildBlockPairs.py`.

For each record, tokens that pass three filters are selected as **blocking tokens**:
- length ≥ `minBlkTokenLen`
- not all-digits (when `excludeNumericBlocks = True`)
- corpus frequency between 2 and `beta`

Two records land in the same **block** only if they share the **exact same
blocking token string**. Every pair of records within the same block becomes a
candidate pair, which is then scored by DWM55.

**Key constraint:** `"smith"` and `"smth"` never meet — different strings, different
blocks, no candidate pair generated between them.

---

## How Vector Blocking Works (`SXX-vector-parms.txt`)

Vector blocking is handled by `DWM42_VectorBlockPairs.py`.

The same three token filters are applied, but instead of grouping records by
identical token strings, each record is encoded as a **multi-hot vector** over
the vocabulary of all qualifying blocking tokens. An **HNSW approximate
nearest-neighbour (ANN) index** then finds records whose blocking-token vectors
are close in cosine space.

Steps inside `DWM42_VectorBlockPairs`:

1. Apply the same token filters as classic blocking (length, digits, beta)
2. Encode each record as a sparse one-hot vector per blocking token, then
   aggregate into one multi-hot vector per record
3. L2-normalise all vectors so cosine similarity equals the dot product
4. Build an HNSW index over the normalised vectors
5. Retrieve the top-k approximate neighbours per record
6. Re-compute exact cosine similarity for each candidate pair and keep only
   pairs above `vectorBlockThreshold`
7. Return the surviving pairs in `"refID_A|refID_B"` format — identical to
   what classic blocking returns

Everything downstream (DWM55 scoring, DWM80 transitive closure, clustering) is
**unchanged**. The only difference is which candidate pairs reach DWM55.

**Key advantage:** records with similar-but-not-identical blocking tokens (e.g.
slight spelling variations) can still become candidate pairs, improving recall
at the blocking stage without changing the scoring logic.

---

## The Three New Parameters

These appear only in `SXX-vector-parms.txt` files:

```
useVectorBlocking    = True
vectorBlockThreshold = 0.5
vectorBlockTopK      = 10
```

| Parameter | Type | Default | Effect |
|---|---|---|---|
| `useVectorBlocking` | Boolean | `False` | `True` activates vector blocking; `False` runs classic DWM42 (safe default — nothing breaks if the parameter is absent) |
| `vectorBlockThreshold` | Float 0–1 | `0.5` | Cosine similarity a pair must reach to become a candidate. Lower = more pairs (looser blocking); higher = fewer pairs (stricter, approaches exact-match behaviour near 1.0) |
| `vectorBlockTopK` | Integer | `10` | Number of approximate neighbours retrieved per record before exact refinement. Higher = more recall, more computation |

### Tuning `vectorBlockThreshold`

| Threshold range | Effect |
|---|---|
| `0.3 – 0.4` | Loose — many more candidate pairs than classic; catches fuzzier matches but gives DWM55 more work |
| `0.5` | Balanced starting point (default in all vector parms files) |
| `0.7 – 0.9` | Tight — approaches exact-match behaviour, pair counts similar to classic blocking |

Compare the `Total Unduplicated Pairs` line in the log between a classic run
and a vector run to see how much the blocking has widened or narrowed.

---

## Running a Classic vs Vector Comparison

```bash
python DWM00_Driver.py
# → Enter 1
# → Enter S12-parms.txt          (classic run)

python DWM00_Driver.py
# → Enter 1
# → Enter S12-vector-parms.txt   (vector run)
```

Or run both in a single session using a list file:

```
# list.txt
S12-parms.txt
S12-vector-parms.txt
```

```bash
python DWM00_Driver.py
# → Enter 2
# → Enter list.txt
```

Results for each run are written to the log file (`DWM_Log_YYYYMMDD_HH_MM.txt`)
and the Excel report (`DWM_Results_YYYYMMDD_HH_MM.xlsx`).

---

## File Reference

| Dataset | Classic parms | Vector parms |
|---|---|---|
| S2 | `S2-parms.txt` | `S2-vector-parms.txt` |
| S3 | `S3-parms.txt` | `S3-vector-parms.txt` |
| S4 | `S4-parms.txt` | `S4-vector-parms.txt` |
| S5 | `S5-parms.txt` | `S5-vector-parms.txt` |
| S6 | `S6-parms.txt` | `S6-vector-parms.txt` |
| S7 | `S7-parms.txt` | `S7-vector-parms.txt` |
| S8 | `S8-parms.txt` | `S8-vector-parms.txt` |
| S12 | `S12-parms.txt` | `S12-vector-parms.txt` |
| S14 | `S14-parms.txt` | `S14-vector-parms.txt` |
| S16 | `S16-parms.txt` | `S16-vector-parms.txt` |
| S1G | `S1G-parms.txt` | `S1G-vector-parms.txt` |
