# Cross-session workload estimation (COG-BCI) — protocol and development log

## Benchmark (mirrors the 1st pBCI competition; Frontiers in Neuroergonomics 2022)
- Task: MATB easy / medium / difficult (3-class). Train on sessions S1+S2, test on S3.
- 2-s non-overlapping epochs. Metric: per-subject accuracy (balanced classes), averaged over subjects.
- Reference bar: competition winner 54.3% on 15 subjects (different but related data).
- Secondary benchmark: N-back 0/1/2 with the same session protocol.

## Data
COG-BCI cleaned cache (nwcf_v31 c00: 128 Hz, 58 channels, ASR, average reference).
Disclosure: these recordings were analysed before for other questions (NWCF on N-back
0- vs 2-back), never for cross-session decoding.

## Split (fixed before any decoding; docs/subject_split.json, seed 20261007)
Development half (14 subjects): all method design and tuning.
Sealed test half (15 subjects): evaluated once, with a frozen method, after a dated pre-registration.

## Log
- 2026-10-07: split created; baseline feasibility on the development half only.
- 2026-10-07: DATA DEFECT FOUND. Baseline B01 on the development half showed identical results for
  subjects 20, 23 and 27. Their cleaned EEG is byte-identical to that of subjects 18, 19, 21, 22 and
  24 in every session, while the dataset's notebook.mat gives these subjects distinct session dates
  and task orders. Conclusion: the public release contains one participant's EEG under 8 IDs.
  The v1 split leaked one person into both halves and is discarded, and the B01 v1 numbers are void.
  New split v2 (seed 20261008) on 22 distinct participants (one copy kept as subject 18):
  11 development / 11 sealed test.
- 2026-10-07: split v3 replaces v2 (no results computed under v2). Development = the 11 distinct
  subjects whose baseline accuracies were viewed in the voided v1 run (1,3,4,7,8,9,10,14,16,17,28);
  sealed test = the 11 never-analysed distinct subjects (2,5,6,11,12,13,15,18,25,26,29).
  Rationale: the sealed half must contain no subject whose results have been seen.
