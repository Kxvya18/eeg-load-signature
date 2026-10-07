
## 2026-10-07 — dev-only probes (sealed subjects untouched)
X01 block confound (N=11 dev x 3 sessions, within-session, TS+LR):
  N-back shuffled 0.959 | N-back leave-block-out 0.904 | block identity within level 0.737 (chance 0.33) | MATB shuffled 0.975
  Pre-stated criterion (C >> 0.33 AND A-B >= 0.10): C passes, A-B = 0.055 fails -> "published SOTA is time-decoding" framing DROPPED.
X02 label-free time-direction nulling (within-block feature slopes, all sessions incl. unlabeled S3), FB-TS-RC + LR, S1+S2 -> S3:
  MATB k0 0.630 -> k1..k32 0.616-0.625 ; N-back k0 0.475 -> 0.457-0.466.  NO-GO (nulling does not help, slightly hurts).
Observation: N-back within-session leave-block-out 0.904 vs cross-session 0.475 -> the discriminative structure is session-specific.
X03 workload-direction stability (high-minus-low class-mean difference; cosine), dev N=11:
  re-centered broadband TS : N-back within-session (block halves) 0.731 | across sessions 0.097 | MATB across 0.324
  sensor log band power    : N-back within 0.595 | across 0.413 | MATB across 0.477
  -> in fixed sensor coordinates the code is moderately stable; per-session re-centering (whitening frame)
     makes cross-session directions near-orthogonal. Candidate mechanism for cross-session failure: frame
     rotation introduced by session-specific whitening (cf. RPA rotation step, Rodrigues 2019) — check prior art.
2026-10-07 prior-art check (label-free rotation alignment): CROWDED -> rotation-fix method dropped.
  RMRA (IEEE 9438678, unsupervised rotation), SPDAlign (arXiv 2610.06315, unsupervised Wasserstein Procrustes + theorem
  that forward-model shifts are recoverable by linear SPD maps), SPDIM (ICLR 2025), OSPDIM (arXiv 2608.05315),
  physics-informed URDA (arXiv 2403.15415), Mellot et al. harmonization (re-center/re-scale/rotate), RPA (TBME 2019).
2026-10-07 new direction scoped (prior art): content fingerprinting of EEG recordings for dataset integrity /
  pretraining-evaluation contamination audit. Prior art: subject-split leakage (Brookshire 2024), EEG biometric
  perceptual hashing for authentication (UNT), FM papers state downstream recordings "may overlap TUEG" without
  verification (arXiv 2607.24519, 2605.08495). No recording-level, preprocessing-invariant EEG fingerprint found.

## 2026-10-07 — prior-art check: Ideas 1 (task-general load) + 4 (load vs time-on-task) + NWCF
Closest work:
- Task-general load: "Continuous decoding of cognitive load ... task-general and task-specific correlates" (J Neural Eng 2020,
  10.1088/1741-2552/abb9bc); "Aperiodic and oscillatory systems underpinning human domain-general cognition"
  (Commun Biol 2024): aperiodic broadband power and alpha/beta generalise across tasks; mid-frontal theta does not.
  Cross-task PSD topographies are task-dependent (PMC8637174). Cross-task ML: tensor TL, SCVCNet, SCDA (JBHI 2024), ASFM.
- Load vs time-on-task: ICA-based correlates of fatigue/effort/workload in ATC (PMC5447707); inverse time-on-task theta
  effects (Arnau 2021). Descriptive separation only; no identifiable decomposition, no cross-task/cross-session decoder.
- Topology for workload: "Topological EEG-based FC analysis for mental workload state recognition" (2023, incl. MATB).
- COG-BCI original paper: within-task accuracy MATB 69.4%, N-back 65.0%; no cross-task transfer reported.
Consequences for design:
- Aperiodic (broadband offset/exponent) features MUST be a competing feature family (strongest prior for task-generality).
- Novel element must be the method: joint identifiable decomposition {task-general load, task-specific load, time-on-task,
  session} with a worst-case (max-min over tasks) objective; evaluated leave-one-task-out x cross-session.
- NWCF competes as the network-topology family against the 2023 topological-FC workload method.

## 2026-10-07 — feature families (F01) and validation (V01)
- Task order per session from notebook.mat (codes 3/4/5 = 2/1/0-back, 6/7/8 = MATB easy/med/diff); order is
  pseudorandomised per subject x session -> load level and time-on-task (slot) are separable by design.
- F01 families on identical 4-s windows: BP, APER (robust log-log 1/f fit), OSC (power above 1/f), RIEM
  (theta/alpha/beta re-centred tangent space), NWCF (frozen v3.1 descriptors, M58, aec_orth+wpli, 3 bands,
  unchanged), TOPO (classical Rips/graph/clique from the same cache).
- V01: fast aperiodic fit vs specparam 2.0 (fixed mode), n=360 channel-windows: offset r=0.925, exponent r=0.950.

## 2026-10-07 — X10 cross-task baselines (dev N=11; low vs high training, AUC on unseen task; C=0.1 fixed)
| setting | BP | APER | OSC | RIEM | NWCF | TOPO | ALL |
|---|---|---|---|---|---|---|---|
| same task, S1+S2->S3, MATB | .909 | .882 | .841 | .963 | .676 | .608 | .971 |
| same task, S1+S2->S3, N-back | .788 | .662 | .720 | .828 | .622 | .613 | .854 |
| cross task, S1+S2->S3, N-back->MATB | .564 | .582 | .575 | .713 | .514 | .482 | .708 |
| cross task, S1+S2->S3, MATB->N-back | .669 | .629 | .637 | .635 | .550 | .529 | .683 |
| cross task, same session, N-back->MATB | .641 | .580 | .614 | .583 | .511 | .527 | .590 |
| cross task, same session, MATB->N-back | .617 | .584 | .632 | .595 | .528 | .515 | .607 |
- Task-specificity confirmed: AUC 0.83-0.97 within task -> 0.51-0.71 across tasks. Room for a task-general method.
- NWCF (frozen) and classical topology are weak for load (0.61-0.68 within task, ~chance across tasks).
- Time leakage (Spearman of score with time-on-task within level) is small and negative (-0.01 to -0.16),
  largest for RIEM/ALL within MATB: decoder outputs drift down with time-on-task.
