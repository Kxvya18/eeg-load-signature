# EEG Topology Brain Dynamics

A research-oriented Python project for analyzing workload-related EEG brain dynamics using functional connectivity, graph-based modeling, topological data analysis, and machine-learning validation. The project studies whether changes between low-workload and high-workload EEG states can be captured through topology-aware representations of brain networks.

This repository is prepared as a compact GitHub package. It keeps the source code, one representative final output set, method-validation artifacts, cross-configuration checks, and robustness summaries. Raw EEG data, large intermediate topology dumps, repeated configuration folders, PDFs, caches, and redundant plots are excluded so the project stays readable and professional.

## Research Motivation

EEG workload analysis is often approached through spectral power, channel-level statistics, or conventional connectivity measures. This project explores a richer question:

> Can the shape of EEG functional-connectivity networks reveal workload-related brain-state structure beyond standard signal features?

To answer this, the pipeline converts EEG recordings into windowed connectivity networks, transforms those networks into graph and topological representations, extracts persistence-based and clique-based descriptors, and evaluates whether those descriptors separate workload conditions and remain stable under validation checks.

The project combines signal processing, network science, persistent homology, statistical testing, and machine-learning evaluation in a single reproducible workflow.

## Dataset Context

The project is designed around the STEW EEG workload dataset structure. In this workflow, each subject has paired low-workload and high-workload EEG recordings stored locally as text files such as:

```text
STEW Dataset/
  sub01_lo.txt
  sub01_hi.txt
  sub02_lo.txt
  sub02_hi.txt
  ...
```

The raw dataset is not included in this repository. EEG datasets are usually large, and public biomedical datasets may have separate access and usage terms. The README therefore documents the expected local folder structure while keeping the GitHub repository focused on code and compact result artifacts.

From the implemented pipeline:

- EEG recordings use 14 channels: `AF3`, `F7`, `F3`, `FC5`, `T7`, `P7`, `O1`, `O2`, `P8`, `T8`, `FC6`, `F4`, `F8`, `AF4`.
- Sampling rate is handled as 128 Hz.
- Each subject has paired low-workload and high-workload recordings.
- The analysis uses frontal, posterior, temporal, whole-head, and confirmatory frontal-region channel groupings.
- The representative included output set reports 40 paired subjects after preprocessing/exclusion checks.

## Project Pipeline

```text
Raw STEW EEG files
        |
        v
Signal conversion and referencing
        |
        v
Filtering, artifact handling, and windowing
        |
        v
Alpha-band signal and functional-connectivity estimation
        |
        v
Graph and topology feature extraction
        |
        v
Statistical testing, controls, and surrogate checks
        |
        v
Feature-family comparison and ML validation
        |
        v
Figures, tables, robustness summaries, and cross-config checks
```

## Preprocessing Workflow

The preprocessing pipeline in `stew_pipeline.py` performs the main EEG preparation steps before topology extraction:

1. Loads paired low/high workload recordings for each subject.
2. Converts raw values into microvolt-scale EEG values using project constants.
3. Applies average referencing across channels.
4. Creates MNE-compatible EEG objects for cleaning and downstream processing.
5. Applies high-pass filtering, notch filtering, and alpha-band filtering.
6. Uses artifact handling with ASR-style cleaning where possible.
7. Segments recordings into fixed-length analysis windows.
8. Builds channel-region groupings for frontal, posterior, temporal, whole-head, and confirmatory frontal analyses.

The pipeline uses 4-second windows and focuses strongly on alpha-band functional-connectivity structure because alpha activity is often relevant in cognitive workload and attention-related EEG analysis.

## Functional-Connectivity Topology and Feature Families

After preprocessing, the project estimates alpha-band functional connectivity and treats each connectivity matrix as a structured object for geometric and topological analysis. The emphasis is not only on whether individual channels change under workload, but on whether workload induces a measurable reorganization of mesoscale brain-network geometry.

The implemented feature space includes:

- alpha-envelope, phase-locking, and correlation-derived connectivity operators
- region-conditioned graph representations over frontal, posterior, temporal, confirmatory-frontal, and whole-head channel systems
- H1 persistence summaries capturing loop-like organization in neural connectivity geometry
- persistence landscapes and persistence images for vectorized topological learning
- dynamic landscape-volatility features describing temporal instability of topology across windows
- persistent-cocycle circular-coordinate summaries for exploratory phase-geometry interpretation
- surrogate-resistant topology descriptors evaluated against phase-randomized and IAAFT-style controls
- topology-vector representations used for feature-family ablation and leave-one-subject-out machine-learning validation

This design treats topology as a candidate mesoscopic biomarker: a representation of coordinated network organization that sits between raw EEG channel signals and high-level cognitive workload labels.

## NWCF: Custom Neurophysiological Weighted Clique Filtration

`stew_nwcf.py` implements Neurophysiological Weighted Clique Filtration (NWCF), a custom method developed in this project for topology-aware EEG functional-connectivity analysis. The central motivation is that cognitive workload may reorganize the mesoscale coordination structure of the brain, not merely the amplitude of isolated channels or the strength of individual pairwise connections.

NWCF introduces a clique-cohesion reweighting of the functional-connectivity distance matrix before persistent homology is computed. A pair of EEG channels may be weakly coupled directly, but still participate in a strongly supported triadic motif through a third channel. NWCF uses this local clique support to encode higher-order neurophysiological organization into the filtration itself.

In this project, NWCF is used as a proposed mesoscopic descriptor of workload-sensitive brain-network organization. It is designed to capture coordinated frontal and distributed connectivity patterns that conventional alpha power, direct FC strength, and ordinary graph statistics may not represent adequately.

The repository includes method-validation artifacts for NWCF under:

- counterfactual filtration checks
- clique-order ablation
- filtration responsibility mapping
- method-control summaries
- channel-level and edge-level responsibility analyses

These outputs are stored in:

```text
stew_nwcf_method_validation/
```

## PDRH: Persistent Diagram Recurrence Homology

`stew_pdrh.py` implements Persistent Diagram Recurrence Homology style descriptors. Instead of analyzing only a single connectivity window, PDRH studies how persistence diagrams evolve across the session.

The method builds a recurrence-style representation from the sequence of window-level persistence diagrams and summarizes whether the session-level topology shows recurring or cyclic structure across time. This is useful for studying dynamic brain-state organization rather than only static snapshots.

## Statistical Analysis and Feature Selection

The analysis scripts evaluate both confirmatory and exploratory features. The workflow includes:

- paired low/high workload comparison
- rank-biserial correlation style effect summaries
- corrected significance reporting
- permutation-style checks where implemented
- surrogate controls using phase-randomized and IAAFT-style comparisons
- feature-family ablation
- topology-versus-alpha-power controls
- exclusion-sensitivity analysis
- cross-analysis evidence mapping
- edge-level and node-level topology-driver summaries

Feature selection is treated as part of the research workflow rather than a single black-box step. The project compares feature families such as alpha power, conventional connectivity, topology summaries, persistence landscapes, persistence images, and topology-vector representations. It then evaluates which families contribute useful separation between workload states.

## Machine-Learning Validation

Machine-learning is used as secondary validation for the extracted EEG/topology features. The included result artifacts contain model-comparison and feature-family evaluation outputs such as:

- leave-one-subject-out style validation
- balanced accuracy
- ROC-AUC
- sensitivity and specificity
- confusion-count summaries
- feature-family ablation
- topology-vector model comparisons
- linear model feature-importance stability

The representative output set includes tables such as:

```text
Table26_MLFeatureFamilyAblation.csv
Table27_MLFeatureImportanceStability.csv
Table29_MLTopologyVectorModels.csv
```

This keeps the machine-learning component tied to interpretable research questions: which signal, graph, or topology representations carry workload-related information?

## Representative Results Included

The compact output package keeps one representative final run:

```text
stew_output_final_84aec/
```

This folder includes selected figures, CSV tables, configuration snapshots, and summaries. It contains outputs for:

- confirmatory frontal topology analysis
- regional and whole-head topology summaries
- topology controlling for alpha power
- classification and feature-family evaluation
- dynamic landscape volatility
- frontal functional-connectivity topology
- edge-level topology-driver analysis
- subjective-rating association checks
- topology-vector machine-learning models
- network node-driver summaries

The included summary reports a primary frontal dynamic alpha-FC topology effect and keeps classification as a secondary validation component rather than the only goal of the project.

## Robustness and Cross-Configuration Checks

The repository includes compact validation folders:

| Folder | Purpose |
|---|---|
| `stew_nwcf_method_validation/` | Validates clique-filtration behavior and method controls |
| `stew_nwcf_cross_config_generalization/` | Checks whether NWCF-style effects generalize across connectivity/configuration variants |
| `stew_robustness_summary/` | Stores compact robustness outputs and recurrence summaries |

These checks are included because EEG research pipelines can be sensitive to preprocessing choices, connectivity definitions, window length, and feature construction. The project therefore keeps validation artifacts alongside the main output set.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── stew_pipeline.py
├── stew_analysis.py
├── stew_advanced_analysis.py
├── stew_topology_core.py
├── stew_topology_extensions.py
├── stew_nwcf.py
├── stew_pdrh.py
├── stew_robustness.py
├── stew_run_all_configs.py
├── stew_nwcf_cross_config_generalization.py
├── stew_*_plots.py
├── stew_output_final_84aec/
├── stew_nwcf_method_validation/
├── stew_nwcf_cross_config_generalization/
└── stew_robustness_summary/
```

## Core Scripts

| File | Purpose |
|---|---|
| `stew_pipeline.py` | Main EEG preprocessing, windowing, connectivity, and topology feature extraction workflow |
| `stew_analysis.py` | Statistical analysis, feature-family comparison, model evaluation, and summary table generation |
| `stew_advanced_analysis.py` | Extended topology, feature, and machine-learning analyses |
| `stew_topology_core.py` | Core persistence and topology helper functions |
| `stew_topology_extensions.py` | Additional topology descriptors, signal summaries, and representation helpers |
| `stew_nwcf.py` | Neurophysiological Weighted Clique Filtration implementation |
| `stew_pdrh.py` | Persistent Diagram Recurrence Homology descriptors |
| `stew_robustness.py` | Robustness and sensitivity analysis |
| `stew_nwcf_cross_config_generalization.py` | Cross-configuration analysis utility |
| `stew_run_all_configs.py` | Batch execution utility for multiple connectivity/configuration settings |

## Setup

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Some EEG and topology dependencies can be platform-sensitive. If installation fails, install the scientific Python stack first, then install `mne`, `asrpy`, and `ripser` separately.

## Example Commands

Run the main processing pipeline after placing the dataset locally:

```bash
python stew_pipeline.py
```

Run analysis on generated feature outputs:

```bash
python stew_analysis.py
```

Run advanced topology analysis:

```bash
python stew_advanced_analysis.py
```

Run robustness analysis:

```bash
python stew_robustness.py
```

Run cross-configuration comparison if required artifacts are available:

```bash
python stew_nwcf_cross_config_generalization.py
```

## GitHub Packaging Notes

This repository excludes raw datasets, generated caches, PDFs, repeated configuration folders, and large intermediate TDA result dumps. The included artifacts are selected to make the project understandable as a research and portfolio repository while still allowing the full workflow to be regenerated locally with the dataset.
