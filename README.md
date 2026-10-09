# McTernan Lab Repository

Analysis and modelling code from the McTernan Lab (The Francis Crick Institute) for the **PhosphoMAX** project. The project measures how macrocyclic phosphonate hosts (e.g. PMP5, PMP6) bind drug-like guest molecules.

The repository has two independent parts:

| Folder | What it is | Runs on |
|---|---|---|
| [`FDA_launcher/`](FDA_launcher) | **PhosphoMAX**, a desktop app for analysing fluorescent indicator displacement assays (FDA/IDA) from 384-well plate-reader data | Your Mac (or any machine with Python + Qt) |
| [`Computational Modelling/`](Computational%20Modelling) | Host–guest docking → complex assembly → CREST conformer-search pipeline | A SLURM HPC cluster |

---

## Repository layout

```
McTernan-Lab-Repository/
├── README.md
├── FDA_launcher/
│   ├── PhosphoMAX.app          # double-click launcher (macOS)
│   ├── launch_phosphomax.py    # finds a suitable Python, then starts the GUI
│   ├── fda_launcher.py         # the GUI (PySide6): mode picker + all analysis windows
│   ├── pipeline_fda.py         # Direct binding (Kd) pipeline
│   ├── pipeline_ki.py          # Competitive binding (Ki) pipeline
│   ├── pipeline_spectral.py    # Spectral-scan (Ex/Em) pipeline
│   ├── chromatic_db.py         # plate-reader filter → dye resolution
│   ├── session_io.py           # save/load .phosmax session files
│   └── layout_utils.py         # shared figure-layout maths
└── Computational Modelling/
    ├── pipeline.sh             # main driver (phases 0–5)
    ├── dock_worker.sh          # SLURM array task: one Vina docking job
    ├── crest_worker.sh         # SLURM array task: one CREST job
    ├── prepare_guests.py       # SMILES → 3D full-H guest .mol2 (RDKit)
    ├── reform_complex.py       # host + docked guest → rehydrogenated complex
    ├── classify_binding.py     # bound / unbound classification by geometry
    ├── crest_energy_plot.py    # CREST conformer-energy summary plots
    ├── render_complexes.py     # PyMOL images of complexes
    └── project_manifest.json   # hosts + guest library for the paper
```

---

## 1. PhosphoMAX: FDA analysis app (`FDA_launcher/`)

### What it does

The assay runs in two stages, using Echo-dispensed 384-well plates read on a plate reader:

1. **Direct titration.** The host is titrated against a fixed concentration of fluorescent dye (e.g. DAPI, H33258). This gives the host–dye **Kd** and the host concentration needed for near-maximal dye binding (IC90).
2. **Competitive titration.** Guests are titrated into the pre-formed host–dye complex. Displacement of the dye lowers the fluorescence, which gives each guest's **Ki**.

When the app opens, it asks you to choose a mode:

| Mode | Purpose |
|---|---|
| **Direct Binding (Kd)** | Fits host–dye titrations (one-site or quadratic/ligand-depletion model, chosen by the fit) |
| **Competitive Binding (Ki)** | Fits guest displacement curves with several models (Standard + Cheng–Prusoff, free Hill slope, and Wang's exact competitive model) and picks the best by AICc |
| **Spectral Scan (Ex/Em + Binding)** | Plots excitation/emission scans, then fits binding at a chosen wavelength through the Direct Binding pipeline |
| **Summary Plots** | Heatmaps and other overview plots built from saved fit-result files |
| **Compare Runs** | Overlays repeat experiments, checks reproducibility, and produces pooled fits with optional Grubbs outlier rejection |

Every fit is labelled **PASS**, **INCONCLUSIVE** or **FAIL**. The labels come from adjustable quality gates: adjusted R² (default pass ≥ 0.90, inconclusive band 0.75–0.90), whether Kd/Ki falls inside the tested concentration range (0.1×–10×), residual checks, and others. The defaults can be changed in the GUI.

### Installation

You need Python 3.10+ with:

```
PySide6  numpy  pandas  scipy  matplotlib  seaborn  openpyxl
```

For example, with conda:

```bash
conda create -n phosphomax python=3.10
conda activate phosphomax
pip install PySide6 numpy pandas scipy matplotlib seaborn openpyxl
```

### Launching

- **macOS:** double-click `FDA_launcher/PhosphoMAX.app`. On first launch it looks for a Python interpreter that can import PySide6, matplotlib, pandas, numpy and scipy. It saves the interpreter it finds to `.phosphomax_launcher.json` so later launches start quickly. Problems are shown in a dialog box and written to `launcher.log`.
  - Keep the `.app` in the same folder as the `.py` files. If you want it somewhere else, make an alias instead of moving it.
  - If you change conda environments, delete `.phosphomax_launcher.json` so the launcher searches again.
- **Any platform:** activate your environment and run
  ```bash
  python FDA_launcher/fda_launcher.py
  ```

### Preparing an experiment folder

Point the app at an experiment folder with this structure:

```
<experiment>/
├── Raw/             # plate-reader exports (.xlsx; .csv for spectral scans)
├── dye_map/         # Echo transfer export(s) for the dye  (.xlsx)
├── host_map/        # Echo transfer export(s) for the host (.xlsx)
├── guest_map/       # Echo transfer export(s) for guests   (.xlsx, competitive only)
├── blank_map/       # Echo transfer export(s) for blank wells (.xlsx)
├── buffer_map/      # optional: tick "buffer" in the app to use
└── exceptions_map/  # optional: Echo "Exceptions" reports; failed dispenses are
                     #   excluded from fitting and plotted as black crosses
```

- **Mapping files** are standard Echo transfer exports. Each must contain the columns `Compound ID`, `Destination Well`, `Destination Concentration`, `Destination Unit` and `Destination Plate Name`. In `blank_map`, blank wells use `Compound ID = blank`, and dye names must exactly match the ones in `dye_map`.
- **Raw file names:** the plate name is the first `_`-separated token that is not a date. For example, `260324_Plate1_….xlsx` gives the plate name `Plate1`, which must match `Destination Plate Name` in the maps.
### Reference tables (not stored in the repo)

The app also reads two small lookup tables. They are not in the repository, so keep your own copies. By default the app looks for them in folders next to `fda_launcher.py`, but you can point it at any folder in the GUI. Each folder can hold one or more `.xlsx` files.

| Folder (default) | Needed for | Columns | Example row |
|---|---|---|---|
| `Comp_Kd/` | Competitive Binding (**required**) | `Host`, `Dye`, `Dye_Concentration` (µM), `Kd_uM` | `PMP6-Na, DAPI, 1, 1.23` |
| `Chromatic_DB/` | Multi-chromatic plates (optional) | `Dye`, `Filter`, `Aliases` (`;`-separated) | `H33258, 355-15/465-20, H33; H332` |

- **Kd table:** these are the host–dye Kd values from your direct titrations. The Ki pipeline needs them for the Cheng–Prusoff / Wang corrections, so add a row whenever you characterise a new host–dye pair.
- **Chromatic DB:** when several dyes are read on one plate, the app reads the filter settings (`Ex-bw/Em-bw`) from each raw file's header and uses this table to work out which dye belongs to which channel. It is only used when *Multi-chromatic plates* is ticked.

### Outputs

Files are written to the output folder you choose, and each file name starts with a timestamp:

```
<output>/
├── output_reports/   # mapping, raw, merged and blank-corrected data (.csv) + QC plots (.pdf)
└── results/
    ├── *_binding_fit_results.xlsx      (or *_competitive_fit_results.xlsx)
    ├── *_data.xlsx                      wide-format FI−F0, one sheet each for PASS / INCONCLUSIVE / FAIL
    └── plots/                           binding-curve PDFs (grid + optional individual curves)
```

Use **💾 Save Session** to save the whole analysis (every setting, dataframe and fit) as a single `.phosmax` file. You can reopen it later on any machine without the original raw data. The format is a plain zip of JSON + NumPy arrays (no pickle), so sessions stay readable as the code changes.

> Ki values can be ranked directly **within** one host/dye pair. Comparisons **across** different dyes are only approximate, because the Cheng–Prusoff correction is different for each dye.

### Version history

Track changes with git commits rather than dated copies of the scripts. Older dated snapshots (`applet_versions/`) were removed in commit `b02000f` and can still be recovered from git history, for example with `git checkout b02000f~1 -- FDA_launcher/applet_versions`.

---

## 2. Computational modelling pipeline (`Computational Modelling/`)

This pipeline predicts host–guest complex structures and runs a conformer search on each one. It goes from docking all host × guest pairs to a CREST GFN-FF conformer ensemble for every complex.

### Pipeline phases

| Phase | Step | Tooling |
|---|---|---|
| 0 | Validate tools, inputs and Python packages | — |
| 0.5 | *(optional)* Build missing guest structures from `guests_smiles.csv` | RDKit (`prepare_guests.py`) |
| 1 | Convert hosts/guests to PDBQT, compute each host's centre of mass, write Vina box configs | Open Babel |
| 2 | Dock every host × guest pair | AutoDock Vina (SLURM array in batch mode) |
| 3 | Join host + top-ranked docked pose, rehydrogenate, run QC (H count, clashes, bonds across host/guest) | Open Babel (`reform_complex.py`) |
| 4 | Build `crest_manifest.tsv` | — |
| 5 | Conformer search for each complex (`--gfnff --quick --noreftopo`, ALPB water) | CREST / xtb (SLURM array) |

In batch mode the whole chain runs from one command: the script submits the docking array, then a dependent job that runs reform → manifest → CREST submission once docking finishes.

### Requirements

- A SLURM cluster
- A conda environment (default name `crest`) containing: `obabel` (Open Babel), `vina`, `crest`, `xtb`, `rdkit`, `numpy`, `scipy`, `matplotlib`
- PyMOL, for `render_complexes.py` only

Cluster-specific settings are in the **USER SETTINGS** block at the top of `pipeline.sh`. Change them before your first run:
`CONDA_ENV`, `CONDA_MODULE`, `CONDA_SETUP`, the partitions (`VINA_PARTITION`, `CREST_PARTITION`), CPU/memory/time limits, the Vina box size and exhaustiveness, and `CREST_SOLVENT`.

### Setting up a project

Copy all the scripts into a project directory on the cluster, then add:

```
project/
├── hosts/              # REQUIRED: host structures (*.mol2 preferred, *.xyz accepted)
│                       #   with the experimentally correct protonation / H count
├── guests/             # guest structures (*.mol2 / *.xyz), and/or…
├── guests_smiles.csv   # optional: name,category,smiles,charge,pubchem_cid
├── charges.csv         # optional: name,charge   (e.g. 6MINUS_PMP6,-6). Unlisted molecules default to 0
└── curated/            # optional: <host>.mol2 with curated protonation, used as-is for every complex of that host
```

Use `.mol2` for hosts if you can. With `.xyz`, Open Babel has to guess bond orders from the geometry, which is less reliable.

### Running

```bash
chmod +x pipeline.sh dock_worker.sh crest_worker.sh

# Test a single pair interactively (inside an srun session on a compute node).
# Pauses at checkpoints so you can inspect each stage.
./pipeline.sh --mode interactive --host 6MINUS_PMP6 --guest Doxorubicin

# Full run: all hosts × all guests, fully chained on SLURM
./pipeline.sh --mode batch

# Resume, skipping stages that are already done
./pipeline.sh --mode batch --skip-prep --skip-dock --skip-reform

# Verbose tracing
DEBUG=1 ./pipeline.sh --mode interactive --host 6MINUS_PMP6 --guest Doxorubicin
```

### Outputs

```
pdbqt/  ligands_pdbqt/  configs/  centers.csv        # Phase 1
docking_results/<host>/<guest>_docked.pdbqt           # Phase 2
docking_results/docking_scores.csv                    # top-1 Vina scores for all pairs
reformed/<host>/<guest>/complex_full.mol2, complex.xyz, reform_summary.txt   # Phase 3
crest_manifest.tsv                                    # Phase 4
crest_results/<host>_<guest>/crest_conformers.xyz …   # Phase 5
pipeline.log                                          # appended on every run
```

### Post-analysis

Run these from the project directory after the pipeline finishes:

```bash
# Classify each complex as bound/unbound using convex-hull containment + contact counts
python3 classify_binding.py --project-dir .            # → binding_classification.csv
#   options: --threshold 50  --ensemble-bound-frac 0.5  --contact-cutoff 4.5  --no-crest

# Plot CREST conformer-energy distributions for every complex
python3 crest_energy_plot.py --dir crest_results --sort span   # → crest_energy_summary.png/.csv

# Render complex images with PyMOL
python3 render_complexes.py --dir reformed                     # docked poses
python3 render_complexes.py --dir reformed --source crest_best # lowest-energy CREST conformer
```

### Notes

- **Metal-containing hosts:** Vina has no atom types for many metals (Pd, Pt, Ru, Co, …). For docking only, `pipeline.sh` maps them to the nearest supported proxy type. CREST runs on the real structure from `reform_complex.py`, so the CREST step is unaffected.
- **No host–guest distance constraint in CREST:** the `--cinp` constraint was removed because of a known crash in CREST 3.0.2 when it is combined with `--gfnff` (crest-lab/crest issues #338, #367, #381). If a guest drifts out of the host during the search, filter those conformers afterwards (for example with `classify_binding.py`).
- **`project_manifest.json`** records the hosts used for the paper (`6MINUS_PMP6`, charge −6; `PMP5_5minus_2`, charge −5) and the 35-guest library with SMILES and PubChem CIDs.

---

## Conventions

- Dates in file and folder names use `YYMMDD` (e.g. `260814_…`).
- Host names follow the form `PMP5-Na`, `PMP6-Na`. Dyes are abbreviated (`DAPI`, `H33` = Hoechst 33258); add new ones and their aliases to the Chromatic DB.
- Generated data (`results/`, `output_reports/`, `results_paper/`, `saved_sessions/`) is git-ignored. Commit code and the small reference tables only.

## References

- Wang, Z.-X. *FEBS Lett.* 1995, 360, 111–114: exact model for competitive binding of two ligands to one site.
- Assay design: DOI [10.1039/d2ob01487d](https://doi.org/10.1039/d2ob01487d); DOI [10.1074/jbc.M115.669333](https://doi.org/10.1074/jbc.M115.669333).
