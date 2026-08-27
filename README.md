# mf6_to_flopy

Translate an existing **MODFLOW 6 model folder** (mfsim.nam + package files)
into a clean, readable **FloPy build script** of explicit constructor calls,
e.g. `ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)`.

Running the generated script rebuilds the model (`sim.write_simulation()`) — a
full round-trip: **MF6 files → FloPy code → MF6 files** — and can optionally
**run the model and post-process the results** (heads, specific discharge, and
the List Budget file), mirroring a standard FloPy workflow.)

## Install

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick start (command line)

```bat
python mf6_to_flopy.py "examples\01_structured_dis" -o build_model.py --postprocess
python build_model.py
```

`--postprocess` generates a complete **build → run → analyse** script: it runs
the model, plots heads (one PNG per layer), plots specific-discharge vectors,
and reads/prints the List Budget file (saving budget CSVs). It implies `--run`
and `--plots`, and auto-adds an Output Control (OC) package if the model lacks
one so heads and budget are recorded.

Try the other examples the same way:

```bat
python mf6_to_flopy.py "examples\02_unstructured_disv"  -o build_disv.py
python mf6_to_flopy.py "examples\03_unstructured_disu"  -o build_disu.py
python mf6_to_flopy.py "examples\04_coupled_gwf_gwt"    -o build_coupled.py
```

## Quick start (notebook)

Open `notebooks/mf6_to_flopy_tutorial.ipynb` and run the cells top to bottom
(*Kernel → Restart & Run All*). It points at `examples/01_structured_dis` by
default; change the `model_folder` variable in step 2 to convert your own model.

## Options

| flag | effect |
|------|--------|
| `--plots` | append head-contour plotting code |
| `--export` | append grid + attribute shapefile export |
| `--vtk` | with `--export`, also write VTK (needs `vtk`) |
| `--keep-external` | keep `OPEN/CLOSE` refs instead of inlining data |
| `--run` | append `sim.run_simulation()` |
| `--postprocess` | full workflow: run + heads + specific discharge + List Budget (implies `--run`/`--plots`) |
| `--no-sanitize` | disable auto-fix of foreign/absolute paths in name files |
| `--sim-ws DIR` | workspace for the rebuilt model (default `rebuilt`) |

## Robustness notes

* **Foreign/absolute paths** — models exported by ModelMuse / GMS often embed
  absolute paths from the build machine (e.g. `C:/Users/Someone/.../Model1.tdis`).
  The tool auto-sanitizes these at load time on a temporary copy (your files are
  never modified). Use `--no-sanitize` to disable, or run the standalone
  `fix_mf6_paths.py` to repair the folder in place (with a backup).
* **Any package / grid** — the tool introspects the loaded simulation rather
  than hard-coding packages, so it handles DIS/DISV/DISU, boundary packages,
  observations (OBS), storage, transport, exchanges, and more.

See `MF6_to_FloPy_Documentation.docx` for full details.
