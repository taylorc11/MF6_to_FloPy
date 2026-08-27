#!/usr/bin/env python3
"""
mf6_to_flopy.py
===============

Translate an existing MODFLOW 6 model folder (mfsim.nam + package files) into a
clean, human-readable FloPy build script of explicit constructor calls, e.g.

    ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=strt)

Running the generated script rebuilds the model (sim.write_simulation()) -- a
full round-trip: MF6 files -> FloPy code -> MF6 files. The generated script can
also be extended to run the model and post-process the results (heads, specific
discharge, and the List Budget file), mirroring a standard FloPy workflow.

Key features
------------
  --plots         Append head-contour plotting code (one PNG per layer).
  --export        Append grid + attribute shapefile export code.
  --vtk           With --export, also export the model to VTK (needs 'vtk').
  --keep-external Preserve OPEN/CLOSE external file references instead of inlining.
  --run           Append sim.run_simulation().
  --postprocess   Append a full post-processing block: run the model, read and
                  plot heads, plot specific-discharge vectors, and read/print
                  the List Budget file (implies --run and --plots). This makes
                  the generated script a complete build-run-analyse workflow.
  --no-sanitize   Do not auto-fix foreign/absolute paths in the name files.

The tool is package-agnostic: it introspects whatever the loaded simulation
contains (any grid type, any packages, multiple models, solvers, exchanges) and
also auto-repairs foreign/absolute paths that ModelMuse / GMS sometimes bake
into the name files.
"""

import argparse
import importlib.util   # submodule import required; 'import importlib' alone
                        # does not attach .util on all Python builds.
import inspect
import os
import re
import shutil
import sys
import tempfile


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
_REQUIRED = {"numpy": "numpy", "flopy": "flopy"}


def check_dependencies(need_plots=False, need_export=False, need_vtk=False,
                       need_postprocess=False):
    missing = [pip for imp, pip in _REQUIRED.items()
               if importlib.util.find_spec(imp) is None]
    if missing:
        print("=" * 70)
        print(" Missing required package(s): " + ", ".join(missing))
        print(" Install with:  python -m pip install " + " ".join(missing))
        print("=" * 70)
        sys.exit(1)

    warn = []
    if (need_plots or need_postprocess) and importlib.util.find_spec("matplotlib") is None:
        warn.append("matplotlib (for --plots/--postprocess)")
    if need_postprocess and importlib.util.find_spec("pandas") is None:
        warn.append("pandas (for --postprocess List Budget tables)")
    if need_export and importlib.util.find_spec("geopandas") is None:
        warn.append("geopandas (for --export shapefiles)")
    if need_vtk and importlib.util.find_spec("vtk") is None:
        warn.append("vtk (for --vtk)")
    if warn:
        print("-" * 70)
        print(" NOTE: optional package(s) not found: " + ", ".join(warn))
        print(" The generated script will still be written; install them so")
        print(" the corresponding cells run:")
        pkgs = " ".join(w.split()[0] for w in warn)
        print(f"     python -m pip install {pkgs}")
        print("-" * 70)


import numpy as np  # noqa: E402


# ===========================================================================
# Path sanitizer
# ===========================================================================
def _basename(token):
    t = token.strip().strip("'\"")
    return os.path.basename(t.replace("\\", "/"))


def _token_is_fixable(token, local_files):
    t = token.strip().strip("'\"")
    if ("/" in t) or ("\\" in t):
        return _basename(t) in local_files
    return False


def sanitize_model_folder(model_folder, verbose=True):
    if not os.path.isdir(model_folder):
        return model_folder, False
    local_files = set(os.listdir(model_folder))
    nam_files = [f for f in local_files if f.lower().endswith(".nam")]
    if not nam_files:
        return model_folder, False

    fixes = []
    for nf in nam_files:
        with open(os.path.join(model_folder, nf), "r",
                  encoding="utf-8", errors="replace") as fh:
            for line in fh:
                for tok in line.split():
                    if _token_is_fixable(tok, local_files):
                        fixes.append((nf, tok.strip().strip("'\"")))
    if not fixes:
        return model_folder, False

    if verbose:
        print("Detected absolute/foreign path(s) in name files; "
              "auto-sanitizing a temporary copy (originals untouched):")
        for nf, old in fixes:
            print(f"    [{nf}] {old}  ->  {_basename(old)}")

    tmp = tempfile.mkdtemp(prefix="mf6_sanitized_")
    for f in os.listdir(model_folder):
        src = os.path.join(model_folder, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(tmp, f))
    for nf in nam_files:
        path = os.path.join(tmp, nf)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        out = []
        for line in lines:
            parts = re.split(r"(\s+)", line)
            for i, p in enumerate(parts):
                if _token_is_fixable(p, local_files):
                    parts[i] = _basename(p)
            out.append("".join(parts))
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(out)
    return tmp, True


# ===========================================================================
# Value formatting helpers
# ===========================================================================
_AUTO_SKIP = {"maxbound", "nbound"}


def _norm(name):
    return name.replace("-", "_")


def _py_scalar(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        f = float(v)
        return int(f) if f.is_integer() else f
    return v


def _collapse_array(arr):
    a = np.asarray(arr)
    if a.size == 0:
        return a.tolist()
    if np.all(a == a.flat[0]):
        return _py_scalar(a.flat[0])
    if a.ndim == 3:
        per_layer, constant_layers = [], True
        for lay in range(a.shape[0]):
            layer = a[lay]
            if np.all(layer == layer.flat[0]):
                per_layer.append(_py_scalar(layer.flat[0]))
            else:
                constant_layers = False
                break
        if constant_layers:
            return per_layer
    return a.tolist()


def _flatten_keyword_record(dname, rec):
    key = _norm(dname).lower()
    tokens = []
    for row in np.asarray(rec).tolist():
        row = row if isinstance(row, (list, tuple)) else [row]
        for tok in row:
            if tok is None:
                continue
            if isinstance(tok, (bool, np.bool_)):
                continue
            if isinstance(tok, str) and tok.lower() == key:
                continue
            tokens.append(_py_scalar(tok))
    return tokens


def _fmt_recarray(rec, dname=None):
    names = list(rec.dtype.names)
    if len(rec) == 1 and len(names) == 1:
        return _py_scalar(rec[0][0])
    rows = []
    for row in rec:
        rec_out = []
        for name in names:
            val = row[name]
            # Skip unset fields: None, or NaN (how MF6/FloPy marks an empty
            # optional column such as an aux value or boundname that was
            # declared but not populated).
            if val is None:
                continue
            if isinstance(val, (float, np.floating)) and np.isnan(val):
                continue
            if name in ("cellid", "id", "id2") or isinstance(
                val, (tuple, list, np.ndarray)
            ):
                try:
                    rec_out.append(tuple(int(x) for x in val))
                    continue
                except (TypeError, ValueError):
                    pass
            rec_out.append(_py_scalar(val))
        rows.append(rec_out)
    return rows


def _fmt_value(val, dname=None):
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, (np.integer, np.floating, np.bool_)):
        return _py_scalar(val)
    if isinstance(val, np.recarray):
        return _fmt_recarray(val, dname)
    if isinstance(val, np.ndarray):
        if val.dtype.names:
            return _fmt_recarray(val, dname)
        return _collapse_array(val)
    if isinstance(val, dict):
        if set(val.keys()) & {"filename", "data", "factor", "iprn"}:
            return {k: _fmt_value(v, dname) for k, v in val.items()}

        def _is_int_key(k):
            if isinstance(k, (int, np.integer)):
                return True
            if isinstance(k, str):
                return k.lstrip("+-").isdigit()
            return False

        keys_are_periods = all(_is_int_key(k) for k in val.keys())
        out = {}
        for k, v in val.items():
            fv = _fmt_value(v, dname)
            if fv is None:
                continue
            if isinstance(fv, (list, dict)) and len(fv) == 0:
                continue
            out[int(k) if keys_are_periods else k] = fv
        return out
    if isinstance(val, (list, tuple)):
        rows = list(val)
        if rows and all(
            isinstance(r, (list, tuple)) and len(r) == 1
            and isinstance(r[0], (bool, np.bool_, str)) for r in rows
        ):
            return [
                _py_scalar(r[0]) for r in rows
                if not isinstance(r[0], (bool, np.bool_))
            ]
        return [_fmt_value(v, dname) for v in val]
    return val


# ===========================================================================
# Code generation
# ===========================================================================
def _ctor_params(cls):
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return {}
    return {_norm(p): p for p in params}


def _period_dict(ds, nper):
    raw = ds.get_data()
    if isinstance(raw, dict):
        return raw
    out = {}
    for kper in range(nper):
        try:
            v = ds.get_data(kper)
        except Exception:
            v = None
        if v is not None:
            out[kper] = v
    if not out and raw is not None:
        out[0] = raw
    return out


def _external_record(ds, keep_external):
    if not keep_external:
        return None
    try:
        storage = ds._get_storage_obj() if hasattr(ds, "_get_storage_obj") else None
    except Exception:
        storage = None
    if storage is None:
        return None
    try:
        for layer_storage in getattr(storage, "layer_storage", []) or []:
            fname = getattr(layer_storage, "fname", None)
            if fname:
                rec = {"filename": os.path.basename(str(fname))}
                factor = getattr(layer_storage, "factor", None)
                if factor not in (None, 1.0, 1):
                    rec["factor"] = _py_scalar(factor)
                return rec
    except Exception:
        return None
    return None


def _iter_set_data(package, nper=1, keep_external=False):
    for bname, blk in package.blocks.items():
        for dname, ds in blk.datasets.items():
            ext = _external_record(ds, keep_external)
            if ext is not None:
                if _norm(dname) not in _AUTO_SKIP:
                    yield dname, ext
                continue
            try:
                if bname == "period":
                    raw = _period_dict(ds, nper)
                    if not raw:
                        raw = None
                else:
                    raw = ds.get_data()
            except Exception:
                raw = None
            if raw is None:
                continue
            if _norm(dname) in _AUTO_SKIP:
                continue
            if isinstance(raw, np.recarray) and any(
                str(fn).startswith(_norm(dname)) for fn in raw.dtype.names
            ):
                fval = _flatten_keyword_record(dname, raw)
            else:
                fval = _fmt_value(raw, dname)
            if fval is None:
                continue
            if isinstance(fval, (list, dict)) and len(fval) == 0:
                if _norm(dname) not in ("newtonoptions", "xt3doptions"):
                    continue
            yield dname, fval


def _emit_package(package, varname, parent_var, lines, nper=1,
                  keep_external=False, indent="    "):
    cls = type(package)
    clsname = cls.__name__
    accepted = _ctor_params(cls)
    kwargs = []
    pname = getattr(package, "package_name", None)
    if pname and "pname" in accepted:
        kwargs.append(("pname", repr(pname)))
    for dname, fval in _iter_set_data(package, nper, keep_external):
        norm = _norm(dname)
        if norm not in accepted:
            continue
        real = accepted[norm]
        if real in ("pname", "self", "model", "simulation", "loading_package"):
            continue
        kwargs.append((real, repr(fval)))
    lines.append(f"{varname} = flopy.mf6.{clsname}(")
    lines.append(f"{indent}{parent_var},")
    for k, v in kwargs:
        lines.append(f"{indent}{k}={v},")
    lines.append(")")
    lines.append("")


def _safe_var(base, used):
    name = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(base))
    if not name or name[0].isdigit():
        name = "p_" + name
    orig, n = name, 1
    while name in used:
        name = f"{orig}_{n}"
        n += 1
    used.add(name)
    return name


def _solver_model_map(sim):
    mapping = []
    try:
        sg = sim.name_file.solutiongroup.get_data()
    except Exception:
        sg = None
    if not sg:
        return mapping
    for _grp, rec in sg.items():
        for row in rec:
            names = list(row.dtype.names)
            fname = row["slnfname"]
            model_names = [
                row[n] for n in names
                if n.startswith("slnmnames") and row[n] not in (None, "")
            ]
            mapping.append((os.path.basename(str(fname)), model_names))
    return mapping


def _has_oc(model):
    """True if the model already has an Output Control package."""
    for pkg in model.packagelist:
        if getattr(pkg, "package_type", "").lower() == "oc":
            return True
    return False


def build_lines(sim, sim_ws, add_run, add_plots, add_export, add_vtk,
                keep_external, add_postprocess, model_folder):
    lines, used = [], set()
    try:
        nper = int(sim.tdis.nper.get_data())
    except Exception:
        nper = 1

    # header
    lines += [
        '"""',
        "Auto-generated FloPy build script.",
        f"Source model: {os.path.abspath(model_folder).replace(chr(92), '/')}",
        "Generated by mf6_to_flopy.py",
        '"""',
        "import os",
        "import flopy",
    ]
    if add_plots or add_postprocess:
        lines.append("import numpy as np")
        lines.append("import matplotlib.pyplot as plt")
    lines += [
        "",
        f"sim_ws = {sim_ws!r}",
        "os.makedirs(sim_ws, exist_ok=True)",
        "",
    ]

    # simulation
    sim_name = sim.name or "sim"
    lines += [
        "# " + "-" * 66,
        "# Simulation",
        "# " + "-" * 66,
        "sim = flopy.mf6.MFSimulation(",
        f"    sim_name={sim_name!r},",
        '    version="mf6",',
        f"    exe_name={sim.exe_name!r},",
        "    sim_ws=sim_ws,",
        ")",
        "",
    ]
    used.add("sim")

    exchange_pkgs = list(getattr(sim, "exchange_files", []) or [])
    exchange_ids = {id(p) for p in exchange_pkgs}

    ims_var_by_file, emitted = {}, set()
    for pkg in sim.sim_package_list:
        if id(pkg) in exchange_ids or id(pkg) in emitted:
            continue
        emitted.add(id(pkg))
        clsname = type(pkg).__name__
        base = getattr(pkg, "package_type", clsname.lower())
        var = _safe_var(base, used)
        lines.append(f"# --- simulation package: {clsname} ---")
        _emit_package(pkg, var, "sim", lines, nper, keep_external)
        if clsname == "ModflowIms":
            fn = getattr(pkg, "filename", None)
            if fn:
                ims_var_by_file[os.path.basename(str(fn))] = var

    reg_lines = []
    for sln_file, model_names in _solver_model_map(sim):
        var = ims_var_by_file.get(sln_file)
        if var and model_names:
            names_repr = ", ".join(f"{m!r}" for m in model_names)
            reg_lines.append(f"sim.register_ims_package({var}, [{names_repr}])")
    if reg_lines:
        lines.append("# --- register solvers with their model(s) ---")
        lines += reg_lines
        lines.append("")

    # models
    model_vars = {}
    first_gwf_var = None
    for model_name in sim.model_names:
        model = sim.get_model(model_name)
        mclsname = type(model).__name__
        mvar = _safe_var(model_name, used)
        model_vars[model_name] = mvar
        if mclsname == "ModflowGwf" and first_gwf_var is None:
            first_gwf_var = mvar

        lines += [
            "# " + "-" * 66,
            f"# Model: {model_name}  ({mclsname})",
            "# " + "-" * 66,
        ]
        model_opts = []
        for _b, blk in model.name_file.blocks.items():
            for dname, ds in blk.datasets.items():
                if dname == "packages":
                    continue
                try:
                    raw = ds.get_data()
                except Exception:
                    raw = None
                if raw is None:
                    continue
                if isinstance(raw, np.recarray) and any(
                    str(fn).startswith(_norm(dname)) for fn in raw.dtype.names
                ):
                    fval = _flatten_keyword_record(dname, raw)
                else:
                    fval = _fmt_value(raw, dname)
                if fval is None:
                    continue
                model_opts.append((_norm(dname), repr(fval)))
        lines.append(f"{mvar} = flopy.mf6.{mclsname}(")
        lines.append("    sim,")
        lines.append(f"    modelname={model_name!r},")
        for k, v in model_opts:
            lines.append(f"    {k}={v},")
        lines.append(")")
        lines.append("")

        for pkg in model.packagelist:
            clsname = type(pkg).__name__
            base = getattr(pkg, "package_type", clsname.lower())
            pvar = _safe_var(base, used)
            lines.append(f"# --- {clsname} ---")
            _emit_package(pkg, pvar, mvar, lines, nper, keep_external)

        # If post-processing is requested but the model has no OC, add one so
        # that heads and budget are recorded (mirrors the standard workflow).
        if add_postprocess and mclsname == "ModflowGwf" and not _has_oc(model):
            hf = f"{model_name}.hds"
            bf = f"{model_name}.cbb"
            ocvar = _safe_var("oc", used)
            lines.append("# --- ModflowGwfoc (added for post-processing) ---")
            lines.append(f"{ocvar} = flopy.mf6.ModflowGwfoc(")
            lines.append(f"    {mvar},")
            lines.append(f"    head_filerecord={[hf]!r},")
            lines.append(f"    budget_filerecord={[bf]!r},")
            lines.append("    saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],")
            lines.append("    printrecord=[('HEAD', 'LAST')],")
            lines.append(")")
            lines.append("")

    # exchanges
    if exchange_pkgs:
        lines += ["# " + "-" * 66, "# Exchanges", "# " + "-" * 66]
        try:
            exg_rows = list(sim.name_file.exchanges.get_data())
        except Exception:
            exg_rows = []
        for i, pkg in enumerate(exchange_pkgs):
            clsname = type(pkg).__name__
            evar = _safe_var(getattr(pkg, "package_type", "exg"), used)
            exgtype = mnamea = mnameb = None
            if i < len(exg_rows):
                row = exg_rows[i]
                exgtype = str(row[0]).upper()
                mnamea, mnameb = row[2], row[3]
            lines.append(f"# --- {clsname} ---")
            lines.append(f"{evar} = flopy.mf6.{clsname}(")
            lines.append("    sim,")
            if exgtype:
                lines.append(f"    exgtype={exgtype!r},")
            if mnamea is not None:
                lines.append(f"    exgmnamea={mnamea!r},")
            if mnameb is not None:
                lines.append(f"    exgmnameb={mnameb!r},")
            accepted = _ctor_params(type(pkg))
            for dname, fval in _iter_set_data(pkg, nper, keep_external):
                norm = _norm(dname)
                if norm in accepted and norm not in (
                    "exgtype", "exgmnamea", "exgmnameb"
                ):
                    lines.append(f"    {accepted[norm]}={fval!r},")
            lines.append(")")
            lines.append("")

    # write
    lines += [
        "# " + "-" * 66,
        "# Write the model input files",
        "# " + "-" * 66,
        "sim.write_simulation()",
        "",
    ]

    # run (also implied by --postprocess)
    if add_run or add_postprocess:
        lines += [
            "# " + "-" * 66,
            "# Run the model",
            "# " + "-" * 66,
            "success, buff = sim.run_simulation()",
            "if not success:",
            "    raise Exception('MODFLOW 6 did not terminate normally.')",
            "print('Run successful:', success)",
            "",
        ]

    mvar = first_gwf_var or (model_vars[sim.model_names[0]] if sim.model_names else "gwf")

    # plots (also implied by --postprocess)
    if (add_plots or add_postprocess) and first_gwf_var:
        lines += [
            "# " + "-" * 66,
            "# Post-process: heads (map view, one PNG per layer)",
            "# " + "-" * 66,
            "try:",
            f"    hds = {mvar}.output.head()",
            "    heads = hds.get_data()  # shape (nlay, nrow, ncol)",
            "    for k in range(heads.shape[0]):",
            "        fig, ax = plt.subplots(figsize=(6, 5))",
            f"        pmv = flopy.plot.PlotMapView(model={mvar}, layer=k, ax=ax)",
            "        arr = pmv.plot_array(heads[k])",
            "        cs = pmv.contour_array(heads[k], colors='black')",
            "        ax.clabel(cs, fmt='%2.1f')",
            "        pmv.plot_grid(lw=0.3)",
            "        plt.colorbar(arr, ax=ax, shrink=0.6, label='Head')",
            "        ax.set_title(f'Head - layer {k + 1}')",
            "        fig.savefig(os.path.join(sim_ws, f'head_layer_{k + 1}.png'),",
            "                    dpi=150, bbox_inches='tight')",
            "        plt.close(fig)",
            "    print('Wrote head-contour PNG(s) to', sim_ws)",
            "except Exception as exc:",
            "    print('Head plotting skipped:', exc)",
            "",
        ]

    # specific discharge + list budget (only with --postprocess)
    if add_postprocess and first_gwf_var:
        lines += [
            "# " + "-" * 66,
            "# Post-process: specific-discharge vectors (layer 1)",
            "# " + "-" * 66,
            "try:",
            f"    cbb = {mvar}.output.budget()",
            "    spdis = cbb.get_data(text='SPDIS')[0]",
            "    qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(",
            f"        spdis, {mvar})",
            "    fig, ax = plt.subplots(figsize=(6, 5))",
            f"    pmv = flopy.plot.PlotMapView(model={mvar}, layer=0, ax=ax)",
            "    pmv.plot_array(heads[0])",
            "    pmv.plot_vector(qx, qy, normalize=False, color='blue')",
            "    pmv.plot_grid(lw=0.3)",
            "    ax.set_title('Specific discharge - layer 1')",
            "    fig.savefig(os.path.join(sim_ws, 'specific_discharge.png'),",
            "                dpi=150, bbox_inches='tight')",
            "    plt.close(fig)",
            "    print('Wrote specific_discharge.png')",
            "except Exception as exc:",
            "    print('Specific-discharge plotting skipped:', exc)",
            "",
            "# " + "-" * 66,
            "# Post-process: read the List Budget (.lst) file",
            "# " + "-" * 66,
            "try:",
            f"    lst_path = os.path.join(sim_ws, '{sim.model_names[0] if sim.model_names else 'model'}.lst')",
            "    mf_list = flopy.utils.Mf6ListBudget(lst_path)",
            "    inc_df, cum_df = mf_list.get_dataframes()",
            "    print('\\nIncremental budget (last stress period):')",
            "    print(inc_df.tail(1).to_string())",
            "    print('\\nCumulative budget (last stress period):')",
            "    print(cum_df.tail(1).to_string())",
            "    inc_df.to_csv(os.path.join(sim_ws, 'budget_incremental.csv'))",
            "    cum_df.to_csv(os.path.join(sim_ws, 'budget_cumulative.csv'))",
            "    print('Wrote budget_incremental.csv and budget_cumulative.csv')",
            "except Exception as exc:",
            "    print('List Budget reading skipped:', exc)",
            "",
        ]

    # export (shapefile + VTK)
    if add_export:
        lines += [
            "# " + "-" * 66,
            "# Export: grid shapefile" + (" + VTK" if add_vtk else ""),
            "# " + "-" * 66,
            "export_dir = os.path.join(sim_ws, 'gis')",
            "os.makedirs(export_dir, exist_ok=True)",
        ]
        for model_name in sim.model_names:
            evar = model_vars[model_name]
            lines += [
                f"# grid shapefile for model '{model_name}'",
                "try:",
                f"    gdf = {evar}.modelgrid.to_geodataframe()",
                f"    gdf.to_file(os.path.join(export_dir, '{model_name}_grid.shp'))",
                f"    print('Wrote {model_name}_grid.shp')",
                "except Exception as exc:",
                "    print('Grid shapefile export skipped:', exc)",
                "try:",
                f"    {evar}.export(os.path.join(export_dir, '{model_name}_attrs.shp'))",
                f"    print('Wrote {model_name}_attrs.shp')",
                "except Exception as exc:",
                "    print('Attribute shapefile export skipped:', exc)",
            ]
            if add_vtk:
                lines += [
                    "try:",
                    "    from flopy.export.vtk import Vtk",
                    f"    vtk = Vtk(model={evar}, binary=True, xml=False, pvd=False,",
                    "              vertical_exageration=1.0, smooth=False)",
                    f"    vtk.add_model({evar})",
                    f"    vtk.write(os.path.join(export_dir, '{model_name}_vtk'))",
                    f"    print('Wrote VTK for {model_name}')",
                    "except Exception as exc:",
                    "    print('VTK export skipped:', exc)",
                ]
        lines.append("")

    return lines, nper, sim_name, exchange_pkgs


def generate_script(model_folder, out_path, sim_ws="rebuilt", add_run=False,
                    add_plots=False, add_export=False, add_vtk=False,
                    keep_external=False, add_postprocess=False, sanitize=True):
    import flopy

    if not os.path.isdir(model_folder):
        sys.exit(f"ERROR: not a folder: {model_folder}")

    print(f"Loading MF6 simulation from: {model_folder}")
    load_dir, tmp_dir = model_folder, None
    if sanitize:
        load_dir, was = sanitize_model_folder(model_folder, verbose=True)
        if was:
            tmp_dir = load_dir
    try:
        sim = flopy.mf6.MFSimulation.load(sim_ws=load_dir, verbosity_level=0)
    except Exception as exc:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        print("\nERROR: FloPy could not load the model.")
        print(f"  {type(exc).__name__}: {str(exc)[:300]}")
        sys.exit(1)

    lines, nper, sim_name, exchange_pkgs = build_lines(
        sim, sim_ws, add_run, add_plots, add_export, add_vtk,
        keep_external, add_postprocess, model_folder,
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Wrote FloPy build script -> {out_path}")
    print(f"  simulation     : {sim_name}")
    print(f"  models         : {', '.join(sim.model_names) or '(none)'}")
    print(f"  stress periods : {nper}")
    if exchange_pkgs:
        print(f"  exchanges      : {len(exchange_pkgs)}")
    print(f"  options        : plots={add_plots}, export={add_export}, "
          f"vtk={add_vtk}, keep_external={keep_external}, "
          f"postprocess={add_postprocess}, run={add_run or add_postprocess}")
    return out_path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Translate an MF6 output folder into a readable FloPy build script."
    )
    ap.add_argument("model_folder", help="Folder containing mfsim.nam and package files")
    ap.add_argument("-o", "--out", default="build_model.py",
                    help="Output FloPy script path (default: build_model.py)")
    ap.add_argument("--sim-ws", default="rebuilt",
                    help="Workspace written into the generated script (default: rebuilt)")
    ap.add_argument("--plots", action="store_true",
                    help="Append head-contour plotting code")
    ap.add_argument("--export", action="store_true",
                    help="Append grid + attribute shapefile export code")
    ap.add_argument("--vtk", action="store_true",
                    help="With --export, also export the model to VTK (needs 'vtk')")
    ap.add_argument("--keep-external", action="store_true",
                    help="Preserve OPEN/CLOSE external file references instead of inlining")
    ap.add_argument("--run", action="store_true",
                    help="Append sim.run_simulation()")
    ap.add_argument("--postprocess", action="store_true",
                    help="Append full workflow: run, plot heads, plot specific "
                         "discharge, and read the List Budget file (implies --run/--plots)")
    ap.add_argument("--no-sanitize", action="store_true",
                    help="Do not auto-fix foreign/absolute paths in the name files")
    args = ap.parse_args()

    check_dependencies(need_plots=args.plots, need_export=args.export,
                       need_vtk=args.vtk, need_postprocess=args.postprocess)

    generate_script(
        args.model_folder, args.out, sim_ws=args.sim_ws, add_run=args.run,
        add_plots=args.plots, add_export=args.export, add_vtk=args.vtk,
        keep_external=args.keep_external, add_postprocess=args.postprocess,
        sanitize=not args.no_sanitize,
    )


if __name__ == "__main__":
    main()
