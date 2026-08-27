#!/usr/bin/env python3
"""
fix_mf6_paths.py
================

Repair a MODFLOW 6 model folder whose name files (mfsim.nam and the model
*.nam files) contain absolute or foreign paths left over from the machine where
the model was built (a common ModelMuse / GMS export artifact), e.g.

    TDIS6  C:/Users/Someone/Documents/Model/Model1.tdis

FloPy resolves such references relative to the folder you load, so a path from
another machine makes loading fail with FileNotFoundError / MFDataException.

This script rewrites every file reference in the name files down to just its
basename (Model1.tdis) whenever a file with that basename actually exists in the
folder -- so the model loads cleanly wherever it lives. Originals are backed up
to a "_path_backup" subfolder first. Use --dry-run to preview without writing.

Usage
-----
    python fix_mf6_paths.py <MODEL_FOLDER> [--dry-run]
"""

import argparse
import os
import re
import shutil
import sys


def _basename(token):
    t = token.strip().strip("'\"")
    return os.path.basename(t.replace("\\", "/"))


def _token_is_fixable(token, local_files):
    t = token.strip().strip("'\"")
    if ("/" in t) or ("\\" in t):
        return _basename(t) in local_files
    return False


def fix_folder(model_folder, dry_run=False):
    if not os.path.isdir(model_folder):
        sys.exit(f"ERROR: not a folder: {model_folder}")
    local_files = set(os.listdir(model_folder))
    nam_files = [f for f in local_files if f.lower().endswith(".nam")]
    if not nam_files:
        sys.exit("ERROR: no .nam files found -- is this an MF6 model folder?")

    changes = {}
    for nf in nam_files:
        path = os.path.join(model_folder, nf)
        file_changes = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                for tok in line.split():
                    if _token_is_fixable(tok, local_files):
                        file_changes.append((tok, _basename(tok)))
        if file_changes:
            changes[nf] = file_changes

    if not changes:
        print("No absolute/foreign path references found -- nothing to fix.")
        print("The folder should load as-is.")
        return

    print("The following name-file references will be shortened to basenames:\n")
    for nf, lst in changes.items():
        print(f"  {nf}:")
        for old, new in lst:
            print(f"      {old}")
            print(f"        ->  {new}")
    print()

    if dry_run:
        print("--dry-run: no files were modified.")
        return

    backup = os.path.join(model_folder, "_path_backup")
    os.makedirs(backup, exist_ok=True)
    for nf in changes:
        shutil.copy2(os.path.join(model_folder, nf), os.path.join(backup, nf))
    print(f"Backed up original name file(s) to: {backup}")

    for nf in changes:
        path = os.path.join(model_folder, nf)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        new_lines = []
        for line in lines:
            parts = re.split(r"(\s+)", line)
            for i, p in enumerate(parts):
                if _token_is_fixable(p, local_files):
                    parts[i] = _basename(p)
            new_lines.append("".join(parts))
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(new_lines)
        print(f"Fixed: {nf}")

    print("\nDone. The model folder should now load with FloPy / mf6_to_flopy.py.")
    print("If anything looks wrong, restore the files from _path_backup/.")


def main():
    ap = argparse.ArgumentParser(
        description="Rewrite absolute/foreign path references in MF6 name "
                    "files to plain basenames so the model loads locally."
    )
    ap.add_argument("model_folder", help="Folder containing mfsim.nam and *.nam files")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview the changes without modifying any files")
    args = ap.parse_args()
    fix_folder(args.model_folder, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
