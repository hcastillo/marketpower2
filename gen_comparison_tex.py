#!/usr/bin/env python
# coding: utf-8
"""
Generates a beamer .tex file comparing 2 or 3 experiment directories side by side.

Usage:
  python gen_comparison_tex.py --dir1 C:/experiments/0207/exp_min_p_0_01_mc500 \\
                               --dir2 C:/experiments/2306/exp_min_p_0_01_smooth_mc500_no_smooth \\
                               --output doc/experiments.tex \\
                               --label1 "$\\omega = 0.55$ (02/07)" \\
                               --label2 "$\\omega = 0.35$ (23/06)" \\
                               --title "Shock parameters" \\
                               --subtitle "Parameter $\\omega = 0.55$ vs $\\omega = 0.35$ (MC = 500)" \\
                               --table

  python gen_comparison_tex.py --dir1 C:/experiments/dirA --dir2 C:/experiments/dirB \\
                               --dir3 C:/experiments/dirC \\
                               --label1 "Config A" --label2 "Config B" --label3 "Config C" \\
                               --output doc/comparison.tex --table
"""
import argparse
import csv
import gzip
import os
import re
import sys
import xml.etree.ElementTree as ET


EXCLUDED_SUFFIXES = ("_boxplot",)
EXCLUDED_FILES = frozenset()

PNG_ORDER = [
    "asset_i", "asset_j", "bad_debt", "bankruptcies", "bankruptcies_all",
    "bankruptcy_contagion", "bankruptcy_rationed", "capacity",
    "communities", "communities_not_alone",
    "d1", "d2", "deposits", "equity", "equity_borrowers", "equity_lenders",
    "gcs", "grade_avg", "ir", "ir_max", "ir_min", "ir_weighted",
    "ir_weighted_max", "ir_weighted_min", "leverage", "liquidity", "loans",
    "num_banks", "num_loans", "num_of_rationed", "potential_lenders",
    "prob_bankruptcy", "prob_surviving", "profits", "psi", "rationing",
    "reserves", "var_D", "var_D1", "var_D2",
]


# ---------- GDT / CSV data loading ----------

def _parse_gdt(filepath):
    """Parse a results.gdt file, return (x_name, x_values, {var: [means]})"""
    with gzip.open(filepath, "rb") as f:
        tree = ET.parse(f)
    root = tree.getroot()

    var_names = []
    for var in root.find("variables"):
        var_names.append(var.attrib["name"])

    x_name = var_names[0]
    data_vars = {}
    for name in var_names[1:]:
        if not name.startswith("std_") and not name.endswith("_parent"):
            data_vars[name] = []

    obs_elems = root.find("observations")
    x_values = []
    for obs in obs_elems:
        parts = obs.text.strip().split()
        x_values.append(parts[0])
        idx = 1
        for name in var_names[1:]:
            if name.startswith("std_") or name.endswith("_parent"):
                continue
            data_vars[name].append(float(parts[idx]) if idx < len(parts) else 0.0)
            idx += 2

    return x_name, x_values, data_vars


def _parse_csv(filepath):
    """Parse a results.csv file, return (x_name, x_values, {var: [means]})"""
    with open(filepath, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    headers = lines[1].strip().split(";")
    x_name = headers[0]
    var_indices = {}
    for idx, h in enumerate(headers[1:]):
        if not h.startswith("std_") and not h.endswith("_parent"):
            var_indices[h] = (idx + 1, True)
        else:
            var_indices[h] = (idx + 1, False)

    data_vars = {v: [] for v in var_indices if var_indices[v][1]}
    x_values = []
    for line in lines[2:]:
        parts = line.strip().split(";")
        x_values.append(parts[0])
        for v, (idx, is_mean) in var_indices.items():
            if is_mean:
                data_vars[v].append(float(parts[idx]) if idx < len(parts) else 0.0)
    return x_name, x_values, data_vars


def load_data(directory):
    """Try to load results data from directory. Returns (x_name, x_values, {var: [means]}) or None."""
    gdt_path = os.path.join(directory, "results.gdt")
    if os.path.isfile(gdt_path):
        try:
            return _parse_gdt(gdt_path)
        except Exception as e:
            print(f"  Warning: could not parse {gdt_path}: {e}", file=sys.stderr)

    csv_path = os.path.join(directory, "results.csv")
    if os.path.isfile(csv_path):
        try:
            return _parse_csv(csv_path)
        except Exception as e:
            print(f"  Warning: could not parse {csv_path}: {e}", file=sys.stderr)

    return None


# ---------- PNG discovery ----------

def find_common_pngs(dirs):
    png_sets = []
    for d in dirs:
        s = {f[:-4] for f in os.listdir(d) if f.lower().endswith(".png")}
        png_sets.append(s)

    common = png_sets[0]
    for s in png_sets[1:]:
        common &= s

    result = []
    for base in sorted(common):
        if base in EXCLUDED_FILES:
            continue
        if any(base.endswith(s) for s in EXCLUDED_SUFFIXES):
            continue
        result.append(base)

    ordered = [b for b in PNG_ORDER if b in result]
    extra = sorted(set(result) - set(PNG_ORDER))
    return ordered + extra


def column_width(n):
    if n == 2:
        return "0.48"
    elif n == 3:
        return "0.31"
    elif n == 4:
        return "0.23"
    else:
        return f"{0.92 / n - 0.01:.2f}"


def fmt_val(v):
    """Format a numeric value for LaTeX table (up to 4 decimal places)."""
    if abs(v) >= 10000:
        return f"{v:.2f}"
    elif abs(v) >= 100:
        return f"{v:.3f}"
    elif abs(v) >= 1:
        return f"{v:.4f}"
    elif abs(v) >= 0.01:
        return f"{v:.4f}"
    else:
        return f"{v:.4f}"


def fmt_x(v):
    """Format x-axis value (remove redundant precision)."""
    try:
        num = float(v)
        if num == int(num):
            return str(int(num))
        return f"{num:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return v


# ---------- TeX generation ----------

def generate_tex(dirs, labels, output, title, subtitle, author, theme_name, with_table):
    n = len(dirs)
    png_bases = find_common_pngs(dirs)
    if not png_bases:
        print("No common PNG files found across all directories.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(png_bases)} common PNGs to compare across {n} directories.")

    all_data = []
    x_name = None
    if with_table:
        print("Loading results data for tables...")
        for d in dirs:
            data = load_data(d)
            all_data.append(data)
            if data is not None:
                print(f"  {d}: loaded ({len(data[2])} variables, {len(data[1])} points)")
                if x_name is None:
                    x_name = data[0]
            else:
                print(f"  {d}: no results file found (tables will be skipped for this directory)")
        if x_name is None:
            x_name = "p"
            print("  Using default x parameter name 'p'")

    col_w = column_width(n)
    lines = []
    a = lines.append

    a("\\documentclass{beamer}")
    a("")
    a("\\mode<presentation>")
    a("{")
    a(f"  \\usetheme{{{theme_name}}}")
    a("  \\definecolor{pantone2955}{RGB}{0,51,96}")
    a("  \\setbeamercolor{structure}{fg=pantone2955}")
    a("}")
    a("")
    a("\\usepackage[utf8]{inputenc}")
    a("\\usepackage[english]{babel}")
    a("\\usepackage{graphicx}")
    a("\\usepackage{booktabs}")
    if with_table:
        a("\\usepackage{longtable}")
    a("")
    a(f"\\title{{{title}}}")
    if subtitle:
        a(f"\\subtitle{{{subtitle}}}")
    a(f"\\author{{{author}}}")
    a("\\institute{Universitat Jaume I}")
    a("\\date{\\today}")
    a("")
    a("\\begin{document}")
    a("")
    a("\\begin{frame}")
    a("  \\titlepage")
    a("\\end{frame}")
    a("")

    for base in png_bases:
        a("\\begin{frame}{Comparison of " + base.replace("_", "\\_") + "}")
        a("  \\begin{columns}[T]")
        for i in range(n):
            a(f"    \\begin{{column}}{{{col_w}\\textwidth}}")
            a("      \\centering")
            a("      \\textbf{\\large " + labels[i] + "}\\\\[0.2cm]")
            a("      \\includegraphics[width=\\textwidth]{" +
              os.path.join(dirs[i], base + ".png").replace("\\", "/") + "}")

            if with_table and i < len(all_data) and all_data[i] is not None:
                _, x_vals, var_data = all_data[i]
                if base in var_data:
                    means = var_data[base]
                    a("")
                    a("      \\vspace{0.12cm}")
                    a("      \\tiny")
                    a("      \\begin{tabular}{lr}")
                    a("        \\toprule")
                    a(f"        {x_name} & {base.replace('_', '\\_')} \\\\")
                    a("        \\midrule")
                    for j in range(len(x_vals)):
                        a(f"        {fmt_x(x_vals[j])} & {fmt_val(means[j])} \\\\")
                    a("        \\bottomrule")
                    a("      \\end{tabular}")

            a("    \\end{column}")
        a("  \\end{columns}")
        a("\\end{frame}")
        a("")

    a("\\end{document}")

    outdir = os.path.dirname(output)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Generated: {os.path.abspath(output)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a beamer .tex comparing 2 or 3 experiment result directories."
    )
    parser.add_argument("--dir1", required=True, help="First experiment directory (leftmost column)")
    parser.add_argument("--dir2", required=True, help="Second experiment directory")
    parser.add_argument("--dir3", default=None, help="Third experiment directory")
    parser.add_argument("--output", "-o", required=True, help="Output .tex file path (e.g. doc/experiments.tex)")
    parser.add_argument("--label1", default="Experiment 1", help="Label for the first experiment")
    parser.add_argument("--label2", default="Experiment 2", help="Label for the second experiment")
    parser.add_argument("--label3", default=None, help="Label for the third experiment")
    parser.add_argument("--title", default="Comparison of experiments", help="Document title")
    parser.add_argument("--subtitle", default="", help="Document subtitle")
    parser.add_argument("--author", default="Héctor Castillo Andreu", help="Author name")
    parser.add_argument("--theme", default="Warsaw", help="Beamer theme name")
    parser.add_argument("--table", action="store_true", help="Add data tables below each graph")
    args = parser.parse_args()

    dirs = []
    labels = []
    for i in range(1, 4):
        d = getattr(args, f"dir{i}")
        l = getattr(args, f"label{i}")
        if d is not None:
            if not os.path.isdir(d):
                parser.error(f"Directory not found: {d}")
            dirs.append(d)
            labels.append(l if l is not None else f"Experiment {i}")

    if len(dirs) < 2:
        parser.error("At least --dir1 and --dir2 are required.")

    generate_tex(
        dirs=dirs,
        labels=labels,
        output=args.output,
        title=args.title,
        subtitle=args.subtitle,
        author=args.author,
        theme_name=args.theme,
        with_table=args.table,
    )


if __name__ == "__main__":
    main()
