#!/usr/bin/env python
# coding: utf-8
"""
Generate a beamer .tex file that compares the decomposition of bankruptcies
across experiments as p evolves.

For each experiment directory, the script expects results data containing:
  - bankruptcy_rationed
  - bankruptcy_contagion
  - bankruptcy_borrowers

The figure stacks these components as cumulative flow areas over p.
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import xml.etree.ElementTree as ET


def _parse_gdt(filepath):
    with gzip.open(filepath, "rb") as f:
        tree = ET.parse(f)
    root = tree.getroot()

    var_names = [var.attrib["name"] for var in root.find("variables")]
    x_name = var_names[0]
    data_vars = {name: [] for name in var_names[1:] if not name.startswith("std_") and not name.endswith("_parent")}

    x_values = []
    for obs in root.find("observations"):
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
    for filename in ("results.gdt", "results.csv"):
        path = os.path.join(directory, filename)
        if os.path.isfile(path):
            try:
                return _parse_gdt(path) if filename.endswith(".gdt") else _parse_csv(path)
            except Exception as e:
                print(f"  Warning: could not parse {path}: {e}", file=sys.stderr)
    return None


def fmt_x(v):
    try:
        num = float(v)
        if num == int(num):
            return str(int(num))
        return f"{num:.4f}".rstrip("0").rstrip(".")
    except ValueError:
        return v


def generate_tex(dirs, labels, output, title, subtitle, author, theme_name):
    all_data = []
    x_name = None
    for d in dirs:
        data = load_data(d)
        all_data.append(data)
        if data is not None and x_name is None:
            x_name = data[0]
    if x_name is None:
        x_name = "p"

    lines = []
    a = lines.append
    a("\\documentclass{beamer}")
    a("\\mode<presentation>")
    a("{")
    a(f"  \\usetheme{{{theme_name}}}")
    a("  \\definecolor{pantone2955}{RGB}{0,51,96}")
    a("  \\setbeamercolor{structure}{fg=pantone2955}")
    a("}")
    a("\\usepackage[utf8]{inputenc}")
    a("\\usepackage[english]{babel}")
    a("\\usepackage{graphicx}")
    a("\\usepackage{booktabs}")
    a("\\usepackage{pgfplots}")
    a("\\pgfplotsset{compat=1.18}")
    a("\\title{" + title + "}")
    if subtitle:
        a("\\subtitle{" + subtitle + "}")
    a(f"\\author{{{author}}}")
    a("\\institute{Universitat Jaume I}")
    a("\\date{\\today}")
    a("\\begin{document}")
    a("\\begin{frame}")
    a("  \\titlepage")
    a("\\end{frame}")
    a("")

    # One page per experiment, with stacked flow areas.
    for i, d in enumerate(dirs):
        label = labels[i]
        data = all_data[i]
        a(f"\\begin{{frame}}{{Bankruptcy decomposition: {label}}}")
        if data is None:
            a("\\centering No data found in this directory.")
        else:
            _, x_vals, var_data = data
            rat = var_data.get("bankruptcy_rationed", [])
            cont = var_data.get("bankruptcy_contagion", [])
            borr = var_data.get("bankruptcy_borrowers", [])
            if not (rat and cont and borr):
                a("\\centering Missing required variables in results file.")
            else:
                a("\\begin{tikzpicture}")
                a("\\begin{axis}[width=\\textwidth,height=0.7\\textheight, xlabel={$p$}, ylabel={Bankruptcies}, legend style={at={(0.5,-0.18)},anchor=north,legend columns=3}, ymin=0, area style, stack plots=y, enlarge x limits=false]")
                a("\\addplot+[fill=blue!45] coordinates {")
                for x, y in zip(x_vals, rat):
                    a(f"({fmt_x(x)},{y})")
                a("};\\addlegendentry{rationing}")
                a("\\addplot+[fill=orange!55] coordinates {")
                for x, y in zip(x_vals, cont):
                    a(f"({fmt_x(x)},{y})")
                a("};\\addlegendentry{contagion}")
                a("\\addplot+[fill=green!55] coordinates {")
                for x, y in zip(x_vals, borr):
                    a(f"({fmt_x(x)},{y})")
                a("};\\addlegendentry{borrowers}")
                a("\\end{axis}")
                a("\\end{tikzpicture}")
        a("\\end{frame}")

    a("\\end{document}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Written to {output}, {len(lines)} lines")


def main():
    parser = argparse.ArgumentParser(description="Generate bankruptcy decomposition comparison tex")
    parser.add_argument("--dir", dest="dirs", action="append", required=True, help="Experiment directory (repeatable)")
    parser.add_argument("--label", dest="labels", action="append", required=True, help="Label for each directory (repeatable)")
    parser.add_argument("--output", required=True, help="Output .tex file")
    parser.add_argument("--title", default="Bankruptcy decomposition", help="Document title")
    parser.add_argument("--subtitle", default="", help="Document subtitle")
    parser.add_argument("--author", default="OpenCode", help="Author")
    parser.add_argument("--theme", default="default", help="Beamer theme")
    args = parser.parse_args()

    if len(args.dirs) != len(args.labels):
        parser.error("You must provide the same number of --dir and --label arguments")

    generate_tex(args.dirs, args.labels, args.output, args.title, args.subtitle, args.author, args.theme)


if __name__ == "__main__":
    main()
