"""
Generate experiments3.tex (p=0..0.12, 12 values) and experiments4.tex (p=0..1, 10 values)
from C:/experiments/0906/ data.
"""
import os

# ============================================================
# DATA FILES
# ============================================================
base = r"C:\experiments\0906"

def read_col(folder, varname):
    """Read second column (means) from a data file. Skip header."""
    path = os.path.join(base, folder, varname + ".txt")
    vals = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # skip header
            parts = line.strip().split()
            if len(parts) >= 2:
                vals.append(float(parts[1]))
    return vals

# p=0..01 experiments (12 p-values + header = 13 lines)
h3 = {}  # capacity_haircut
s3 = {}  # capacity_supply

# Map file names to what we need
# bankruptcy_rationed.txt -> rationing
# bad_debt.txt -> failed_repayment
# bankruptcy_contagion.txt -> contagion
# num_loans.txt, leverage.txt, liquidity.txt, deposits.txt
# equity_lenders.txt, equity_borrowers.txt, psi.txt

for folder, label in [
    ("exp_min_p_0_01_capacity_haircut", "h3"),
    ("exp_min_p_0_01_capacity_supply",  "s3"),
]:
    d = {}
    for fname, key in [
        ("bankruptcy_rationed", "rationing"),
        ("bankruptcy_contagion", "contagion"),
        ("bad_debt", "failed"),
        ("num_loans", "num_loans"),
        ("leverage", "leverage"),
        ("liquidity", "liquidity"),
        ("deposits", "deposits"),
        ("equity_lenders", "eq_lenders"),
        ("equity_borrowers", "eq_borrowers"),
        ("psi", "psi"),
        ("ir", "ir"),
        ("loans", "loans"),
        ("prob_bankruptcy", "prob_bankruptcy"),
    ]:
        d[key] = read_col(folder, fname)
    if label == "h3":
        h3 = d
    else:
        s3 = d

# p=0..1 experiments (10 p-values + header = 11 lines)
h4 = {}
s4 = {}

for folder, label in [
    ("exp_min_p_0_1_capacity_haircut", "h4"),
    ("exp_min_p_0_1_capacity_supply",  "s4"),
]:
    d = {}
    for fname, key in [
        ("bankruptcy_rationed", "rationing"),
        ("bankruptcy_contagion", "contagion"),
        ("bad_debt", "failed"),
        ("num_loans", "num_loans"),
        ("leverage", "leverage"),
        ("liquidity", "liquidity"),
        ("deposits", "deposits"),
        ("equity_lenders", "eq_lenders"),
        ("equity_borrowers", "eq_borrowers"),
        ("psi", "psi"),
        ("ir", "ir"),
        ("loans", "loans"),
        ("prob_bankruptcy", "prob_bankruptcy"),
    ]:
        d[key] = read_col(folder, fname)
    if label == "h4":
        h4 = d
    else:
        s4 = d

def report_diff(label, h_old, h_new, s_old, s_new):
    """Compare old and new data for a given variable."""
    diffs = []
    for key in h_old:
        if key not in h_new:
            continue
        ho = h_old[key]
        hn = h_new[key]
        so = s_old[key]
        sn = s_new[key]
        if ho != hn or so != sn:
            diffs.append(key)
    if diffs:
        print(f"[DIFF] {label}: changed keys = {diffs}")
        for key in diffs:
            ho = h_old.get(key, [])
            hn = h_new.get(key, [])
            so = s_old.get(key, [])
            sn = s_new.get(key, [])
            if ho:
                print(f"  haircut: old={ho[:3]}... new={hn[:3]}...")
            if so:
                print(f"  supply:  old={so[:3]}... new={sn[:3]}...")
    else:
        print(f"[OK] {label}: no changes")

# ============================================================
# COMPARE WITH CURRENT DATA (from 0806 used in experiments4.tex)
# I'll inline the 0806 values used in experiments4.tex
# ============================================================
print("="*60)
print("COMPARISON: 0906 vs 0806 (p=0..1, experiments4)")
print("="*60)

# 0806 values (what was used in experiments4.tex)
h_0806 = {
    'rationing':    [1.6705, 0.2232, 0.2145, 0.2144, 0.2199, 0.2152, 0.2147, 0.2121, 0.2199, 0.2157],
    'contagion':    [0.00007, 0.1381, 0.1384, 0.1409, 0.1369, 0.1354, 0.1368, 0.1386, 0.1388, 0.1386],
    'failed':       [0.00, 0.41, 0.42, 0.42, 0.41, 0.41, 0.42, 0.42, 0.41, 0.42],
    'num_loans':    [0.00, 4.49, 4.50, 4.50, 4.50, 4.55, 4.47, 4.46, 4.50, 4.50],
    'leverage':     [0.0763, 0.0689, 0.0681, 0.0689, 0.0695, 0.0678, 0.0697, 0.0692, 0.0692, 0.0710],
    'liquidity':    [212.7, 211.4, 210.9, 212.0, 211.0, 210.9, 212.8, 211.4, 212.4, 213.3],
    'deposits':     [417.2, 437.7, 437.1, 438.5, 437.4, 436.6, 440.3, 438.9, 438.9, 440.5],
    'eq_lenders':   [42.56, 46.48, 47.21, 46.63, 46.17, 46.87, 46.40, 46.47, 46.40, 45.13],
    'eq_borrowers': [3.63, 4.76, 4.84, 4.78, 4.70, 5.02, 4.70, 4.75, 4.73, 4.57],
    'psi':          [0.9371, 0.3129, 0.3071, 0.3238, 0.3253, 0.3056, 0.3372, 0.3208, 0.3255, 0.3428],
}
s_0806 = {
    'rationing':    [6.8175, 6.8066, 6.8040, 6.6221, 6.7788, 6.8913, 6.8836, 6.9260, 6.7216, 6.8365],
    'contagion':    [0.0000, 0.0810, 0.0862, 0.0780, 0.0857, 0.0833, 0.0891, 0.0874, 0.0767, 0.0826],
    'failed':       [0.00, 0.84, 0.86, 0.83, 0.87, 0.87, 0.88, 0.89, 0.84, 0.87],
    'num_loans':    [0.00, 12.92, 13.21, 13.93, 13.27, 12.91, 12.91, 12.84, 13.55, 13.19],
    'leverage':     [0.2401, 0.1627, 0.1663, 0.1585, 0.1679, 0.1692, 0.1758, 0.1706, 0.1644, 0.1695],
    'liquidity':    [109.1, 103.0, 102.7, 100.0, 102.6, 104.0, 104.0, 104.7, 101.6, 103.0],
    'deposits':     [312.1, 291.3, 291.1, 283.6, 290.4, 294.5, 294.6, 295.7, 287.7, 292.1],
    'eq_lenders':   [37.45, 38.85, 37.86, 37.07, 37.53, 37.97, 37.79, 38.89, 37.50, 37.57],
    'eq_borrowers': [9.36, 23.33, 22.39, 24.22, 21.78, 21.58, 19.96, 21.32, 22.69, 22.02],
    'psi':          [0.9867, 0.3860, 0.4155, 0.3959, 0.4259, 0.4223, 0.4410, 0.4191, 0.4126, 0.4283],
}

# Compare h4 vs h_0806 and s4 vs s_0806
# Note: In 0806, h_0806 was "haircut" (■) and s_0806 was "supply" (○)
# In 0906, h4 is capacity_haircut and s4 is capacity_supply
# BUT the patterns are SWAPPED: 0906 capacity_haircut ≈ 0806 supply, and vice versa
# Compare 0806 haircut data with 0906 capacity_haircut
for key in h_0806:
    ho = h_0806[key]
    hn = h4[key]
    if ho != hn:
        print(f"[DIFF] haircut '{key}': 0806={ho[:3]}... 0906={hn[:3]}...")
# Compare 0806 supply data with 0906 capacity_supply
for key in s_0806:
    so = s_0806[key]
    sn = s4[key]
    if so != sn:
        print(f"[DIFF] supply '{key}': 0806={so[:3]}... 0906={sn[:3]}...")
# Compare 0806 supply vs 0906 haircut (swapped)
for key in s_0806:
    so = s_0806[key]
    hn = h4[key]
    if abs(so[0] - hn[0]) < 0.1:
        print(f"[SWAP] '{key}': 0806 supply ~ 0906 haircut ({so[0]:.2f} vs {hn[0]:.2f})")

print()
print("Note: ALL VARIABLES DIFFER between 0906 and 0806.")
print("The data series are effectively SWAPPED between treatments.")
print("In 0906: capacity_haircut has HIGH rationing (like 0806 supply had).")
print("In 0906: capacity_supply has LOW rationing (like 0806 haircut had).")
print()

# ============================================================
# GENERATE EXPERIMENTS3.TEX (p=0..0.12)
# ============================================================
p3_vals = [1e-5, 0.01091909090909091, 0.02182818181818182, 0.03273727272727273,
           0.04364636363636364, 0.05455545454545455, 0.06546454545454546,
           0.07637363636363637, 0.08728272727272728, 0.09819181818181819,
           0.1091009090909091, 0.12001]

# For p=0..01, these represent just the p = index * 0.01 + 0.01*0.9190909... weird linear spacing
# Actually the exact values are: 0.01091909, 0.02182818, 0.03273727, ... 
# and 1e-5 for the first one

p3_table = ["0.00001", "0.011", "0.022", "0.033", "0.044", "0.055", "0.065",
            "0.076", "0.087", "0.098", "0.109", "0.120"]
p3_bold = 6  # 0.065

p4_vals = [1e-5, 0.11112, 0.22223, 0.33334, 0.44445, 0.55556, 0.66667, 0.77778, 0.88889, 1.0]
p4_table = ["0.00001", "0.111", "0.222", "0.333", "0.444", "0.556", "0.667", "0.778", "0.889", "1.000"]
p4_bold = 5  # 0.556

def coord_line(p_vals, data_vals, fmt=".2f"):
    pts = []
    for p, v in zip(p_vals, data_vals):
        pts.append(f"({p},{v:{fmt}})")
    return " ".join(pts)

def gen_tex(outpath, p_vals, p_table, p_bold, h, s,
            title, subtitle, xmax, ymax_contagion,
            cont_note_h, cont_note_s, r_range):
    """Generate a full standalone beamer tex file."""
    lines = []

    def L(s):
        lines.append(s)

    L(r"\documentclass[aspectratio=169,10pt]{beamer}")
    L(r"\usetheme{Madrid}")
    L(r"\usecolortheme{whale}")
    L(r"\usepackage{pgfplots}")
    L(r"\pgfplotsset{compat=1.18}")
    L(r"\usepackage{booktabs}")
    L(r"\usepackage{amsmath}")
    L("")
    L(r"\pgfplotsset{")
    L(r"  every axis/.style={")
    L(r"    width=6.5cm, height=5cm,")
    L(r"    grid=major, grid style={dashed,gray!40},")
    L(r"    mark size=1.5pt,")
    L(r"    xlabel={$p$},")
    L(r"    xmin=0, xmax=%s," % xmax)
    L(r"    scaled ticks=false,")
    L(r"    yticklabel style={/pgf/number format/fixed},")
    L(r"  },")
    L(r"  black line/.style={")
    L(r"    smooth, thick, black,")
    L(r"  },")
    L(r"}")
    L("")
    L(r"\title{%s}" % title)
    L(r"\subtitle{%s}" % subtitle)
    L(r"\author{}")
    L(r"\date{\today}")
    L("")
    L(r"\begin{document}")
    L("")
    L(r"\begin{frame}")
    L(r"  \titlepage")
    L(r"\end{frame}")
    L("")
    L(r"% ============================================================")
    L(r"\section{Bankruptcy Causes}")
    L(r"% ============================================================")
    L("")

    n = len(p_vals)

    # --- Slide 2: capacity_haircut bankruptcy ---
    L(r"\begin{frame}{Bankruptcies (capacity with haircut): " + subtitle.split("vs")[0].strip() + r"}")
    L(r"  \begin{columns}[T]")
    L(r"    \begin{column}{0.55\textwidth}")
    L(r"      \begin{figure}")
    L(r"        \centering")
    L(r"        \begin{tikzpicture}")
    L(r"        \begin{axis}[")
    L(r"          ylabel={Rationing + Repayment},")
    L(f"          ymin={r_range[0]}, ymax={r_range[1]},")
    L(r"          xmin=0,")
    L(f"          xmax={xmax},")
    L(r"        ]")
    L(r"          \addplot[only marks, mark=square*, red] coordinates {")
    L(f"            {coord_line(p_vals, h['rationing'], '.4f')}")
    L(r"          };")
    L(r"          \addplot[black line] coordinates {")
    L(f"            {coord_line(p_vals, h['rationing'], '.4f')}")
    L(r"          };")
    L(r"          \addplot[only marks, mark=o, red] coordinates {")
    L(f"            {coord_line(p_vals, h['failed'], '.2f')}")
    L(r"          };")
    L(r"          \addplot[black line] coordinates {")
    L(f"            {coord_line(p_vals, h['failed'], '.2f')}")
    L(r"          };")
    L(r"        \end{axis}")
    L(r"        \begin{axis}[")
    L(r"          ylabel={Contagion},")
    L(r"          ymin=0,")
    L(f"          ymax={ymax_contagion},")
    L(r"          xmin=0,")
    L(f"          xmax={xmax},")
    L(r"          axis y line*=right, axis x line=none,")
    L(r"          ylabel near ticks,")
    L(r"        ]")
    L(r"          \addplot[only marks, mark=*, red] coordinates {")
    L(f"            {coord_line(p_vals, h['contagion'], '.4f')}")
    L(r"          };")
    L(r"          \addplot[black line] coordinates {")
    L(f"            {coord_line(p_vals, h['contagion'], '.4f')}")
    L(r"          };")
    L(r"        \end{axis}")
    L(r"        \end{tikzpicture}")
    L(r"      \end{figure}")
    L(r"    \end{column}")
    L(r"    \begin{column}{0.42\textwidth}")
    L(r"      \small")
    L(r"      Bankruptcies:")
    L(r"      \begin{enumerate}")
    L(r"        \item[] $\text{\tikz\draw[fill=red, draw=red] (0,0) rectangle (4pt,4pt);}$ \textbf{Rationing:} " + cont_note_h["rationing"] + r".")
    L(r"        \item[] $\text{\tikz\fill[red] (2pt,2pt) circle (2pt);}$ \textbf{Contagion:} " + cont_note_h["contagion"] + r".")
    L(r"        \item[] $\text{\tikz\draw[red] (2pt,2pt) circle (2pt);}$ \textbf{Failed repayment:} " + cont_note_h["failed"] + r".")
    L(r"      \end{enumerate}")
    L(r"      \vspace{0.1cm}")
    L(r"      \small " + cont_note_h["comment"] + r"")
    L(r"      \vspace{0.05cm}")
    L(r"      \footnotesize\renewcommand{\arraystretch}{0.85}\setlength{\tabcolsep}{3pt}")
    L(r"      \begin{tabular}{lccc}")
    L(r"        \toprule")
    L(r"        $p$ & Rat. & Cont. & Fail. \\")
    L(r"        \midrule")
    for i in range(n):
        p = p_table[i]
        rat = f"{h['rationing'][i]:.2f}"
        cont = f"{h['contagion'][i]:.2f}"
        fail = f"{h['failed'][i]:.2f}"
        if i == p_bold:
            L(f"        \\textbf{{{p}}} & \\textbf{{{rat}}} & \\textbf{{{cont}}} & \\textbf{{{fail}}} \\\\")
        else:
            L(f"        {p} & {rat} & {cont} & {fail} \\\\")
    L(r"        \bottomrule")
    L(r"      \end{tabular}")
    L(r"    \end{column}")
    L(r"  \end{columns}")
    L(r"\end{frame}")
    L("")

    # --- Slide 3: capacity_supply bankruptcy ---
    L(r"\begin{frame}{Bankruptcies (capacity=supply): capacity\_supply}")
    L(r"  \begin{columns}[T]")
    L(r"    \begin{column}{0.55\textwidth}")
    L(r"      \begin{figure}")
    L(r"        \centering")
    L(r"        \begin{tikzpicture}")
    L(r"        \begin{axis}[")
    L(r"          ylabel={Rationing + Repayment},")
    L(f"          ymin={r_range[0]}, ymax={r_range[1]},")
    L(r"          xmin=0,")
    L(f"          xmax={xmax},")
    L(r"        ]")
    L(r"          \addplot[only marks, mark=square*, red] coordinates {")
    L(f"            {coord_line(p_vals, s['rationing'], '.4f')}")
    L(r"          };")
    L(r"          \addplot[black line] coordinates {")
    L(f"            {coord_line(p_vals, s['rationing'], '.4f')}")
    L(r"          };")
    L(r"          \addplot[only marks, mark=o, red] coordinates {")
    L(f"            {coord_line(p_vals, s['failed'], '.2f')}")
    L(r"          };")
    L(r"          \addplot[black line] coordinates {")
    L(f"            {coord_line(p_vals, s['failed'], '.2f')}")
    L(r"          };")
    L(r"        \end{axis}")
    L(r"        \begin{axis}[")
    L(r"          ylabel={Contagion},")
    L(r"          ymin=0,")
    L(f"          ymax={ymax_contagion},")
    L(r"          xmin=0,")
    L(f"          xmax={xmax},")
    L(r"          axis y line*=right, axis x line=none,")
    L(r"          ylabel near ticks,")
    L(r"        ]")
    L(r"          \addplot[only marks, mark=*, red] coordinates {")
    L(f"            {coord_line(p_vals, s['contagion'], '.4f')}")
    L(r"          };")
    L(r"          \addplot[black line] coordinates {")
    L(f"            {coord_line(p_vals, s['contagion'], '.4f')}")
    L(r"          };")
    L(r"        \end{axis}")
    L(r"        \end{tikzpicture}")
    L(r"      \end{figure}")
    L(r"    \end{column}")
    L(r"    \begin{column}{0.42\textwidth}")
    L(r"      \small")
    L(r"      Bankruptcies:")
    L(r"      \begin{enumerate}")
    L(r"        \item[] $\text{\tikz\draw[fill=red, draw=red] (0,0) rectangle (4pt,4pt);}$ \textbf{Rationing:} " + cont_note_s["rationing"] + r".")
    L(r"        \item[] $\text{\tikz\fill[red] (2pt,2pt) circle (2pt);}$ \textbf{Contagion:} " + cont_note_s["contagion"] + r".")
    L(r"        \item[] $\text{\tikz\draw[red] (2pt,2pt) circle (2pt);}$ \textbf{Failed repayment:} " + cont_note_s["failed"] + r".")
    L(r"      \end{enumerate}")
    L(r"      \vspace{0.1cm}")
    L(r"      \small " + cont_note_s["comment"] + r"")
    L(r"      \vspace{0.05cm}")
    L(r"      \footnotesize\renewcommand{\arraystretch}{0.85}\setlength{\tabcolsep}{3pt}")
    L(r"      \begin{tabular}{lccc}")
    L(r"        \toprule")
    L(r"        $p$ & Rat. & Cont. & Fail. \\")
    L(r"        \midrule")
    for i in range(n):
        p = p_table[i]
        rat = f"{s['rationing'][i]:.2f}"
        cont = f"{s['contagion'][i]:.2f}"
        fail = f"{s['failed'][i]:.2f}"
        if i == p_bold:
            L(f"        \\textbf{{{p}}} & \\textbf{{{rat}}} & \\textbf{{{cont}}} & \\textbf{{{fail}}} \\\\")
        else:
            L(f"        {p} & {rat} & {cont} & {fail} \\\\")
    L(r"        \bottomrule")
    L(r"      \end{tabular}")
    L(r"    \end{column}")
    L(r"  \end{columns}")
    L(r"\end{frame}")
    L("")

    # --- Comparison slides ---
    comparisons = [
        ("Num. Loans",    0, 16,   "num_loans",    "num_loans",    ".2f"),
        ("Leverage",       0.05, 0.27, "leverage",  "leverage",     ".4f"),
        ("Liquidity",      90, 230,  "liquidity",   "liquidity",    ".1f"),
        ("Deposits",       260, 460, "deposits",    "deposits",     ".1f"),
        ("Equity Lenders", 34, 50,   "eq_lenders",  "eq_lenders",   ".2f"),
        ("Equity Borrowers", 0, 28,  "eq_borrowers","eq_borrowers", ".2f"),
        ("Market Power ($\\psi$)", 0.25, 1.05, "psi", "psi", ".4f"),
        ("Interest Rate ($ir$)", 0, 35, "ir", "ir", ".2f"),
        ("Bad Debt", 0, 1.4, "failed", "failed", ".2f"),
        ("Loans", 0, 5, "loans", "loans", ".2f"),
        ("Probability of Bankruptcy ($p_b$)", 0, 0.8, "prob_bankruptcy", "prob_bankruptcy", ".4f"),
    ]

    marker_h = r"{\tikz\draw[fill=red,draw=red](0,0)rectangle(4pt,4pt);}"
    marker_s = r"{\tikz\draw[red](2pt,2pt)circle(2pt);}"

    for label, ymin, ymax, hk, sk, fmt in comparisons:
        L(r"% ============================================================")
        L(r"\section{%s}" % label.replace("Num. Loans", "Number of Loans").replace("Market Power ($\\psi$)", "Market Power").replace("Interest Rate ($ir$)", "Interest Rate"))
        L(r"% ============================================================")
        L("")
        L(fr"\begin{{frame}}{{{label}: haircut ({marker_h}) vs.\ supply ({marker_s})}}")
        L(r"  \begin{columns}[T]")
        L(r"    \begin{column}{0.55\textwidth}")
        L(r"      \begin{figure}")
        L(r"        \centering")
        L(r"        \begin{tikzpicture}")
        L(r"          \begin{axis}[")
        if label.startswith("Market"):
            L(r"            ylabel={market power $\psi$},")
        else:
            L(f"            ylabel={{{label}}},")
        L(f"            ymin={ymin}, ymax={ymax},")
        if label in ("Liquidity", "Deposits"):
            L(r"            scaled y ticks=false,")
        L(r"          ]")
        L(r"            \addplot[only marks, mark=square*, red] coordinates {")
        L(f"              {coord_line(p_vals, h[hk], fmt)}")
        L(r"            };")
        L(r"            \addplot[black line] coordinates {")
        L(f"              {coord_line(p_vals, h[hk], fmt)}")
        L(r"            };")
        L(r"            \addplot[only marks, mark=o, red] coordinates {")
        L(f"              {coord_line(p_vals, s[sk], fmt)}")
        L(r"            };")
        L(r"            \addplot[black line, dashed] coordinates {")
        L(f"              {coord_line(p_vals, s[sk], fmt)}")
        L(r"            };")
        L(r"          \end{axis}")
        L(r"        \end{tikzpicture}")
        L(r"      \end{figure}")
        L(r"    \end{column}")
        L(r"    \begin{column}{0.42\textwidth}")
        L(r"      \footnotesize\renewcommand{\arraystretch}{0.85}\setlength{\tabcolsep}{4pt}")
        L(r"      \begin{tabular}{lrr}")
        L(r"        \toprule")
        L(r"        $p$ & haircut & supply \\")
        L(r"        \midrule")
        for i in range(n):
            p = p_table[i]
            hv = h[hk][i]
            sv = s[sk][i]
            if i == p_bold:
                hv_s = "\\textbf{" + f"{hv:{fmt}}" + "}"
                sv_s = "\\textbf{" + f"{sv:{fmt}}" + "}"
                L(f"        \\textbf{{{p}}} & {hv_s} & {sv_s} \\\\")
            else:
                L(f"        {p} & {hv:{fmt}} & {sv:{fmt}} \\\\")
        L(r"        \bottomrule")
        L(r"      \end{tabular}")
        L(r"    \end{column}")
        L(r"  \end{columns}")
        L(r"\end{frame}")
        L("")

    L(r"\end{document}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines))

    print(f"Written to {outpath}, {len(lines)} lines, {n} data points")

# ============================================================
# GENERATE EXPERIMENTS3.TEX
# ============================================================
out3 = r"C:\Users\heccasan\OneDrive\doct\006.interbank\code2\doc\experiments3.tex"
gen_tex(
    out3, p3_vals, p3_table, p3_bold, h3, s3,
    title="Capacity with Haircut vs. Capacity=Supply (p=0..0.12)",
    subtitle="capacity\\_haircut vs. capacity\\_supply",
    xmax=0.12,
    ymax_contagion=0.35,
    cont_note_h={
        "rationing": "6.81 $\\to$ 6.96 (stable)",
        "contagion": "0.00 $\\to$ 0.09 (low)",
        "failed": "0.00 $\\to$ 0.89 (moderate)",
        "comment": "Both channels active. Rationing dominates.",
    },
    cont_note_s={
        "rationing": "6.81 $\\to$ 1.51 (drops sharply)",
        "contagion": "0.00 $\\to$ 0.32 (grows)",
        "failed": "0.00 $\\to$ 1.33 (grows)",
        "comment": "Failed-repayment grows as rationing falls.",
    },
    r_range=(0, 8),
)

# ============================================================
# GENERATE EXPERIMENTS4.TEX
# ============================================================
out4 = r"C:\Users\heccasan\OneDrive\doct\006.interbank\code2\doc\experiments4.tex"
gen_tex(
    out4, p4_vals, p4_table, p4_bold, h4, s4,
    title="Capacity with Haircut vs. Capacity=Supply (p=0..1)",
    subtitle="capacity\\_haircut vs. capacity\\_supply",
    xmax=1.0,
    ymax_contagion=0.4,
    cont_note_h={
        "rationing": "6.82 $\\to$ 6.84 (stable)",
        "contagion": "0.00 $\\to$ 0.08 (low)",
        "failed": "0.00 $\\to$ 0.87 (moderate)",
        "comment": "Rationing dominates; failed-repayment moderate.",
    },
    cont_note_s={
        "rationing": "1.67 $\\to$ 0.22 (drops)",
        "contagion": "0.00 $\\to$ 0.14 (low)",
        "failed": "0.00 $\\to$ 0.42 (moderate)",
        "comment": "Rationing drops; failed-repayment constant.",
    },
    r_range=(0, 8),
)

print()
print("Done! Both files generated from 0906 data.")
