import math

# ============================================================
# DATA: 10 p-values
# ============================================================
p_coords = [0.00001, 0.11112, 0.22223, 0.33334, 0.44445, 0.55556, 0.66667, 0.77778, 0.88889, 1.00000]
p_table  = ["0.00001", "0.111", "0.222", "0.333", "0.444", "0.556", "0.667", "0.778", "0.889", "1.000"]
p_bold  = 5  # index of bold row (0.556)

# Haircut data (from C:\experiments\0806\exp_min_p_0_1_capacity_haircut)
h = {}
h['rationing']     = [1.6705, 0.2232, 0.2145, 0.2144, 0.2199, 0.2152, 0.2147, 0.2121, 0.2199, 0.2157]
h['contagion']     = [0.00007, 0.1381, 0.1384, 0.1409, 0.1369, 0.1354, 0.1368, 0.1386, 0.1388, 0.1386]
h['failed']        = [0.00, 1.83, 1.84, 1.84, 1.84, 1.84, 1.86, 1.86, 1.84, 1.85]
h['num_loans']     = [0.00, 4.49, 4.50, 4.50, 4.50, 4.55, 4.47, 4.46, 4.50, 4.50]
h['leverage']      = [0.0763, 0.0689, 0.0681, 0.0689, 0.0695, 0.0678, 0.0697, 0.0692, 0.0692, 0.0710]
h['liquidity']     = [212.7, 211.4, 210.9, 212.0, 211.0, 210.9, 212.8, 211.4, 212.4, 213.3]
h['deposits']      = [417.2, 437.7, 437.1, 438.5, 437.4, 436.6, 440.3, 438.9, 438.9, 440.5]
h['eq_lenders']    = [42.56, 46.48, 47.21, 46.63, 46.17, 46.87, 46.40, 46.47, 46.40, 45.13]
h['eq_borrowers']  = [3.63, 4.76, 4.84, 4.78, 4.70, 5.02, 4.70, 4.75, 4.73, 4.57]
h['psi']           = [0.9371, 0.3129, 0.3071, 0.3238, 0.3253, 0.3056, 0.3372, 0.3208, 0.3255, 0.3428]

# Supply data (from C:\experiments\0806\exp_min_p_0_1_capacity_supply)
s = {}
s['rationing']     = [6.8175, 6.8066, 6.8040, 6.6221, 6.7788, 6.8913, 6.8836, 6.9260, 6.7216, 6.8365]
s['contagion']     = [0.0000, 0.0810, 0.0862, 0.0780, 0.0857, 0.0833, 0.0891, 0.0874, 0.0767, 0.0826]
s['failed']        = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
s['num_loans']     = [0.00, 12.92, 13.21, 13.93, 13.27, 12.91, 12.91, 12.84, 13.55, 13.19]
s['leverage']      = [0.2401, 0.1627, 0.1663, 0.1585, 0.1679, 0.1692, 0.1758, 0.1706, 0.1644, 0.1695]
s['liquidity']     = [109.1, 103.0, 102.7, 100.0, 102.6, 104.0, 104.0, 104.7, 101.6, 103.0]
s['deposits']      = [312.1, 291.3, 291.1, 283.6, 290.4, 294.5, 294.6, 295.7, 287.7, 292.1]
s['eq_lenders']    = [37.45, 38.85, 37.86, 37.07, 37.53, 37.97, 37.79, 38.89, 37.50, 37.57]
s['eq_borrowers']  = [9.36, 23.33, 22.39, 24.22, 21.78, 21.58, 19.96, 21.32, 22.69, 22.02]
s['psi']           = [0.9867, 0.3860, 0.4155, 0.3959, 0.4259, 0.4223, 0.4410, 0.4191, 0.4126, 0.4283]

def coord_line(values, fmt=".2f"):
    """Generate a pgfplots coordinate line from p_coords and values."""
    pts = []
    for p, v in zip(p_coords, values):
        pts.append(f"({p:.5f},{v:{fmt}})")
    return "            ".join(pts)

def table_rows(values, fmt=".2f", bold_row=None):
    """Generate table rows from p_table and values."""
    rows = []
    for i, (p, v) in enumerate(zip(p_table, values)):
        if i == p_bold:
            rows.append(f"        \\textbf{{{p}}} & \\textbf{{{v:{fmt}}}}")
        else:
            rows.append(f"        {p} & {v:{fmt}}")
    return rows

def table_hs(h_values, s_values, fmt=".2f", bold_row=None):
    """Generate table rows with two data columns."""
    rows = []
    for i, (p, hv, sv) in enumerate(zip(p_table, h_values, s_values)):
        if i == p_bold:
            rows.append(f"        \\textbf{{{p}}} & \\textbf{{{hv:{fmt}}}} & \\textbf{{{sv:{fmt}}}} \\\\")
        else:
            rows.append(f"        {p} & {hv:{fmt}} & {sv:{fmt}} \\\\")
    return rows

# ============================================================
# GENERATE TEX
# ============================================================
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
L(r"    xmin=0, xmax=1.0,")
L(r"    xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},")
L(r"  },")
L(r"  black line/.style={")
L(r"    smooth, thick, black,")
L(r"  },")
L(r"}")
L("")
L(r"\title{Capacity with Haircut vs.\ Capacity=Supply (p=0..1)}")
L(r"\subtitle{exp\_min\_p\_0\_1\_capacity\_haircut vs.\ exp\_min\_p\_0\_1\_capacity\_supply}")
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

# ==================== SLIDE 2: haircut bankruptcy ====================
L(r"\begin{frame}{Bankruptcies (capacity with haircut): exp\_min\_p\_0\_1\_capacity\_haircut}")
L(r"  \begin{columns}[T]")
L(r"    \begin{column}{0.55\textwidth}")
L(r"      \begin{figure}")
L(r"        \centering")
L(r"        \begin{tikzpicture}")
L(r"        \begin{axis}[")
L(r"          ylabel={Rationing + Repayment},")
L(r"          ymin=0, ymax=8,")
L(r"          xmin=0, xmax=1.0,")
L(r"          xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},")
L(r"        ]")
L(r"          \addplot[only marks, mark=square*, red] coordinates {")
L(f"            {coord_line(h['rationing'], '.4f')}")
L(r"          };")
L(r"          \addplot[black line] coordinates {")
L(f"            {coord_line(h['rationing'], '.4f')}")
L(r"          };")
L(r"          \addplot[only marks, mark=o, red] coordinates {")
L(f"            {coord_line(h['failed'], '.2f')}")
L(r"          };")
L(r"          \addplot[black line] coordinates {")
L(f"            {coord_line(h['failed'], '.2f')}")
L(r"          };")
L(r"        \end{axis}")
L(r"        \begin{axis}[")
L(r"          ylabel={Contagion},")
L(r"          ymin=0, ymax=0.4,")
L(r"          xmin=0, xmax=1.0,")
L(r"          axis y line*=right, axis x line=none,")
L(r"          ylabel near ticks,")
L(r"          yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},")
L(r"        ]")
L(r"          \addplot[only marks, mark=*, red] coordinates {")
L(f"            {coord_line(h['contagion'], '.4f')}")
L(r"          };")
L(r"          \addplot[black line] coordinates {")
L(f"            {coord_line(h['contagion'], '.4f')}")
L(r"          };")
L(r"        \end{axis}")
L(r"        \end{tikzpicture}")
L(r"      \end{figure}")
L(r"    \end{column}")
L(r"    \begin{column}{0.42\textwidth}")
L(r"      \small")
L(r"      Bankruptcies:")
L(r"      \begin{enumerate}")
L(r"        \item[] $\text{\tikz\draw[fill=red, draw=red] (0,0) rectangle (4pt,4pt);}$ \textbf{Rationing:} 1.67 $\to$ 0.22 (drops).")
L(r"        \item[] $\text{\tikz\fill[red] (2pt,2pt) circle (2pt);}$ \textbf{Contagion:} 0.00 $\to$ 0.14 (low).")
L(r"        \item[] $\text{\tikz\draw[red] (2pt,2pt) circle (2pt);}$ \textbf{Failed repayment:} 0.00 $\to$ 1.85 (grows).")
L(r"      \end{enumerate}")
L(r"      \vspace{0.2cm}")
L(r"      \small Failed-repayment activates as connectivity grows.")
L(r"      \vspace{0.2cm}")
L(r"      \begin{tabular}{lccc}")
L(r"        \toprule")
L(r"        $p$ & Rat. & Cont. & Fail. \\")
L(r"        \midrule")
for i in range(10):
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

# ==================== SLIDE 3: supply bankruptcy ====================
L(r"\begin{frame}{Bankruptcies (capacity=supply): exp\_min\_p\_0\_1\_capacity\_supply}")
L(r"  \begin{columns}[T]")
L(r"    \begin{column}{0.55\textwidth}")
L(r"      \begin{figure}")
L(r"        \centering")
L(r"        \begin{tikzpicture}")
L(r"        \begin{axis}[")
L(r"          ylabel={Rationing + Repayment},")
L(r"          ymin=0, ymax=8,")
L(r"          xmin=0, xmax=1.0,")
L(r"          xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},")
L(r"        ]")
L(r"          \addplot[only marks, mark=square*, red] coordinates {")
L(f"            {coord_line(s['rationing'], '.4f')}")
L(r"          };")
L(r"          \addplot[black line] coordinates {")
L(f"            {coord_line(s['rationing'], '.4f')}")
L(r"          };")
L(r"          \addplot[only marks, mark=o, red] coordinates {")
L(f"            {coord_line(s['failed'], '.2f')}")
L(r"          };")
L(r"          \addplot[black line] coordinates {")
L(f"            {coord_line(s['failed'], '.2f')}")
L(r"          };")
L(r"        \end{axis}")
L(r"        \begin{axis}[")
L(r"          ylabel={Contagion},")
L(r"          ymin=0, ymax=0.4,")
L(r"          xmin=0, xmax=1.0,")
L(r"          axis y line*=right, axis x line=none,")
L(r"          ylabel near ticks,")
L(r"          yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},")
L(r"        ]")
L(r"          \addplot[only marks, mark=*, red] coordinates {")
L(f"            {coord_line(s['contagion'], '.4f')}")
L(r"          };")
L(r"          \addplot[black line] coordinates {")
L(f"            {coord_line(s['contagion'], '.4f')}")
L(r"          };")
L(r"        \end{axis}")
L(r"        \end{tikzpicture}")
L(r"      \end{figure}")
L(r"    \end{column}")
L(r"    \begin{column}{0.42\textwidth}")
L(r"      \small")
L(r"      Bankruptcies:")
L(r"      \begin{enumerate}")
L(r"        \item[] $\text{\tikz\draw[fill=red, draw=red] (0,0) rectangle (4pt,4pt);}$ \textbf{Rationing:} 6.82 $\to$ 6.84 (stable).")
L(r"        \item[] $\text{\tikz\fill[red] (2pt,2pt) circle (2pt);}$ \textbf{Contagion:} 0.00 $\to$ 0.08 (low).")
L(r"        \item[] $\text{\tikz\draw[red] (2pt,2pt) circle (2pt);}$ \textbf{Failed repayment:} 0.00 (null).")
L(r"      \end{enumerate}")
L(r"      \vspace{0.2cm}")
L(r"      \small No failed-repayment channel.")
L(r"      \vspace{0.2cm}")
L(r"      \begin{tabular}{lccc}")
L(r"        \toprule")
L(r"        $p$ & Rat. & Cont. & Fail. \\")
L(r"        \midrule")
for i in range(10):
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

# ==================== COMPARISON SLIDES ====================
# For each variable: (label, ymin, ymax, h_key, s_key, fmt)
comparisons = [
    ("Num. Loans",    0, 16,   "num_loans",    "num_loans",    ".2f"),
    ("Leverage",       0.05, 0.27, "leverage",  "leverage",     ".4f"),
    ("Liquidity",      90, 230,  "liquidity",   "liquidity",    ".1f"),
    ("Deposits",       260, 460, "deposits",    "deposits",     ".1f"),
    ("Equity Lenders", 34, 50,   "eq_lenders",  "eq_lenders",   ".2f"),
    ("Equity Borrowers", 0, 28,  "eq_borrowers","eq_borrowers", ".2f"),
    ("Market Power ($\\psi$)", 0.25, 1.05, "psi", "psi", ".4f"),
]

slide_num = 4
for label, ymin, ymax, hk, sk, fmt in comparisons:
    L(r"% ============================================================")
    if label.startswith("Market"):
        L(r"\section{Market Power}")
    else:
        L(fr"\section{{{label}}}")
    L(r"% ============================================================")
    L("")
    marker_h = r"{\tikz\draw[fill=red,draw=red](0,0)rectangle(4pt,4pt);}"
    marker_s = r"{\tikz\draw[red](2pt,2pt)circle(2pt);}"
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
    # ■ = haircut (solid)
    L(r"            \addplot[only marks, mark=square*, red] coordinates {")
    L(f"              {coord_line(h[hk], fmt)}")
    L(r"            };")
    L(r"            \addplot[black line] coordinates {")
    L(f"              {coord_line(h[hk], fmt)}")
    L(r"            };")
    # ○ = supply (dashed)
    L(r"            \addplot[only marks, mark=o, red] coordinates {")
    L(f"              {coord_line(s[sk], fmt)}")
    L(r"            };")
    L(r"            \addplot[black line, dashed] coordinates {")
    L(f"              {coord_line(s[sk], fmt)}")
    L(r"            };")
    L(r"          \end{axis}")
    L(r"        \end{tikzpicture}")
    L(r"      \end{figure}")
    L(r"    \end{column}")
    L(r"    \begin{column}{0.42\textwidth}")
    L(r"      \small")
    L(r"      \begin{tabular}{lrr}")
    L(r"        \toprule")
    L(r"        $p$ & haircut & supply \\")
    L(r"        \midrule")
    for i in range(10):
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

# Write
outpath = r"C:\Users\heccasan\OneDrive\doct\006.interbank\code2\doc\experiments4.tex"
with open(outpath, "w") as f:
    f.write("\n".join(lines))
print(f"Written to {outpath}")
print(f"Total lines: {len(lines)}")
