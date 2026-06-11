"""
Generate experiments6.tex: capacity_haircut slides for multiple variables.
"""
import os

base = r"C:\experiments\0906"

def read_slide_data(folder):
    p_vals = []
    data = {}
    for fname, key in [
        ("bankruptcy_rationed", "rationing"),
        ("bankruptcy_contagion", "contagion"),
        ("bad_debt", "failed"),
    ]:
        path = os.path.join(base, folder, fname + ".txt")
        vals = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    if fname == "bankruptcy_rationed":
                        p_vals.append(float(parts[0]))
                    vals.append(float(parts[1]))
        data[key] = vals
    return p_vals, data

def read_var(folder, fname):
    path = os.path.join(base, folder, fname + ".txt")
    pv, dv = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                pv.append(float(parts[0]))
                dv.append(float(parts[1]))
    return pv, dv

p12, d12 = read_slide_data("exp_min_p_0_01_capacity_haircut")
p1, d1 = read_slide_data("exp_min_p_0_1_capacity_haircut")
n12 = len(p12)
n1 = len(p1)

p1_table = ["0.00001", "0.111", "0.222", "0.333", "0.444", "0.556", "0.667", "0.778", "0.889", "1.000"]
p12_table = ["0.00001", "0.011", "0.022", "0.033", "0.044", "0.055", "0.065",
             "0.076", "0.087", "0.098", "0.109", "0.120"]

def coord_line(p_vals, data_vals, fmt):
    pts = []
    for p, v in zip(p_vals, data_vals):
        pts.append(f"({p},{v:{fmt}})")
    return " ".join(pts)

lines = []
def L(s):
    lines.append(s)

# ---- header ----
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
L(r"    scaled ticks=false,")
L(r"    xticklabel style={font=\tiny, /pgf/number format/fixed},")
L(r"    yticklabel style={font=\tiny, /pgf/number format/fixed},")
L(r"  },")
L(r"  black line/.style={")
L(r"    smooth, thick, black,")
L(r"  },")
L(r"}")
L("")
L(r"\title{Capacity with Haircut}")
L(r"\author{}")
L(r"\date{\today}")
L("")
L(r"\begin{document}")
L("")
L(r"\begin{frame}")
L(r"  \titlepage")
L(r"\end{frame}")
L("")

# ---- bankruptcy slide (multi-series) ----
L(r"\begin{frame}{Bankruptcies (capacity with haircut)}")
L(r"  \begin{columns}[T]")
L(r"    \begin{column}{0.4\textwidth}")
L(r"      \centering")
L(r"      \small $p=[0,1]$\\")
L(r"      \begin{figure}")
L(r"        \centering")
L(r"        \begin{tikzpicture}")
L(r"        \begin{axis}[")
L(r"          ylabel={Rationing + Repayment},")
L(r"          ymin=0, ymax=8,")
L(r"          xmin=0, xmax=1.0,")
L(r"          width=5cm, height=3.8cm,")
L(r"        ]")
L(f"          \\addplot[only marks, mark=square*, red] coordinates {{ {coord_line(p1, d1['rationing'], '.4f')} }};")
L(f"          \\addplot[black line] coordinates {{ {coord_line(p1, d1['rationing'], '.4f')} }};")
L(f"          \\addplot[only marks, mark=o, red] coordinates {{ {coord_line(p1, d1['failed'], '.2f')} }};")
L(f"          \\addplot[black line] coordinates {{ {coord_line(p1, d1['failed'], '.2f')} }};")
L(r"        \end{axis}")
L(r"        \begin{axis}[")
L(r"          ylabel={Contagion},")
L(r"          ymin=0, ymax=0.4,")
L(r"          xmin=0, xmax=1.0,")
L(r"          axis y line*=right, axis x line=none,")
L(r"          ylabel near ticks,")
L(r"          width=5cm, height=3.8cm,")
L(r"        ]")
L(f"          \\addplot[only marks, mark=*, red] coordinates {{ {coord_line(p1, d1['contagion'], '.4f')} }};")
L(f"          \\addplot[black line] coordinates {{ {coord_line(p1, d1['contagion'], '.4f')} }};")
L(r"        \end{axis}")
L(r"        \end{tikzpicture}")
L(r"      \end{figure}")
L(r"      \vspace{-0.3cm}")
L(r"      \tiny\renewcommand{\arraystretch}{0.85}\setlength{\tabcolsep}{2pt}")
L(r"      \centering")
L(r"      \begin{tabular}{@{}l@{\hspace{0.3cm}}l@{}}")
L(r"        \begin{tabular}{lccc}")
L(r"          \toprule")
L(r"          $p$ & Rat. & Cont. & Fail. \\")
L(r"          \midrule")
for i in range(n1):
    rat = f"{d1['rationing'][i]:.2f}"
    cont = f"{d1['contagion'][i]:.2f}"
    fail = f"{d1['failed'][i]:.2f}"
    L(f"          {p1_table[i]} & {rat} & {cont} & {fail} \\\\")
L(r"          \bottomrule")
L(r"        \end{tabular}")
L(r"        &")
L(r"        \begin{tabular}{l}")
L(r"          \textcolor{red}{$\blacksquare$} Rationing \\")
L(r"          \textcolor{red}{$\circ$} Failed \\")
L(r"          \textcolor{red}{$\bullet$} Contagion")
L(r"        \end{tabular}")
L(r"      \end{tabular}")
L(r"    \end{column}")
L(r"    \begin{column}{0.6\textwidth}")
L(r"      \centering")
L(r"      \small $p=[0,0.12]$\\")
L(r"      \begin{figure}")
L(r"        \centering")
L(r"        \begin{tikzpicture}")
L(r"        \begin{axis}[")
L(r"          ylabel={Rationing + Repayment},")
L(r"          ymin=0, ymax=8,")
L(r"          xmin=0, xmax=0.12,")
L(r"          width=7cm, height=3.8cm,")
L(r"        ]")
L(f"          \\addplot[only marks, mark=square*, red] coordinates {{ {coord_line(p12, d12['rationing'], '.4f')} }};")
L(f"          \\addplot[black line] coordinates {{ {coord_line(p12, d12['rationing'], '.4f')} }};")
L(f"          \\addplot[only marks, mark=o, red] coordinates {{ {coord_line(p12, d12['failed'], '.2f')} }};")
L(f"          \\addplot[black line] coordinates {{ {coord_line(p12, d12['failed'], '.2f')} }};")
L(r"        \end{axis}")
L(r"        \begin{axis}[")
L(r"          ylabel={Contagion},")
L(r"          ymin=0, ymax=0.4,")
L(r"          xmin=0, xmax=0.12,")
L(r"          axis y line*=right, axis x line=none,")
L(r"          ylabel near ticks,")
L(r"          width=7cm, height=3.8cm,")
L(r"        ]")
L(f"          \\addplot[only marks, mark=*, red] coordinates {{ {coord_line(p12, d12['contagion'], '.4f')} }};")
L(f"          \\addplot[black line] coordinates {{ {coord_line(p12, d12['contagion'], '.4f')} }};")
L(r"        \end{axis}")
L(r"        \end{tikzpicture}")
L(r"      \end{figure}")
L(r"      \vspace{-0.3cm}")
L(r"      \tiny\renewcommand{\arraystretch}{0.85}\setlength{\tabcolsep}{2pt}")
L(r"      \centering")
L(r"      \begin{tabular}{@{}l@{\hspace{0.3cm}}l@{}}")
L(r"        \begin{tabular}{lccc}")
L(r"          \toprule")
L(r"          $p$ & Rat. & Cont. & Fail. \\")
L(r"          \midrule")
for i in range(n12):
    rat = f"{d12['rationing'][i]:.2f}"
    cont = f"{d12['contagion'][i]:.2f}"
    fail = f"{d12['failed'][i]:.2f}"
    L(f"          {p12_table[i]} & {rat} & {cont} & {fail} \\\\")
L(r"          \bottomrule")
L(r"        \end{tabular}")
L(r"        &")
L(r"        \begin{tabular}{l}")
L(r"          \textcolor{red}{$\blacksquare$} Rationing \\")
L(r"          \textcolor{red}{$\circ$} Failed \\")
L(r"          \textcolor{red}{$\bullet$} Contagion")
L(r"        \end{tabular}")
L(r"      \end{tabular}")
L(r"    \end{column}")
L(r"  \end{columns}")
L(r"\end{frame}")
L("")

# ---- helper for single-variable slides ----
def make_slide(var_key, var_label, folder1, folder12, ymax1, ymax12, prec):
    p1v, v1 = read_var(folder1, var_key)
    p12v, v12 = read_var(folder12, var_key)
    nd1 = len(v1)
    nd12 = len(v12)
    L(r"\begin{frame}{%s (capacity with haircut)}" % var_label)
    L(r"  \begin{columns}[T]")
    # left column
    L(r"    \begin{column}{0.4\textwidth}")
    L(r"      \centering")
    L(r"      \small $p=[0,1]$\\")
    L(r"      \begin{figure}")
    L(r"        \centering")
    L(r"        \begin{tikzpicture}")
    L(r"        \begin{axis}[")
    L(r"          ylabel={%s}," % var_label)
    L(r"          ymin=0,")
    L(r"          xmin=0, xmax=1.0,")
    L(r"          width=5cm, height=3.8cm,")
    if ymax1 is not None:
        L(r"          ymax=%s," % ymax1)
    L(r"        ]")
    # Build format strings explicitly to avoid f-string in f-string issues
    fmt_coord = f".{prec}f"
    L(f"          \\addplot[only marks, mark=square*, red] coordinates {{ {coord_line(p1v, v1, fmt_coord)} }};")
    L(f"          \\addplot[black line] coordinates {{ {coord_line(p1v, v1, fmt_coord)} }};")
    L(r"        \end{axis}")
    L(r"        \end{tikzpicture}")
    L(r"      \end{figure}")
    L(r"      \vspace{-0.3cm}")
    L(r"      \tiny\renewcommand{\arraystretch}{0.85}\setlength{\tabcolsep}{2pt}")
    L(r"      \centering")
    L(r"      \begin{tabular}{lr}")
    L(r"        \toprule")
    L(r"        $p$ & Val. \\")
    L(r"        \midrule")
    for i in range(nd1):
        L(f"          {p1_table[i]} & {v1[i]:.{prec}f} \\\\")
    L(r"        \bottomrule")
    L(r"      \end{tabular}")
    L(r"    \end{column}")
    # right column
    L(r"    \begin{column}{0.6\textwidth}")
    L(r"      \centering")
    L(r"      \small $p=[0,0.12]$\\")
    L(r"      \begin{figure}")
    L(r"        \centering")
    L(r"        \begin{tikzpicture}")
    L(r"        \begin{axis}[")
    L(r"          ylabel={%s}," % var_label)
    L(r"          ymin=0,")
    L(r"          xmin=0, xmax=0.12,")
    L(r"          width=7cm, height=3.8cm,")
    if ymax12 is not None:
        L(r"          ymax=%s," % ymax12)
    L(r"        ]")
    L(f"          \\addplot[only marks, mark=square*, red] coordinates {{ {coord_line(p12v, v12, fmt_coord)} }};")
    L(f"          \\addplot[black line] coordinates {{ {coord_line(p12v, v12, fmt_coord)} }};")
    L(r"        \end{axis}")
    L(r"        \end{tikzpicture}")
    L(r"      \end{figure}")
    L(r"      \vspace{-0.3cm}")
    L(r"      \tiny\renewcommand{\arraystretch}{0.85}\setlength{\tabcolsep}{2pt}")
    L(r"      \centering")
    L(r"      \begin{tabular}{lr}")
    L(r"        \toprule")
    L(r"        $p$ & Val. \\")
    L(r"        \midrule")
    for i in range(nd12):
        L(f"          {p12_table[i]} & {v12[i]:.{prec}f} \\\\")
    L(r"        \bottomrule")
    L(r"      \end{tabular}")
    L(r"    \end{column}")
    L(r"  \end{columns}")
    L(r"\end{frame}")
    L("")

f1 = "exp_min_p_0_1_capacity_haircut"
f12 = "exp_min_p_0_01_capacity_haircut"

make_slide("loans",          "Loans",            f1, f12, "3",   "3",   2)
make_slide("bad_debt",       "Bad Debt",         f1, f12, "1",   "1",   2)
make_slide("num_loans",      "Number of Loans",  f1, f12, "15",  "15",  2)
make_slide("leverage",       "Leverage",         f1, f12, None,  None,  3)
make_slide("liquidity",      "Liquidity",        f1, f12, None,  None,  1)
make_slide("deposits",       "Deposits",         f1, f12, None,  None,  1)
make_slide("psi",            "Market Power ($\\psi$)", f1, f12, "1", "1", 3)
make_slide("ir",             "Interest Rate ($ir$)",   f1, f12, "35", "35", 1)
make_slide("prob_bankruptcy", "Prob. of Bankruptcy ($p_b$)", f1, f12, "1", "1", 3)
make_slide("equity",         "Equity",                    f1, f12, None,  None,  2)

L(r"\end{document}")

outpath = r"C:\Users\heccasan\OneDrive\doct\006.interbank\code2\doc\experiments6.tex"
with open(outpath, "w") as f:
    f.write("\n".join(lines))

print(f"Written to {outpath}, {len(lines)} lines")
