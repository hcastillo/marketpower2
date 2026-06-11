"""
Generate experiments5.tex comparing omega=0.55, 0.15, 0.05 from 0306 data.
"""
import os

base = r"C:\experiments\0306"

def read_col(folder, varname):
    path = os.path.join(base, folder, varname + ".txt")
    vals = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                vals.append(float(parts[1]))
    return vals

folders = [
    ("omega=0.55", "exp_min_p_0_01_omega_0_55", "square*", "black line"),
    ("omega=0.15", "exp_min_p_0_01_omage_0_15", "o", "black line, dashed"),
    ("omega=0.05", "exp_min_p_0_01_omega_0_05", "diamond*", "black line, dotted"),
]

wanted = [
    "bankruptcy_rationed", "bankruptcy_contagion", "bad_debt",
    "num_loans", "leverage", "liquidity", "deposits", "psi",
    "ir", "loans", "prob_bankruptcy",
]

data = {}
styles = {}
for label, folder, mark, line in folders:
    styles[label] = (mark, line)
    d = {}
    for var in wanted:
        d[var] = read_col(folder, var)
    data[label] = d

p_vals = [1e-5, 0.01091909090909091, 0.02182818181818182, 0.03273727272727273,
          0.04364636363636364, 0.05455545454545455, 0.06546454545454546,
          0.07637363636363637, 0.08728272727272728, 0.09819181818181819,
          0.1091009090909091, 0.12001]

p_table = ["0.00001", "0.011", "0.022", "0.033", "0.044", "0.055", "0.065",
           "0.076", "0.087", "0.098", "0.109", "0.120"]
p_bold = 6

n = len(p_vals)

comparisons = [
    ("Rationing", "bankruptcy_rationed", 0, 17, ".4f"),
    ("Contagion", "bankruptcy_contagion", 0, 1.1, ".4f"),
    ("Bad Debt", "bad_debt", 0, 3.5, ".2f"),
    ("Num. Loans", "num_loans", 0, 17, ".2f"),
    ("Leverage", "leverage", 0.05, 0.55, ".4f"),
    ("Liquidity", "liquidity", 90, 220, ".1f"),
    ("Deposits", "deposits", 260, 460, ".1f"),
    ("Market Power ($\\psi$)", "psi", 0.25, 1.05, ".4f"),
    ("Interest Rate ($ir$)", "ir", 0, 35, ".2f"),
    ("Loans", "loans", 0, 7.5, ".2f"),
    ("Probability of Bankruptcy ($p_b$)", "prob_bankruptcy", 0, 0.7, ".4f"),
]

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
L(r"    xmin=0, xmax=0.12,")
L(r"    scaled ticks=false,")
L(r"    yticklabel style={/pgf/number format/fixed},")
L(r"    legend style={at={(0.5,1.03)}, anchor=south, font=\tiny},")
L(r"  },")
L(r"  black line/.style={")
L(r"    smooth, thick, black,")
L(r"  },")
L(r"}")
L("")
L(r"\title{Comparison of Shock Variances ($\omega$)}")
L(r"\subtitle{$\omega=0.55$ vs.\ $\omega=0.15$ vs.\ $\omega=0.05$}")
L(r"\author{}")
L(r"\date{\today}")
L("")
L(r"\begin{document}")
L("")
L(r"\begin{frame}")
L(r"  \titlepage")
L(r"\end{frame}")
L("")

for label, desc, ymin, ymax, fmt in comparisons:
    section_label = label.replace("Num. Loans", "Number of Loans").replace("Market Power ($\\psi$)", "Market Power").replace("Interest Rate ($ir$)", "Interest Rate").replace("Probability of Bankruptcy ($p_b$)", "Probability of Bankruptcy")
    L(r"% ============================================================")
    L(r"\section{%s}" % section_label)
    L(r"% ============================================================")
    L("")
    L(fr"\begin{{frame}}{{{label}: $\omega$ comparison}}")
    L(r"  \begin{columns}[T]")
    L(r"    \begin{column}{0.55\textwidth}")
    L(r"      \begin{figure}")
    L(r"        \centering")
    L(r"        \begin{tikzpicture}")
    L(r"          \begin{axis}[")
    L(f"            ylabel={{{label}}},")
    L(f"            ymin={ymin}, ymax={ymax},")
    L(r"          ]")
    for lname in ["omega=0.55", "omega=0.15", "omega=0.05"]:
        mark, line = styles[lname]
        vals = data[lname][desc]
        L(f"            \\addplot[only marks, mark={mark}, red] coordinates {{")
        pts = " ".join(f"({p},{v:{fmt}})" for p, v in zip(p_vals, vals))
        L(f"              {pts}")
        L(r"            };")
        L(f"            \\addplot[{line}] coordinates {{")
        L(f"              {pts}")
        L(r"            };")
        L(fr"            \addlegendentry{{{lname}}}")
    L(r"          \end{axis}")
    L(r"        \end{tikzpicture}")
    L(r"      \end{figure}")
    L(r"    \end{column}")
    L(r"    \begin{column}{0.42\textwidth}")
    L(r"      \footnotesize\renewcommand{\arraystretch}{0.85}\setlength{\tabcolsep}{3pt}")
    L(r"      \begin{tabular}{lrrr}")
    L(r"        \toprule")
    L(r"        $p$ & $\omega=0.55$ & $\omega=0.15$ & $\omega=0.05$ \\")
    L(r"        \midrule")
    for i in range(n):
        p = p_table[i]
        v55 = data["omega=0.55"][desc][i]
        v15 = data["omega=0.15"][desc][i]
        v05 = data["omega=0.05"][desc][i]
        if i == p_bold:
            L(f"        \\textbf{{{p}}} & \\textbf{{{v55:{fmt}}}} & \\textbf{{{v15:{fmt}}}} & \\textbf{{{v05:{fmt}}}} \\\\")
        else:
            L(f"        {p} & {v55:{fmt}} & {v15:{fmt}} & {v05:{fmt}} \\\\")
    L(r"        \bottomrule")
    L(r"      \end{tabular}")
    L(r"    \end{column}")
    L(r"  \end{columns}")
    L(r"\end{frame}")
    L("")

L(r"\end{document}")

outpath = r"C:\Users\heccasan\OneDrive\doct\006.interbank\code2\doc\experiments5.tex"
with open(outpath, "w") as f:
    f.write("\n".join(lines))

print(f"Written to {outpath}, {len(lines)} lines, {n} data points")
