r"""
Generate experiments3.tex (p=0..0.12) and experiments4.tex (p=0..1)
from C:\experiments\0906 data.
"""
import os

base = r"C:\experiments\0906"

p_table_12 = ["0.00001", "0.011", "0.022", "0.033", "0.044", "0.055", "0.065",
              "0.076", "0.087", "0.098", "0.109", "0.120"]
p_coords_12 = [1e-5, 0.011, 0.022, 0.033, 0.044, 0.055, 0.065,
               0.076, 0.087, 0.098, 0.109, 0.120]

p_table_1 = ["0.00001", "0.111", "0.222", "0.333", "0.444", "0.556", "0.667", "0.778", "0.889", "1.000"]
p_coords_1 = [1e-5, 0.11112, 0.22223, 0.33334, 0.44445, 0.55556, 0.66667, 0.77778, 0.88889, 1.0]

p_bold = 5  # bold row index

FILES = [
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
]

def read_all(folder):
    p_vals = None
    data = {}
    for fname, key in FILES:
        path = os.path.join(base, folder, fname + ".txt")
        vals = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    if p_vals is None:
                        p_vals = []
                    if i - 1 == len(p_vals):
                        p_vals.append(float(parts[0]))
                    vals.append(float(parts[1]))
        data[key] = vals
    return p_vals, data

# p=[0,0.12] data
p12_hair, d12_hair = read_all("exp_min_p_0_01_capacity_haircut")
p12_supp, d12_supp = read_all("exp_min_p_0_01_capacity_supply")
n12 = len(p12_hair)

# p=[0,1] data
p1_hair, d1_hair = read_all("exp_min_p_0_1_capacity_haircut")
p1_supp, d1_supp = read_all("exp_min_p_0_1_capacity_supply")
n1 = len(p1_hair)

def coord_line(p_vals, values, fmt):
    pts = []
    for p, v in zip(p_vals, values):
        pts.append(f"({p:.5f},{v:{fmt}})")
    return "            ".join(pts)

def table_rows_hs(p_table, h_vals, s_vals, fmt, bold=None):
    rows = []
    for i, (p, hv, sv) in enumerate(zip(p_table, h_vals, s_vals)):
        if bold is not None and i == bold:
            rows.append(f"        \\textbf{{{p}}} & \\textbf{{{hv:{fmt}}}} & \\textbf{{{sv:{fmt}}}} \\\\")
        else:
            rows.append(f"        {p} & {hv:{fmt}} & {sv:{fmt}} \\\\")
    return rows

def table_rows3(p_table, h_vals, s_vals, f_vals, fmt_hs=".2f", fmt_f=".2f", bold=None):
    rows = []
    for i, (p, hv, sv, fv) in enumerate(zip(p_table, h_vals, s_vals, f_vals)):
        if bold is not None and i == bold:
            rows.append(f"        \\textbf{{{p}}} & \\textbf{{{hv:{fmt_hs}}}} & \\textbf{{{sv:{fmt_hs}}}} & \\textbf{{{fv:{fmt_f}}}} \\\\")
        else:
            rows.append(f"        {p} & {hv:{fmt_hs}} & {sv:{fmt_hs}} & {fv:{fmt_f}} \\\\")
    return rows

# =====================================================================
# Comparison variable definitions: (key, label, ymin, ymax, fmt)
# =====================================================================
comparisons = [
    ("num_loans",    "Num. Loans",              0, 16,   ".2f"),
    ("leverage",     "Leverage",               0.05,0.27, ".4f"),
    ("liquidity",    "Liquidity",               90, 230,  ".1f"),
    ("deposits",     "Deposits",                260,460,  ".1f"),
    ("eq_lenders",   "Equity Lenders",          34, 50,   ".2f"),
    ("eq_borrowers", "Equity Borrowers",         0, 28,   ".2f"),
    ("psi",          "Market Power ($\\psi$)", 0.25,1.05, ".4f"),
    ("ir",           "Interest Rate ($ir$)",     0, 35,   ".1f"),
    ("failed",       "Bad Debt",                 0, 1.2,  ".2f"),
    ("loans",        "Loans",                    0, 5,    ".2f"),
    ("prob_bankruptcy", "Prob. of Bankruptcy ($p_b$)", 0, 0.8, ".3f"),
]

scaled_vars = {"liquidity", "deposits"}

# =====================================================================
def make_comparison_slide(L, label, key, p_table, p_coords, h_data, s_data,
                          ymin, ymax, fmt, xmax, bold):
    """Generate one comparison slide."""
    h_vals = h_data[key]
    s_vals = s_data[key]

    if "Market" in label:
        ylabel = "market power $\\psi$"
    else:
        ylabel = label

    L(r"\begin{frame}{" + label + r": haircut ($\blacksquare$) vs.\ supply ($\circ$)}")
    L(r"  \begin{columns}[T]")
    L(r"    \begin{column}{0.55\textwidth}")
    L(r"      \begin{figure}")
    L(r"        \centering")
    L(r"        \begin{tikzpicture}")
    L(r"          \begin{axis}[")
    L(f"            ylabel={{{ylabel}}},")
    L(f"            ymin={ymin}, ymax={ymax},")
    if key in scaled_vars:
        L(r"            scaled y ticks=false,")
    L(r"          ]")
    L(r"            \addplot[only marks, mark=square*, red] coordinates {")
    L(f"              {coord_line(p_coords, h_vals, fmt)}")
    L(r"            };")
    L(r"            \addplot[black line] coordinates {")
    L(f"              {coord_line(p_coords, h_vals, fmt)}")
    L(r"            };")
    L(r"            \addplot[only marks, mark=o, red] coordinates {")
    L(f"              {coord_line(p_coords, s_vals, fmt)}")
    L(r"            };")
    L(r"            \addplot[black line, dashed] coordinates {")
    L(f"              {coord_line(p_coords, s_vals, fmt)}")
    L(r"            };")
    L(r"          \end{axis}")
    L(r"        \end{tikzpicture}")
    L(r"      \end{figure}")
    L(r"    \end{column}")
    L(r"    \begin{column}{0.42\textwidth}")
    L(r"      \tiny\renewcommand{\arraystretch}{0.85}\setlength{\tabcolsep}{2pt}")
    L(r"      \begin{tabular}{lrr}")
    L(r"        \toprule")
    L(r"        $p$ & haircut & supply \\")
    L(r"        \midrule")
    for row in table_rows_hs(p_table, h_vals, s_vals, fmt, bold=bold):
        L(row)
    L(r"        \bottomrule")
    L(r"      \end{tabular}")
    L(r"    \end{column}")
    L(r"  \end{columns}")
    L(r"\end{frame}")

# =====================================================================
def make_bankruptcy_slide(L, title, folder_label, p_table, p_coords,
                          h_data, s_data, xmax, bold, contagion_ymax=0.4):
    """Generate a bankruptcy slide."""
    L(r"\begin{frame}{Bankruptcies (" + folder_label + r")}")
    L(r"  \begin{columns}[T]")
    L(r"    \begin{column}{0.55\textwidth}")
    L(r"      \begin{figure}")
    L(r"        \centering")
    L(r"        \begin{tikzpicture}")
    L(r"        \begin{axis}[")
    L(r"          ylabel={Rationing + Repayment},")
    L(r"          ymin=0, ymax=8,")
    L(r"          xmin=0, xmax=" + str(xmax) + ",")
    L(r"          xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},")
    L(r"        ]")
    L(r"          \addplot[only marks, mark=square*, red] coordinates {")
    L(f"            {coord_line(p_coords, h_data['rationing'], '.4f')}")
    L(r"          };")
    L(r"          \addplot[black line] coordinates {")
    L(f"            {coord_line(p_coords, h_data['rationing'], '.4f')}")
    L(r"          };")
    L(r"          \addplot[only marks, mark=o, red] coordinates {")
    L(f"            {coord_line(p_coords, h_data['failed'], '.2f')}")
    L(r"          };")
    L(r"          \addplot[black line] coordinates {")
    L(f"            {coord_line(p_coords, h_data['failed'], '.2f')}")
    L(r"          };")
    L(r"        \end{axis}")
    L(r"        \begin{axis}[")
    L(r"          ylabel={Contagion},")
    L(r"          ymin=0, ymax=" + str(contagion_ymax) + ",")
    L(r"          xmin=0, xmax=" + str(xmax) + ",")
    L(r"          axis y line*=right, axis x line=none,")
    L(r"          ylabel near ticks,")
    L(r"          yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},")
    L(r"        ]")
    L(r"          \addplot[only marks, mark=*, red] coordinates {")
    L(f"            {coord_line(p_coords, h_data['contagion'], '.4f')}")
    L(r"          };")
    L(r"          \addplot[black line] coordinates {")
    L(f"            {coord_line(p_coords, h_data['contagion'], '.4f')}")
    L(r"          };")
    L(r"        \end{axis}")
    L(r"        \end{tikzpicture}")
    L(r"      \end{figure}")
    L(r"    \end{column}")
    L(r"    \begin{column}{0.42\textwidth}")
    L(r"      \tiny")
    L(r"      Bankruptcies:")
    L(r"      \begin{enumerate}")
    L(r"        \item[] $\blacksquare$ \textbf{Rationing.}")
    L(r"        \item[] $\bullet$ \textbf{Contagion.}")
    L(r"        \item[] $\circ$ \textbf{Failed repayment.}")
    L(r"      \end{enumerate}")
    L(r"      \tiny\renewcommand{\arraystretch}{0.85}\setlength{\tabcolsep}{2pt}")
    L(r"      \begin{tabular}{lccc}")
    L(r"        \toprule")
    L(r"        $p$ & Rat. & Cont. & Fail. \\")
    L(r"        \midrule")
    for row in table_rows3(p_table, h_data['rationing'], h_data['contagion'],
                           h_data['failed'], ".2f", ".2f", bold=bold):
        L(row)
    L(r"        \bottomrule")
    L(r"      \end{tabular}")
    L(r"    \end{column}")
    L(r"  \end{columns}")
    L(r"\end{frame}")


# =====================================================================
def generate_file(outpath, xmax_str, xmax, p_table, p_coords,
                  d_hair, d_supp, n, contagion_ymax):
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
    L(r"    xmin=0, xmax=" + xmax_str + ",")
    L(r"    xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=2},")
    L(r"  },")
    L(r"  black line/.style={")
    L(r"    smooth, thick, black,")
    L(r"  },")
    L(r"}")
    L("")
    is_12 = "0.12" in str(xmax)
    if is_12:
        title_range = "p=0..0.12"
        exp_folder = "exp_min_p_0_01"
    else:
        title_range = "p=0..1"
        exp_folder = "exp_min_p_0_1"
    L(r"\title{Capacity with Haircut vs.\ Capacity=Supply (" + title_range + r")}")
    L(r"\subtitle{" + exp_folder + r"\_capacity\_haircut vs.\ " + exp_folder + r"\_capacity\_supply}")
    L(r"\author{}")
    L(r"\date{\today}")
    L("")
    L(r"\begin{document}")
    L("")
    L(r"\begin{frame}")
    L(r"  \titlepage")
    L(r"\end{frame}")
    L("")

    # --- Bankruptcy slides ---
    haircut_label = exp_folder + r"\_capacity\_haircut"
    supply_label  = exp_folder + r"\_capacity\_supply"
    make_bankruptcy_slide(L, "Bankruptcies (capacity with haircut)", haircut_label,
                          p_table, p_coords, d_hair, d_hair, xmax, p_bold, contagion_ymax)
    L("")
    make_bankruptcy_slide(L, "Bankruptcies (capacity=supply)", supply_label,
                          p_table, p_coords, d_supp, d_supp, xmax, p_bold, contagion_ymax)
    L("")

    # --- Comparison slides ---
    for key, label, ymin, ymax, fmt in comparisons:
        # skip bankruptcy keys already shown, and 'failed' is mapped to 'Bad Debt'
        if key in ("rationing", "contagion"):
            continue
        make_comparison_slide(L, label, key, p_table, p_coords,
                              d_hair, d_supp, ymin, ymax, fmt, xmax, p_bold)
        L("")

    L(r"\end{document}")

    with open(outpath, "w") as f:
        f.write("\n".join(lines))
    print(f"Written to {outpath}, {len(lines)} lines")

outdir = r"C:\Users\heccasan\OneDrive\doct\006.interbank\code2\doc"
generate_file(
    os.path.join(outdir, "experiments4.tex"),
    "1.0", 1.0,
    p_table_1, p_coords_1,
    d1_hair, d1_supp, n1, 0.2,
)
generate_file(
    os.path.join(outdir, "experiments3.tex"),
    "0.12", 0.12,
    p_table_12, p_coords_12,
    d12_hair, d12_supp, n12, 0.4,
)
