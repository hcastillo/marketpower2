uv run experiments\exp_min_p_0_01_0507.py --do
uv run experiments\exp_min_p_0_01_0507b.py --do
uv run experiments\exp_min_p_0_01_0507c.py --do

uv run gen_comparison_tex.py --dir1 c:\experiments\0707\exp_min_p_0_01_mc500_omega03 --dir2 c:\experiments\0707\exp_min_p_0_01_mc500_omega05 --dir3 c:\experiments\0707\exp_min_p_0_01_mc500_omega06 --label1 "$\omega=0.35$" --label2 "$\omega=0.55$" --label3 "$\omega=0.6$" --output doc\experiments9.tex --table
cd doc
pdflatex experiments9.tex
cd ..

uv run gen_bankruptcies_descomposition_tex.py --dir1 c:\experiments\0707\exp_min_p_0_01_mc500_omega03 --dir2 c:\experiments\0707\exp_min_p_0_01_mc500_omega05 --dir3 c:\experiments\0707\exp_min_p_0_01_mc500_omega06 --label1 "$\omega=0.35$" --label2 "$\omega=0.55$" --label3 "$\omega=0.6$" --output doc\experiments9b.tex --table
cd doc
pdflatex experiments9b.tex
cd ..

rem uv run experiments/exp_rl_capacity.py --train --timesteps 50000
rem uv run experiments/exp_rl_capacity.py --evaluate --steps 1000

rem uv run experiments/exp_rl_capacity1.py --train --timesteps 50000
rem uv run experiments/exp_rl_capacity1.py --evaluate --steps 1000

rem uv run experiments/exp_rl_psi.py --train --timesteps 50000
rem uv run experiments/exp_rl_psi.py --evaluate --steps 1000

rem uv run experiments/exp_rl_psi_equity.py --train --timesteps 50000
rem uv run experiments/exp_rl_psi_equity.py --evaluate --steps 1000
