@echo off

python -m unittest discover -s tests
if %errorlevel%==0 (
 cd doc
 pdflatex algorithm.tex
 del alg-000001.png
 pdftopng -r 300 algorithm.pdf alg
 cd ..
 pandoc doc\README.tex -t markdown+pipe_tables-simple_tables-multiline_tables -o README.md
 git add .
 git commit -a
 git push
)

