 cd doc
 pdflatex algorithm.tex
 del alg-1.png
 pdftoppm -png algorithm.pdf alg
 cd ..
 pandoc doc\README.tex -t markdown+pipe_tables-simple_tables-multiline_tables -o README.md
 pdflatex doc\README.tex
 move README.pdf doc
 del README.out
 del README.log  
 del readme.aux
 del readme.dvi



rem pytest tests/ --durations=0 -q --tb=no
rem python -m unittest discover -s tests
rem if %errorlevel%==0 (
rem git add .
rem git commit -a
rem git push
rem )

