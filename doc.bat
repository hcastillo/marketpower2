 cd doc
 pdflatex algorithm.tex
 pdflatex README.tex
 del alg-000001.png
 pdftoppm -png archivo.pdf alg
 cd ..
 pandoc doc\README.tex -t markdown+pipe_tables-simple_tables-multiline_tables -o README.md



rem pytest tests/ --durations=0 -q --tb=no
rem python -m unittest discover -s tests
rem if %errorlevel%==0 (
rem git add .
rem git commit -a
rem git push
rem )

