 cd doc
 pdflatex algorithm.tex
 del alg-000001.png
 pdftopng -r 300 algorithm.pdf alg
 cd ..
 pandoc doc\README.tex -t markdown+pipe_tables-simple_tables-multiline_tables -o README.md
