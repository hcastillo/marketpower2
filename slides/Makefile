#!/usr/bin/make -f

DOCNAME:=$(wildcard *_doc.tex)
FNAME:=$(DOCNAME:%_doc.tex=%)

all:
	@echo "Usage: make (pdfs|clean)"; \
	echo

pdfs: ${FNAME}.pdf ${FNAME}_p-1x2.pdf ${FNAME}_p-2x2.pdf

${FNAME}.pdf: ${FNAME}.tex ${FNAME}_doc.tex
	pdflatex ${FNAME}.tex &&			\
	pdflatex ${FNAME}.tex

${FNAME}_p.pdf: ${FNAME}_p.tex ${FNAME}_doc.tex
	pdflatex ${FNAME}_p.tex &&			\
	pdflatex ${FNAME}_p.tex 

${FNAME}_p-1x2.pdf: ${FNAME}_p.pdf
	pdfnup --nup 1x2 --scale 0.9 --delta "1cm 1cm" --frame true \
	       --no-landscape --suffix "1x2" $<

${FNAME}_p-2x2.pdf: ${FNAME}_p.pdf
	pdfnup --nup 2x2 --scale 0.9 --delta "1cm 1cm" --frame true \
	       --landscape --suffix "2x2" $<

${FNAME}_notes.pdf: ${FNAME}_notes.tex ${FNAME}_doc.tex
	pdflatex ${FNAME}_notes.tex &&			\
	pdflatex ${FNAME}_notes.tex 

${FNAME}_notes-2x4.pdf: ${FNAME}_notes.pdf
	pdfnup --nup 2x4 --scale 0.95 --delta "1cm 1cm" --frame true  $<


clean:
	rm -f *~ *.aux *.bm *.dvi *.log *.toc *.nav *.out *.snm *.vrb
	rm -rf auto
