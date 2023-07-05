dual-problems.pdf: dual-problems.tex dual-problems.bib
	cd demos/singularity && python singularity.py && cd ../../
	cd demos/linear-ice-shelf && make results.pdf && cd ../../
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

all: dual-problems.pdf
