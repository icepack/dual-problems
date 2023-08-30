dual-problems.pdf: dual-problems.tex dual-problems.bib
	cd demos/singularity && python singularity.py && cd ../../
	cd demos/linear-ice-shelf && make results.pdf && cd ../../
	cd demos/gibbous-ice-shelf && make gibbous.png && cd ../../
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

all: dual-problems.pdf
