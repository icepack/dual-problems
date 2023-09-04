dual-problems.pdf: dual-problems.tex dual-problems.bib
	cd demos/singularity && python singularity.py && cd ../../
	cd demos/linear-ice-shelf && make results.pdf && cd ../../
	cd demos/linear-ice-stream && make results.pdf && cd ../../
	cd demos/gibbous-ice-shelf && make gibbous.png && cd ../../
	cd demos/mismip && make mismip.pdf && cd ../../
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

all: dual-problems.pdf
