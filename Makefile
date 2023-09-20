dual-problems.pdf: dual-problems.tex dual-problems.bib
	cd demos/singularity && python singularity.py && cd ../../
	cd demos/convergence-tests && make results.pdf && cd ../../
	cd demos/gibbous-ice-shelf && make steady-state.pdf calved-thickness.pdf volumes.pdf && cd ../../
	cd demos/mismip && make mismip.pdf && cd ../../
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

all: dual-problems.pdf
