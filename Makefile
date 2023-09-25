all: dual-problems.pdf

dual-problems.pdf: dual-problems.tex dual-problems.bib
	cd demos/singularity && make primal.pdf dual.pdf && cd ../../
	cd demos/convergence-tests && make results.pdf && cd ../../
	cd demos/slab && make tables/table_alpha0.50.txt && cd ../../
	cd demos/gibbous-ice-shelf && make steady-state.pdf calved-thickness.pdf volumes.pdf && cd ../../
	cd demos/mismip && make mismip.pdf && cd ../../
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

