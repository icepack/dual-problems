%.pdf: %.tex %.bib
	pdflatex $<
	bibtex $(basename $<)
	pdflatex $<
	pdflatex $<

all: dual-problems.pdf
