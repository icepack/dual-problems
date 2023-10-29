volumes.pdf thickness.pdf &: larsen-simulation.h5
	python make_plots.py

larsen-simulation.h5: simulate.py larsen-extrapolated.h5
	python simulate.py

larsen-extrapolated.h5: extrapolate.py larsen-initial.h5
	python extrapolate.py

larsen-initial.h5: initialize.py
	python initialize.py
