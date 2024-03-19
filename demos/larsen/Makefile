volumes.pdf contours.pdf &: larsen-simulation.h5 make_plots.py
	python make_plots.py

larsen-simulation.h5: simulate.py larsen-extrapolated.h5
	python simulate.py \
		--time-to-calve 2.0 \
		--final-time 40.0

larsen-extrapolated.h5: extrapolate.py larsen-initial.h5
	python extrapolate.py

larsen-initial.h5: initialize.py
	python initialize.py
