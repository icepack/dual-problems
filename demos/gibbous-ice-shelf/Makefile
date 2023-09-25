steady-state.pdf calved-thickness.pdf volumes.pdf &: calving.h5 make_plots.py
	python make_plots.py --calving-freq 24.0

calving.h5: steady-state-fine.h5 gibbous.py
	python gibbous.py --resolution 2e3 --final-time 200.0 --num-steps 400 --calving-freq 24.0 --input $< --output $@

steady-state-fine.h5: steady-state-coarse.h5 gibbous.py
	python gibbous.py --resolution 2e3 --final-time 400.0 --num-steps 400 --input $< --output $@

steady-state-coarse.h5: gibbous.py
	python gibbous.py --resolution 5e3 --final-time 400.0 --num-steps 200 --output $@
