gibbous.png: gibbous.h5 make_plots.py
	python make_plots.py --input $< --output $@

steady-state-coarse.h5: gibbous.py
	python gibbous.py --resolution 5e3 --final-time 400.0 --num-steps 200 --output $@

steady-state-fine.h5: steady-state-coarse.h5 gibbous.py
	python gibbous.py --resolution 2e3 --final-time 400.0 --num-steps 400 --input $< --output $@
