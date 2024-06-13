gibbous.pdf volumes.pdf residuals.pdf &: calving.h5 residuals.json make_plots.py
	python make_plots.py --calving-freq 24.0

calving.h5: steady-state-fine.h5 gibbous.py
	python gibbous.py --resolution 2e3 --final-time 200.0 --num-steps 400 --calving-freq 24.0 --input $< --output $@

steady-state-fine.h5: steady-state-coarse.h5 gibbous.py
	python gibbous.py --resolution 2e3 --final-time 400.0 --num-steps 400 --input $< --output $@

residuals.json: steady-state-coarse.h5 primal_dual_calving.py
	python primal_dual_calving.py

steady-state-coarse.h5: gibbous.py
	python gibbous.py --resolution 5e3 --final-time 400.0 --num-steps 200 --output $@
