mismip.h5: mismip.py
	python mismip.py --output $@ --final-time 2000.0 --timestep 2.5 --num-cells 20

mismip.pdf: mismip.h5 make_plots.py
	python make_plots.py --input $< --output $@
