mismip.h5: mismip.py
	python mismip.py --output $@ --final-time 2000.0 --timestep 2.0 --num-cells 30

mismip.pdf: mismip.h5 make_plots.py
	python make_plots.py --input $< --output $@
