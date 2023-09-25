results.pdf: ice_shelf_results.json ice_stream_results.json make_plots.py
	python make_plots.py --output $@

%_results.json: %.py
	python $< --degree 1 --log-nx-min 4 --log-nx-max 7 --num-steps 9 --output $@
	python $< --degree 2 --log-nx-min 3 --log-nx-max 6 --num-steps 9 --output $@
