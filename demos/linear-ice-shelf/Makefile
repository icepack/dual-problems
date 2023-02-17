
results.pdf: results.json make_plots.py
	python make_plots.py --input $< --output $@

results.json: convergence_test.py
	python convergence_test.py --degree 1 --log-nx-min 4 --log-nx-max 7 --num-steps 9 --output results.json
	python convergence_test.py --degree 2 --log-nx-min 3 --log-nx-max 6 --num-steps 9 --output results.json
