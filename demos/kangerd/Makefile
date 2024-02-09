kangerdlugssuaq-year5.h5: simulate.py kangerdlugssuaq-year1.h5
	python simulate.py \
		--input kangerdlugssuaq-year1.h5 \
		--final-time 4.0 \
		--timesteps-per-year 86 \
		--calving \
		--melt-rate 30e3 \
		--mask-smoothing-length 1e3 \
		--snes-max-it 500 \
		--snes-rtol 1e-5 \
		--output kangerdlugssuaq-year5.h5

kangerdlugssuaq-year1.h5: simulate.py kangerdlugssuaq-extrapolated.h5
	python simulate.py \
		--input kangerdlugssuaq-extrapolated.h5 \
		--final-time 1.0 \
		--timesteps-per-year 192 \
		--output kangerdlugssuaq-year1.h5

kangerdlugssuaq-extrapolated.h5: extrapolate.py kangerdlugssuaq-initial.h5
	python extrapolate.py

kangerdlugssuaq-initial.h5: initialize.py
	python initialize.py
