FROM firedrakeproject/firedrake-vanilla:2024-05

ENV PATH=/home/firedrake/firedrake/bin:$PATH

RUN sudo apt-get update && \
    sudo apt-get install -yq patchelf texlive texlive-fonts-extra

RUN python -m pip install netCDF4 matplotlib-scalebar
RUN python -m pip install git+https://github.com/firedrakeproject/irksome.git
RUN python -m pip install git+https://github.com/icepack/Trilinos.git
RUN python -m pip install git+https://github.com/icepack/pyrol.git
RUN python -m pip install git+https://github.com/icepack/icepack.git
RUN python -m pip install git+https://github.com/icepack/icepack2.git

ENV OMP_NUM_THREADS=1
