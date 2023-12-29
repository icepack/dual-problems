
FROM firedrakeproject/firedrake-vanilla:2023-11

MAINTAINER shapero@uw.edu

ENV PATH=/home/firedrake/firedrake/bin:$PATH

RUN sudo apt update && \
    sudo apt install -yq patchelf texlive texlive-fonts-extra

RUN pip install git+https://github.com/firedrakeproject/irksome.git && \
    pip install git+https://github.com/icepack/Trilinos.git && \
    pip install git+https://github.com/icepack/pyrol.git && \
    pip install git+https://github.com/icepack/icepack.git

RUN pip install netCDF4
