
FROM firedrakeproject/firedrake-vanilla:2023-11

MAINTAINER shapero@uw.edu

ENV PATH=/home/firedrake/firedrake/bin:$PATH

RUN sudo apt update && \
    sudo apt install -yq texlive texlive-fonts-extra

RUN git clone https://github.com/icepack/icepack.git && \
    cd icepack && \
    pip install git+https://github.com/icepack/Trilinos.git && \
    pip install git+https://github.com/icepack/pyrol.git && \
    pip install --editable .
