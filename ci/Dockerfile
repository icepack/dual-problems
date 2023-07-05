
FROM firedrakeproject/firedrake@sha256:ee91b0c25ef044b09a3e33807b2a0fddc98df74e7f96d5fac5c0319ab1a95bf1

MAINTAINER shapero@uw.edu

ENV PATH=/home/firedrake/firedrake/bin:$PATH

RUN sudo apt update && \
    sudo apt install -yq texlive texlive-fonts-extra

RUN git clone https://github.com/icepack/icepack.git && \
    cd icepack && \
    pip install --editable .
