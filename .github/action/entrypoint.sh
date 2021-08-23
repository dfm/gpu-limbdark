#!/bin/sh -l

cd /github/workspace
JAX_LIMBDARK_CUDA=yes python3 -m pip install .
python3 -c 'import jax_limbdark;print(jax_limbdark.__version__)'
python3 -c 'import jax_limbdark.gpu_ops'
