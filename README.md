# grokking-mechanism-test
PyTorch tests of the geometric grokking hypothesis — that delayed generalization stems from a geometric phase transition minimizing low-frequency energy of an implicit graph Laplacian in learned representations, rather than from weight decay or circuit efficiency alone.

Torch: pip install -r requirements-torch.txt && KERAS_BACKEND=torch python scripts/train.py

TF: pip install -r requirements-tensorflow.txt && KERAS_BACKEND=tensorflow python scripts/train.py

JAX: pip install -r requirements-jax.txt && KERAS_BACKEND=jax python scripts/train.py