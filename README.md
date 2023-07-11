# NCM-Dataset-Preprocessing
- This repository contains the code for preprocessing the NCM dataset. The dataset is available in [this paper](https://pubsonline.informs.org/doi/10.1287/msom.2020.0923).
- The main purposes of this repository are as follows:
  1. Split user behaviors into several sessions.
  2. Extract the sessions that contain click behaviors.
  3. Extract the corresponding features that related to the click behaviors.
  4. Convert the data above to `torch.Tensor` or `jax.numpy.ndarray` format.