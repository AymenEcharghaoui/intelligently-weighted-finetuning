# Intelligently Reweighting Multiple Reference Models for Direct Preference Optimization of LLMs.

This repository accompanies the paper "Intelligently Reweighting Multiple Reference Models for Direct Preference Optimization of LLMs," written by Skyler Wu and Aymen Echarghaoui and submitted as a final course project to Stanford University's [CS 329H: Machine Learning from Human Preferences](https://web.stanford.edu/class/cs329h/) course, taught by Professor Sanmi Koyejo and Dr. Andy Haupt.

**Environment Setup and Dependencies + Expected Runtime and Computational Requirements:** all experiments were run on Google Colab Pro+ using single NVIDIA A100 High-RAM instances with 80 GB of GPU memory each. The only additional packages the Colab user needs to install can be done via the following: `! pip install bitsandbytes==0.46.0 accelerate==1.7.0`. If the reader is 

We consider these settings to be the minimum requirements for running our experiments as even with these A100 instances, significant engineering (see our paper Appendix) was necessary to fit experiments in memory and with reasonable runtime. All experiments for `UltraFeedback` take 1.5 to 2 hours per seeded trial, while all experiments for `SafeRLHF` take 0.5 to 0.75 hours per seeded trial.

**Reproducing All Results:** Which scripts produce which results/figures in the paper? Gradient-clipping on Online 1 (UltraFeedback) and it still went boom.

**Repository Structure and File Organization:**
The jupyter notebook vdw_vaw_swcw_mrpo_mdpo.ipynb implements the methods `VDW`, `VAW` and `SWCW` for computing the mixture weights $$\alphas$$, for both losses `MRPO` and `MDPO`. It works for both datasets `SafeRLHF` and `Ultra-FeedBack` across the different seeds. 
In the first cell of the notebook, users should select the desired configuration by setting:

- `DATASET` to one of: `"ultrafeedback_binarized"` or `"PKU-SafeRLHF-30K-standard"`
- `SEED` to any seed in `[0, 1, 2, 3, 4]`
- `USE_MRPO_OVER_MDPO` to `True` (use MRPO) or `False` (use MDPO)
- `ALPHA_METHOD` to one of: `["offline_1", "offline_2", "online_1", "arwc_normalized"]`

Once these values are chosen, simply run the notebook to reproduce our results. Here, 

- `offline_1` means `VDW`
- `offline_2` means `VAW`
- `online_1` means `SWCW`
- `arwc_normalized` means `original`


**Links to Any Required Datasets or Instructions for Data Generation:**

