# Intelligently Reweighting Multiple Reference Models for Direct Preference Optimization of LLMs.

This repository accompanies the paper "Intelligently Reweighting Multiple Reference Models for Direct Preference Optimization of LLMs," written by Skyler Wu and Aymen Echarghaoui and submitted as a final course project to Stanford University's [CS 329H: Machine Learning from Human Preferences](https://web.stanford.edu/class/cs329h/) course, taught by Professor Sanmi Koyejo and Dr. Andy Haupt.

**Environment Setup and Dependencies + Expected Runtime and Computational Requirements:** all experiments were run on Google Colab Pro+ using single NVIDIA A100 High-RAM instances with 80 GB of GPU memory each. The only additional packages the Colab user needs to install can be done via the following: `! pip install bitsandbytes==0.46.0 accelerate==1.7.0`. If the reader is 

We consider these settings to be the minimum requirements for running our experiments as even with these A100 instances, significant engineering (see our paper Appendix) was necessary to fit experiments in memory and with reasonable runtime. All experiments for `UltraFeedback` take 1.5 to 2 hours per seeded trial, while all experiments for `SafeRLHF` take 0.5 to 0.75 hours per seeded trial.


**Repository Structure - File Organization - Reproducibility:**
The following maps the method names used in code to the ones used in the paper : 
- `offline_1` means `VDW`
- `offline_2` means `VAW`
- `online_1` means `SWCW` (with the one-hot variant denoted analogously)
- `arwc_normalized` means `original`

The Jupyter Notebook vdw_vaw_swcw_mrpo_mdpo.ipynb implements the methods `VDW`, `VAW` and `SWCW` for computing the mixture weights alphas, for both losses `MRPO` and `MDPO`. It works for both datasets `SafeRLHF` and `Ultra-FeedBack` across the different seeds. 
In the first cell of the notebook, users should select the desired configuration by setting:
- `DATASET` to one of: `"ultrafeedback_binarized"` or `"PKU-SafeRLHF-30K-standard"`
- `SEED` to any seed in `[0, 1, 2, 3, 4]`
- `USE_MRPO_OVER_MDPO` to `True` (use MRPO) or `False` (use MDPO)
- `ALPHA_METHOD` to one of: `["offline_1", "offline_2", "online_1", "arwc_normalized"]`
Once these values are chosen, simply run the notebook **end-to-end without modification** to reproduce our results. 

The Jupyter Notebook `data_process.ipynb` loads preference datasets, computes token lengths using the seven reference tokenizers, processes the data using a worst-case on token length, then saves the processed data into train-valid-test splits for later use. 

The Jupyter Notebook `precomputing_reference_log_probs.ipynb` loads pre-trained language models and reference tokenizers, computes log probabilities and token lengths for all reference models on the preference datasets and saves the results for later use.

The Jupyter Notebook `splitting_over_multiple_seeds.ipynb` first combines all splits per dataset (preference datasets + precomputed reference log probabilities), then per seed shuffles the whole data and creates new train-val-test splits and finally saves the new splits for experiments.

The Jupyter Notebook `baseline_dpo.ipynb` implements the 7 baseline DPO finetuning experiments using Qwen_Qwen2.5-0.5B-Instruct as base model and each of the 7 reference models.
The Jupyter Notebook `tsw.ipynb` implements the Thompson Sampling online method for computing the alpha weights during finetuning.

The Jupyter Notebook `swcw_one_hot.ipynb` implements the SWCW one-hot method for computing the alpha weights during finetuning. 

The Jupyter Notebook `analyzer.ipynb` combines all logs from all experiments, normalizes the names and saves summary files per method.

The Jupyter Notebook `visualizer.ipynb` loads the summary files and produces the plots in the paper. 

The file `helpers.py` contains helper functions used across multiple notebooks, including mdpo, mrpo, log probability computations of the reply given the prompt for the finetuned model, etc. 

The compressed folder `cleaned_results.zip` contains all the summary files produced by `analyzer.ipynb` for all methods, datasets and seeds.

The compressed folder `raw_results.zip` contains all the raw logs produced during finetuning, normalized into one naming convention by `analyzer.ipynb`.

The compressed folder `figures.zip` contains all the figures produced by `visualizer.ipynb` for the paper.

**Links to Any Required Datasets or Instructions for Data Generation:**
We worked with the following two public preference datasets : 

- [UltraFeedback (Binarized)](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) 
- [PKU-safeRLHF](https://huggingface.co/datasets/RLHFlow/PKU-SafeRLHF-30K-standard)
