# Beyond Content: Behavioral Policies Reveal Actors in Information Operations

This repository contains the code to reproduce the experiments and figures for the paper:
> **Beyond Content: Behavioral Policies Reveal Actors in Information Operations**. XXX, _TBD (Under Review)_, 2025.

The codebase builds on the data-processing pipeline introduced in:
> Lanqin Yuan, Philipp J. Schneider, and Marian-Andrei Rizoiu. 2025. **Behavioral Homophily in Social Media via Inverse Reinforcement Learning: A Reddit Case Study**. In _Proceedings of the ACM Web Conference 2025_ (WWW ’25), 576–589.
https://doi.org/10.1145/3696410.3714618

The repository is organized into three main components:
1. **Data processing** – from raw Reddit dumps to user trajectories and inferred policies.
2. **Classification experiments** – training and evaluating policy- and content-based classifiers.
3. **Visualization** – scripts to reproduce the figures in the paper.

## 1. Getting Started

- Python >= 3.10
- Install dependencies
```
pip install -r requirements.txt
```

## 2. Repository Structure

At a high level, the repository is organized as:
```
.
├── raw_dump_processing/    # Reddit dumps → cleaned user trajectories
├── policy_inference/       # IRL / GAIL / empirical policy inference
├── experiments/            # Classification experiments and evaluation
├── visualization/          # Plotting and figure generation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

You do **not** need to run all components if you only want to reproduce specific stages (e.g., classification on pre-computed policies), but the full pipeline is:
1. `raw_dump_processing/`
2. `policy_inference/`
3. `experiments/`
4. `visualization/`

## 3. Data Processing

### Reddit Processing and Trajectory Construction

Scripts for processing raw Reddit data and constructing user trajectories are located in `raw_dump_processing/`.
**Inputs:**
- Monthly Reddit comment and submission dumps from the Pushshift API (compressed JSON files).
- An additional file listing the organic users used in this work.

> **Important (Ethics):**
The list of organic users cannot be shared publicly due to ethics and privacy constraints. To reproduce the full dataset, you will need access to a comparable set of organic users from Reddit and must obtain appropriate ethics approval for your institution.

To recreate our data processing process, run each of the scripts inside `raw_dump_processing' in order of the number at the beginning of each script. 

Each script reads the outputs of the previous step and writes intermediate results to `data/processed/` (or the path specified in the script). The final outputs are:
- A set of user trajectories (trolls and organics) with state–action sequences.
- Metadata used downstream for policy inference and classification.

## 4. Policy Inference

Scripts for inferring user policies from trajectories are located in `policy_inference/`.

Supported policy representations include:
- Empirical policies (state–action frequency estimates).
- MaxEnt Deep IRL policies (maximum-entropy deep inverse reinforcement learning).
- GAIL (Generative Adversarial Imitation Learning) policies.

These scripts assume that all preprocessing steps in `raw_dump_processing/` have been successfully completed.

## 5. Classification Experiments
Classification experiments are located under `experiments/`. These scripts:
- Load the inferred policies (policy-based representations).
- Load content-based representations (e.g. text embeddings).
- Train and evaluate classifiers (random forest, gradient boosting).

## 6. Visualization
Figure-generation scripts are located in `visualization/`. They mainly take the result files from `experiments/` and other raw data and produce the plots shown in the paper.

## 7. Citation
If you use this codebase or build upon our methods, please cite:

```
@article{xxx2025beyondcontent,
  title        = {Beyond Content: Behavioral Policies Reveal Actors in Information Operations},
  author       = {XXX},
  journal      = {TBD (Under Review)},
  year         = {2025}
}

@inproceedings{yuan2025behavioralhomophily,
  title        = {Behavioral Homophily in Social Media via Inverse Reinforcement Learning: A Reddit Case Study},
  author       = {Yuan, Lanqin and Schneider, Philipp J. and Rizoiu, Marian-Andrei},
  booktitle    = {Proceedings of the ACM Web Conference 2025 (WWW '25)},
  pages        = {576--589},
  year         = {2025},
  publisher    = {ACM},
  address      = {New York, NY, USA},
  doi          = {10.1145/3696410.3714618}
}
```

## 8. Contact 
For questions about the code or the paper, please contact:
- XXX
