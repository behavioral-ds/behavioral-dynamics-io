# Beyond Content: Behavioral Policies Reveal Actors in Information Operations

This repository contains the code to reproduce the experiments and figures for the paper:
> *Beyond Content: Behavioral Policies Reveal Actors in Information Operations*
XXX, _TBD_, 2025 (Under Review).

Repository is organised into three sections: Data processing, classification, and visualisation. 

## Data Processing

### Reddit Processing and Trajectory Construction
To recreate our data processing process, run each of the scripts inside "raw_dump_processing" in order of the number at the beginning of each script. These scripts operate on the monthly Reddit dumps provided by the Pushift API and are extremely computationally intensive. An additional file of the non-troll users used in this work is required but not disclosed as per ethics.

### Reference 
These scripts are provided by Lanqin Yuan, Philipp J. Schneider, and Marian-Andrei Rizoiu. 2025. Behavioral Homophily in Social Media via Inverse Reinforcement Learning: A Reddit Case Study. In Proceedings of the ACM on Web Conference 2025 (WWW '25). Association for Computing Machinery, New York, NY, USA, 576–589. https://doi.org/10.1145/3696410.3714618.

```

```


## Policy Inference
Scripts which infer user polices are found scripts in "policy_inference". These require all scripts in reddit processing to be run.

# Classification Experiments
Experiment scripts are found under the "experiments" directory. These scripts generate result outputs which are needed for visualisation.


