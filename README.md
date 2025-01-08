# Repelling Random Walks

This repository accompanies the paper 'Repelling Random Walks', which was published at ICLR 2024 by Isaac Reid, Krzysztof Choromanski, Eli Berger and Adrian Weller. The manuscript can be found [here](https://arxiv.org/pdf/2310.04854). 
Instead of being independent, repelling random walks sample a particular node's neighbours *without* replacement at every timestep. 
Each walk's marginal distribution is unchanged so graph-based Monte Carlo estimators remain unbiased, but the correlations between walkers tend to improve estimator concentration. 
This is an example of a **quasi**-Monte Carlo scheme for estimators defined on discrete spaces -- one of the first of its kind.

<div align="center">
  <img src="/rrws_schematic.png" alt="Alt text" width="500">
</div>

**This repo.** 
This lightweight repo reproduces Fig. 2 of the paper, which shows the kernel approximation using [graph random features](https://arxiv.org/pdf/2310.04859) when walkers are i.i.d., exhibit [antithetic termination](https://arxiv.org/abs/2305.12470), are repelling, or both. As per our theoretical guarantees (see Thm. 3.1), the estimator convergence is improved with repelling walks. 

**Installation instructions.** 
The requirements of the repo are minimal.
For a quick installation, run:

```bash
conda env create -f environment.yml --name new_environment_name
```
in the folder of the downloaded repo in terminal. But in practice one only needs numpy, matplotlib, scipy and ipykernel.

**Significance and extensions.**
Quasi-Monte Carlo (QMC) methods, which enhance the convergence of estimators by introducing correlations between samples, are well-established in the Euclidean setting.
However, their application to discrete spaces, such as graphs, remains unexplored.
This work introduces one of the first QMC algorithms tailored for discrete settings, using interacting random walks.
We hope this contribution sparks further research in this promising area.

