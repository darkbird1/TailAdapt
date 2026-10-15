# TailAdapt: Heavy-Tailed Sparse Variational Adaptation for Long-Tailed Class Incremental Learning (NeurIPS 2026)

## Abstract
Long-Tailed Class-Incremental Learning (LT-CIL) is challenging due to severe class imbalance, where frequent head classes dominate gradient updates and suppress learning for rare tail classes. Beyond data imbalance, we identify asymmetric likelihood signals within each session, where underrepresented classes often provide weaker and more localized evidence for adapting previously learned representations. This asymmetry disrupts the stability-plasticity tradeoff, leading to pronounced forgetting and poor tail-class performance. We propose a hierarchical variational framework, TailAdapt, for prompt and adapter-based continual learning that enables adaptive group-wise regularization of model parameters. TailAdapt employs a hierarchical inverse Gaussian scale-mixture prior, which induces a heavy-tailed marginal Normal Inverse Gaussian (NIG) distribution over prompt and adapter parameters. This formulation encourages selective plasticity so that most parameter groups are strongly regularized to preserve previously learned knowledge, while a small, data-supported subset is allowed to adapt substantially. This selective plasticity allocates adaptation capacity according to the relevance of parameter groups to the current data, mitigating interference from head classes and improving learning for tail classes. Extensive experiments on LT-CIL benchmarks demonstrate consistent improvements over strong baselines, with better tail-class performance, reduced forgetting, and more efficient utilization of prompt and adapter capacity.

## Code Details
The configuration files are provided inside the exps directory and are to be executed using:

```
    python main.py --config=./exps/dataset/experiment.json
```
We are grateful to the following GitHub repos for their codebase and we build our method using their respective codebases as our baseline.

- https://gitlab.com/viper-purdue/panda/
- https://github.com/LAMDA-CL/LAMDA-PILOT/
