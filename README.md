# Adaptive Reduced Basis Trust Region Methods for Inverse Parameter Identification Problems

[![DOI](https://zenodo.org/badge/690415728.svg)](https://zenodo.org/badge/latestdoi/690415728)

```
# ~~~
# This file is part of the paper:
#   
#           "Adaptive Reduced Basis Trust Region Methods for Inverse Parameter Identification Problems"
#
# Preprint: https://arxiv.org/abs/2309.07627
#
# Copyright 2023 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Michael Kartmann, Tim Keil
# ~~~
```

In this repository, we provide the code for the numerical experiments of the paper "Adaptive Reduced Basis Trust Region Methods for Inverse Parameter Identification Problems" by Michael Kartmann, Tim Keil, Mario Ohlberger, Stefan Volkwein, Barbara Kaltenbacher. A preprint is available [here](https://arxiv.org/abs/2309.07627). The implementation is based on Python and includes both the reduction of the infinite-dimensional parameter space and the reduction of the state space in an error-aware trust region framework.

## Setup 

To run the code you need to install the python package [pyMOR](https://pymor.org) Version 2023.01.01 in your (local) environment. This can be done using `conda` via
```
conda install -c conda-forge pymor=2023.1.0
```
or using `pip` via
```
pip install pymor==2023.1.0
```

Then run one of the experiments, e.g. by
```
cd Code
python Main_Reaction.py
```


## Organization of the repository

The repository contains the directory [`Code/`](https://github.com/michikartmann/adaptive_trrb_for_parameter_identification/tree/main/Code), which contains the source code written in the framework of the package [pyMOR](https://pymor.org). The code consists of the main files

* `Main_Reaction.py`: the main file for the reconstruction of the reaction coefficient in Sections 4.2 and 4.4,
* `Main_Diffusion.py`: the main file for the reconstruction of the diffusion coefficient in Sections 4.3 and 4.5,
* `Main_ErrorEstimatorCaseStudy.py`: the main file for comparing different strategies for evaluating the error estimator in Section 4.6.

The modeling and discretizations of the problems as well as the model reduction code are written in a [pyMORish way](https://docs.pymor.org/2023-1-0/technical_overview.html) with the following main components

* `problems.py`: contains the implementation of the analytical problems,
* `discretizer.py`: discretizes the analytical problem to obtain a full-order model (FOM),
* `model.py`: contains the implementation of the full-order or reduced-order model (ROM),
* `reductor.py`: reduces the full-order model to obtain a reduced-order model,

Moreover, the following files contain the code for the variants of the iteratively regularized gauß newton methods (IRGNM)

* `IRGNM.py:` contains the implementation of the FOM IRGNM (see Section 2.2) and the parameter space reduced Qr IRGNM (see Section 3.1),
* `Qr_Vr_TR_IRGNM.py:` contains the implementation of the parameter and state space reduced Qr-Vr TR IRGNM (see Section 3.3).

Further, there is the file `helpers.py` for all other helper functionality. We also provide empty directories [`Code/Diffusion_Plots/`](https://github.com/michikartmann/adaptive_trrb_for_parameter_identification/tree/main/Code/Diffusion_Plots) and [`Code/Reaction_Plots/`](https://github.com/michikartmann/adaptive_trrb_for_parameter_identification/tree/main/Code/Reaction_Plots), where the data is saved after calling the corresponding main files.

## Questions

If there are questions of any kind, don't hesitate to get in touch with us at <michael.kartmann@uni-konstanz.de> or <tim.keil@wwu.de>.
