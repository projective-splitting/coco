
This is the python code for running the experiments in Section 6 of

https://arxiv.org/pdf/1902.09025.pdf

"Single-forward-step projective splitting: Exploiting cocoercivity", Johnstone, Patrick R., and Jonathan Eckstein.
arXiv preprint arXiv:1902.09025 (2019).

The trip advisor data were kindly shared with us by Xiaohan Yan and Jacob Bien
Yan, X., Bien, J.: Rare Feature Selection in High Dimensions. arXiv preprint
arXiv:1803.06675 (2018).

The trip advisor data in R and Python format is also available at
https://github.com/yanxht/TripAdvisorData
and we have included it in Python format in our repo here in
data/trip_advisor

You will need to install numpy and scipy.

To install, simply pull this repository to an appropriate directory and run
python scripts from that directory.

To run the portfolio optimization experiment:
$python run_portfolio.py

To run the group sparse logistic regression experiment:
$python run_group_lr.py

To run the rare feature selection experiment:
$python run_rare_feature.py

various parameters can be set from the command line using the argpass package.
To find out which parameters can be set, run
$python run_portfolio.py -h
$python run_group_lr.py -h
$python run_rare_feature.py -h

The algorithms are implemented in algorithms.py, including
PS1f_bt() - one forward step projective splitting
PS2f_bt() - two forward step projective splitting
PS1f_bt_comp() - one forward step projective splitting  for composite opt problems.
                 For convenience we implemented this in a separate function.
PS2f_bt_comp() - two forward step projective splitting  for composite opt problems.
                 For convenience we implemented this in a separate function.
cpBT() - Chambolle-Pock primal dual splitting back tracking variant
Tseng_product() - Tseng-pd
for_reflect_back() - frb-pd

Additional utilities for group logistic regression are defined in group_lr.py. 
