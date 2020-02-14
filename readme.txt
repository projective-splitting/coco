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

You will need to have numpy and scipy installed. To run this code, simply pull this repository to an appropriate directory and run
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
PS1f_bt() - one forward step projective splitting with backtracking
PS2f_bt() - two forward step projective splitting with backtracking
PS1f_bt_comp()     - one forward step projective splitting  for composite problems, i.e. some of the terms in the objective are composed with a
                     linear operator.
                     For convenience we implemented this in a separate function.
PS2f_bt_comp()     - two forward step projective splitting  for composite problems, i.e. some of the terms in the objective are composed with a
                     linear operator.
                     For convenience we implemented this in a separate function.
cpBT()             - Chambolle-Pock primal dual splitting back tracking variant, from the paper:
                     Malitsky, Y., Pock, T.: A first-order primal-dual algorithm with linesearch. SIAM Journal on Optimization 28(1), 411–432 (2018)
Tseng_product()    - Tseng-pd, from the paper Combettes, P.L., Pesquet, J.C.: Primal-dual splitting algorithm for solving inclusions
                     with mixtures of composite, Lipschitzian, and parallel-sum type monotone operators.
                     Set-Valued and variational analysis 20(2), 307–330 (2012)
for_reflect_back() - frb-pd, from the paper XX

Additional utilities for group logistic regression are defined in group_lr.py.
