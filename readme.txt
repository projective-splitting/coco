This is the python code for running the experiments in Section 6 of

https://arxiv.org/pdf/1902.09025.pdf

[1] "Single-forward-step projective splitting: Exploiting cocoercivity", Johnstone, Patrick R., and Jonathan Eckstein.
arXiv preprint arXiv:1902.09025 (2019).

The trip advisor data were kindly shared with us by Xiaohan Yan and Jacob Bien
Yan, X., Bien, J.: Rare Feature Selection in High Dimensions. arXiv preprint
arXiv:1803.06675 (2018).

The trip advisor data in R and Python format is also available at
https://github.com/yanxht/TripAdvisorData
and we have included it in Python format in our repo here in
data/trip_advisor

In addition to implementations of several variants of projective splitting, this also includes
implementations of
- adaptive three operator splitting,

- Chambolle-Pock primal dual spltting,
  [2] Pedregosa, F., Gidel, G.: Adaptive three-operator splitting. Preprint 1804.02339, ArXiV
    (2018)
  [3] Malitsky, Y., Pock, T.: A first-order primal-dual algorithm with
   linesearch. SIAM Journal on Optimization 28(1), 411–432 (2018)
- Tseng's method applied to the primal-dual inclusion,
  [4] Combettes, P.L., Pesquet, J.C.: Primal-dual splitting algorithm for solving inclusions
  with mixtures of composite, Lipschitzian, and parallel-sum type monotone operators.
- forward-reflected-backward splitting,
  [5] Malitsky, Y., Tam, M.K.: A forward-backward splitting method for monotone inclusions
  without cocoercivity. arXiv preprint arXiv:1808.04162 (2018)

To run the experiments in [1]:
1. Ensure that numpy, scipy, matplotlib, and argpass are installed.
2. Save run_portfolio.py, run_group_lr.py, run_rare_feature.py, algorithms.py, and group_lr.py in the same directory.
3. Save the datasets in a subdirectory of that directory called "data".
(steps 2 and 3 can be accomplished with git pull)
4. Run python from that directory as follows:
  To run the portfolio optimization experiment:
  $python run_portfolio.py

  To run the group sparse logistic regression experiment:
  $python run_group_lr.py

  To run the rare feature selection experiment:
  $python run_rare_feature.py

  This will run one of the parameter settings studied in the paper. Others can be
  ran by modifying the parameters from the command line.

Various parameters can be set from the command line.
To find out which parameters can be set, run
$python run_portfolio.py -h
$python run_group_lr.py -h
$python run_rare_feature.py -h

The algorithms are implemented in algorithms.py, including
PS1f_bt()          - one forward step projective splitting with backtracking
PS2f_bt()          - two forward step projective splitting with backtracking
PS1f_bt_comp()     - one forward step projective splitting  for composite problems, i.e. some of the terms in the objective are composed with a
                     linear operator.
                     For convenience we implemented this in a separate function.
PS2f_bt_comp()     - two forward step projective splitting  for composite problems, i.e. some of the terms in the objective are composed with a
                     linear operator.
                     For convenience we implemented this in a separate function.
adap3op()          - adaptive three operator splitting [2]
cpBT()             - Chambolle-Pock primal dual splitting back tracking variant, [3]
Tseng_product()    - Tseng-pd, [4]
for_reflect_back() - frb-pd, [5]

Additional utilities for group logistic regression are defined in group_lr.py.
