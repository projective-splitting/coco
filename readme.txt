This is the python code for running the experiments in Section 6 of

https://arxiv.org/pdf/1902.09025.pdf

[1] Johnstone, P. R., and Eckstein, J.: "Single-forward-step projective
splitting: Exploiting cocoercivity", arXiv preprint arXiv:1902.09025 (2019).

The TripAdvisor data were kindly shared with us by Xiaohan Yan and Jacob Bien.

[2] Yan, X., and Bien, J.: Rare Feature Selection in High Dimensions. arXiv
preprint arXiv:1803.06675 (2018).

The trip advisor data in R and Python format is also available at
https://github.com/yanxht/TripAdvisorData.  In this repository, it is in Python
format in the directory data/trip_advisor

In addition to implementations of several variants of projective splitting,
this repository also includes implementations of
- adaptive three-operator splitting,
  [3] Pedregosa, F., Gidel, G.: Adaptive three-operator splitting. arXiv
      preprint arXiv:1804.02339 (2018)
- Chambolle-Pock primal dual spltting,
  [4] Malitsky, Y., Pock, T.: A first-order primal-dual algorithm with
      linesearch. SIAM Journal on       Optimization 28(1):411–432 (2018)
- Tseng's method applied to the primal-dual inclusion,
  [5] Combettes, P.L., Pesquet, J.C.: Primal-dual splitting algorithm for
      solving inclusions with mixtures of composite, Lipschitzian, and
      parallel-sum type monotone operators.  Set-Valued and Variational
      Analysis volume 20(2):307–330 (2012)
- forward-reflected-backward splitting,
  [6] Malitsky, Y., Tam, M.K.: A forward-backward splitting method for
      monotone inclusions without cocoercivity. arXiv preprint
      arXiv:1808.04162 (2018)

To run the experiments in [1]:
1. Ensure that Python and the Python packages numpy, scipy, matplotlib,
   and argpass are installed on your system.
2. Navigate to the directory you would like to save the data and issue
   the shell command
   $ git clone https://github.com/projective-splitting/coco.git
   Or, if you're on a Windows or Mac system, click download, download the
   zip file, and unzip it in the directory in which you want locate the
   code
3. Run Python from that directory as follows:

    To run the portfolio optimization experiment:
    $ python run_portfolio.py

    To run the group sparse logistic regression experiment:
    $ python run_group_lr.py

    To run the rare feature selection experiment:
    $ python run_rare_feature.py

  These scripts will run one of the parameter settings studied in the paper.
  Other parameter values may be tested by modifying the parameters from
  the command line.

  Whenever a plot is displayed, the matplotlib package script will wait for
  you to close the plot before proceeding (you may save it first).

Various parameters can be set from the command line.  To find out which
parameters can be set, run the commands

$ python run_portfolio.py -h
$ python run_group_lr.py -h
$ python run_rare_feature.py -h

Following are the specific commands required to reproduce each of the scripts in
[1].
  - portfolio:
    1: python run_portfolio.py --dimension 10000
    2: python run_portfolio.py --dimension 10000 --deltar 0.8
    3: python run_portfolio.py --dimension 10000 --deltar 1.0 --gamma1f 0.5 --gamma2f 10 \
                                --betacp 0.5 --gammafrb 10
    4: python run_portfolio.py --dimension 10000 --deltar 1.5 --gamma1f 5 --gamma2f 10 \
                                --betacp 0.5 --gammafrb 10 --gammatg 10

  - group logistic regression
    1: python run_group_lr.py --lam1 0.05 --lam2 0.05 --dataset breastCancer \
                               --gamma1f 0.05
    2: python run_group_lr.py --lam1 0.5 --lam2 0.5 --dataset breastCancer \
                               --gamma1f 1e2  --gamma2f 1e2 --gammatg 1e5   \
                               --gammafrb 1e5 --betacp 1e-3
    3: python run_group_lr.py --lam1 0.85 --lam2 0.85 --dataset breastCancer \
                               --gamma1f 1e2  --gamma2f 1e5 --gammatg 1e5     \
                               --gammafrb 1e5 --betacp 1e-4
    4: python run_group_lr.py --lam1 0.1 --lam2 0.1 --gamma1f 0.1 --gammatg 1e4 \
                               --gammafrb 1e4 --betacp 1e-4
    5: python run_group_lr.py --lam1 0.5 --lam2 0.5 --gammatg 1e6 --gammafrb 1e6\
    6: python run_group_lr.py --lam1 1.0 --lam2 1.0 --gammatg 1e6 --gammafrb 1e6\
                               --betacp 1e-5

  - rare feature selection
    1: python run_rare_feature.py --gamma1f 1.0
    2: python run_rare_feature.py --lam 1e-2 --gamma2f 1e1 --betacp 1e-3\
                                   --gammatg 1e4 --gammafrb 1e4
    3: python run_rare_feature.py --lam 1e-1 --gamma1f 1e4 --gamma2f 1e5 \
                                   --betacp 1e-7 --gammatg 1e6 --gammafrb 1e6

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
