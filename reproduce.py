'''
Run this script to reproduce the specific experiments in the paper. Follow the directions in the prompt.
'''

import os
import sys
if sys.version_info[0]==2:
    which2run = raw_input("Enter which problem to solve: portfolio or group_lr >>> ")
else:
    which2run = input("Enter which problem to solve: portfolio or group_lr >>> ")

print('You selected '+ which2run)
if which2run=='portfolio':
    print('WARNING: running time on our machine for this experiment was about 2.5 minutes,')
    print('and it may be longer for you.')
    options = '1,2,3,4'
elif which2run=='group_lr':
    print('WARNING: running time on our machine for this experiment was about 5 minutes,')
    print('and it may be longer for you.')
    options = '1,2,3,4,5,6'
elif which2run=='rare_feature':
    print("Sorry, rare feature is no longer supported in this experiment. Previously there was an error in the rare feature code.")
    print("When this was fixed, the performance of our method was similar to others so we simply removed this experiment from the paper.")
    exit() 
else:
    print('You did not enter a correct problem, exiting')
    exit()
whichExp = int(input("Enter which experiment to run from {"+(options)+"} >>> "))
print('You selected '+str(whichExp))

if which2run=='portfolio':
    if whichExp==1:
        os.system(' python run_portfolio.py --dimension 10000')
    elif whichExp==2:
        os.system('python run_portfolio.py --dimension 10000 --deltar 0.8')
    elif whichExp==3:
        os.system('python run_portfolio.py --dimension 10000 --deltar 1.0 --gamma1f 0.5 --gamma2f 10 \
                                --betacp 0.5 --gammafrb 10')
    elif whichExp==4:
        os.system('python run_portfolio.py --dimension 10000 --deltar 1.5 --gamma1f 5 --gamma2f 10 \
                                --betacp 0.5 --gammafrb 10 --gammatg 10')
    else:
        print('You did not enter an experiment number 1, 2, 3, or 4.  Exiting.')
elif which2run=='group_lr':
    if whichExp==1:
        os.system('python run_group_lr.py --lam1 0.05 --lam2 0.05 --dataset breastCancer \
                               --gamma1f 0.05 --iter 3000')
    elif whichExp==2:
        os.system('python run_group_lr.py --lam1 0.5 --lam2 0.5 --dataset breastCancer \
                               --gamma1f 1e2  --gamma2f 1e2 --gammatg 1e5   \
                               --gammafrb 1e5 --betacp 1e-3 --iter 5000 --initial_stepsize 1.0')
    elif whichExp==3:
        os.system('python run_group_lr.py --lam1 0.85 --lam2 0.85 --dataset breastCancer \
                               --gamma1f 1e2  --gamma2f 1e5 --gammatg 1e5     \
                               --gammafrb 1e5 --betacp 1e-4 --iter 3000')
    elif whichExp==4:
        os.system('python run_group_lr.py --lam1 0.1 --lam2 0.1 --gamma1f 0.1 --gammatg 1e4 \
                               --gammafrb 1e4 --betacp 1e-4 --iter 3000')
    elif whichExp==5:
        os.system('python run_group_lr.py --lam1 0.5 --lam2 0.5 --gammatg 1e6 --gammafrb 1e6 --iter 3000 --betacp 1e-3')
    elif whichExp==6:
        os.system('python run_group_lr.py --lam1 1.0 --lam2 1.0 --gammatg 1e6 --gammafrb 1e6\
                               --betacp 1e-5 --iter 3000')
    else:
        print('You did not enter an experiment number 1, 2, 3, 4, 5, or 6.  Exiting.')
else:
    print("enter group_lr or portfolio for the experiment")
