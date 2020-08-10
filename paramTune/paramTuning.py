
import sys
sys.path.append('../')
sys.path.append('../../sphinx_projSplitFit/projSplitFit/')

import projSplit as ps
import regularizers
import lossProcessors as lp
import scipy.sparse.linalg as sl

from utils import setUpRareFeatureProblem
import algorithms as algo
import time

#from matplotlib import pyplot as plt
import numpy as np

'''
Notes
july28 4pm: cp is doing much better than everyone else on 1e-5. May have to scratch that example
for a different val of lambda. Tseng and frb are not competitive so best to just tune over
1f, 2f, and cp. It took 30min to run 1000 iters for all algs.

4:30pm it seems that lam=1e-3 offers pretty good competition between 1f and cp. Both pretty similar.

4:50pm lam=1e-2 it seems that cp is slightly better.

5:20pm lam =0.1 it also seems cp is doing better.

5:40pm setting equalRhos equal to 1 does not help!

11pm : it appears CP is doing better on all examples, not by a huge margin, but slightly better.

july 29 4pm: 1fcomp_g, which is using ProjSplitFit with greedy, does much better for
any lam < 0.01. Like an order magnitude lower function val. Really killing it.
For lam >= 0.1, performance is comparable to CP, or CP is slightly better. So at least
there is an easy story for the MAPR paper.

july 30 6pm : paramTuned psbg and its performance is basically identical to ps2fg. Like
almost exactly the same. So that's a thing. Still, we are getting such a big benefit from
greedy that this definitely deserves to be in the paper.

'''
tstart = time.time()

startLam = -8
endLam = -2
numTrialPointsLam = 4


lambdas2check = np.logspace(startLam,endLam,numTrialPointsLam)
lambdas2check = [0.01]

path2data = '../data/trip_advisor/'

loss = "log"
for lam in lambdas2check:
    print("\n\n\n")
    print("================")
    print(f"checking {lam}")
    print("================")
    print("\n\n\n")
    mu = 0.5

    theGrad,G,Gt,theProx1,theProx2,proxg,proxfstar_4_tseng,theFunc,hfunc,theGradSmart,d,p,X,H,y\
        = setUpRareFeatureProblem(lam,mu,path2data,loss=loss)


    maxIter = 10000
    algos2run = ['ps2fembed_g']
    numTrialPoints = 13
    start = -6
    end = 6
    trials = np.logspace(start,end,numTrialPoints)
    results = {}
    i = 0

    for tuned in trials:
        print(f"tuned = {tuned}")

        if False:
            init = algo.InitPoint([],np.zeros(d),np.zeros(d),np.zeros(p))
            out1f = algo.PS1f_bt_comp(init,maxIter,G,theProx1,theProx2,
                                  theGrad,Gt,theFunc,gamma = tuned,
                                  equalRhos=False,getFuncVals=False,verbose=False)
            results[(i,'1f')] =  out1f.finalFuncVal

        if False:
            init = algo.InitPoint([],[],np.zeros(d),np.zeros(p))
            out2f = algo.PS2f_bt_comp(init,int(0.5*maxIter),G,theProx1,theProx2,theGrad,Gt,
                              theFunc,gamma=tuned,equalRhos=False,getFuncVals=False,verbose=False)
            results[(i,'2f')] =  out2f.finalFuncVal

        if False:
            init = algo.InitPoint(np.zeros(d),np.zeros(p),[],[])
            outcp = algo.cpBT(theProx2, theGradSmart, proxg, theFunc, hfunc, init, iter=maxIter,
                          beta=tuned,stepInc=1.1,K=Gt,Kstar=G,verbose=False,getFuncVals=False)
            results[(i,'cp')] =  outcp.finalFuncVal


        if False:
            init = algo.InitPoint([],[],np.zeros(d),np.zeros(p))
            outtseng = algo.tseng_product(theFunc, proxfstar_4_tseng, proxg, theGrad, init,verbose=False,getFuncVals=False,
                                      iter=maxIter, gamma1=tuned,gamma2=tuned,G=G,Gt=Gt)
            results[(i,'tseng')] =  outtseng.finalFuncVal

        if False:
            outfrb = algo.for_reflect_back(theFunc,proxfstar_4_tseng,proxg,theGrad,init,iter=maxIter,
                                       gamma0=tuned,gamma1=tuned,G=G,Gt=Gt,verbose=False,getFuncVals=False)
            results[(i,'frb')] =  outfrb.finalFuncVal

        if loss == "log":
            loss2use = "logistic"
        else:
            loss2use = 2

        gamma = 1.0
        if True :
            psObj = ps.ProjSplitFit(gamma)
            proc = lp.Forward2Backtrack()
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=proc,
                          embed=regularizers.L1(scaling=(1-mu)*lam))
            (nbeta,ngamma) = H.shape
            shape = (ngamma-1,ngamma)
            G_for_ps = sl.LinearOperator(shape,matvec=lambda x: x[:-1],rmatvec = lambda x : np.concatenate((x,np.array([0]))))
            psObj.addRegularizer(regularizers.L1(scaling = mu*lam,step=tuned),linearOp=G_for_ps)
            psObj.run(nblocks=10,maxIterations=maxIter,verbose=False,keepHistory=False,
                      primalTol=0.0,dualTol=0.0,blockActivation="greedy")
            results[(i,'ps2fembed_g')] = psObj.getObjective()


        if False:
            psObj = ps.ProjSplitFit(gamma)
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=lp.Forward1Backtrack(),
                          embed=regularizers.L1(scaling=(1-mu)*lam))
            (nbeta,ngamma) = H.shape
            shape = (ngamma-1,ngamma)
            G_for_ps = sl.LinearOperator(shape,matvec=lambda x: x[:-1],rmatvec = lambda x : np.concatenate((x,np.array([0]))))
            psObj.addRegularizer(regularizers.L1(scaling = mu*lam),linearOp=G_for_ps)
            psObj.run(nblocks=1,maxIterations=maxIter,verbose=False,keepHistory=False,
                      primalTol=0.0,dualTol=0.0)
            results[(i,'ps1fembed')] = psObj.getObjective()

        if False:
            psObj = ps.ProjSplitFit(gamma)
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=lp.BackwardLBFGS())
            psObj.addRegularizer(regularizers.L1(scaling=(1-mu)*lam),linearOp=H)
            (nbeta,ngamma) = H.shape
            shape = (ngamma-1,ngamma)
            G_for_ps = sl.LinearOperator(shape,matvec=lambda x: x[:-1],rmatvec = lambda x : np.concatenate((x,np.array([0]))))
            psObj.addRegularizer(regularizers.L1(scaling = mu*lam),linearOp=G_for_ps)
            psObj.run(nblocks=10,maxIterations=maxIter,verbose=True,keepHistory=False,
                      primalTol=0.0,dualTol=0.0)
            results[(i,'psb_g')] = psObj.getObjective()



        i += 1

    doWrite = True
    if doWrite:
        if loss == "log":
            sys.stdout = open("results_"+"log_"+str(lam), "a")
        else:
            sys.stdout = open("results_"+str(lam), "a")

        for algos in algos2run:


            print("====================")
            print("====================")
            print(f"results for {algos}")
            print("trials : results")
            for i in range(len(trials)):
                print(f"{trials[i]} : {results[(i,algos)]}")

        sys.stdout.close()
        sys.stdout = sys.__stdout__

tend = time.time()
print(f"TOTAL time {tend-tstart}")
