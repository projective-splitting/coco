import sys

path2projSplitFit = '../projSplitFit/'
sys.path.append(path2projSplitFit)
sys.path.append('paramTune/')

try:
    import projSplitFit as ps
    import regularizers
    import lossProcessors as lp
except:
    print("To use all of the available features, you may need to install projSplitFit package from here: ")
    print("https://github.com/1austrartsua1/projSplitFit")


import scipy.sparse.linalg as sl

from utils import setUpRareFeatureProblem

import numpy as np
import algorithms as algo
import time
from matplotlib import pyplot as plt
import scipy.sparse as sp

from scipy.sparse.linalg import norm as sparse_norm

def getSparsity(gamma,H,plot):
    gammaNNZ = sum(abs(gamma)>0.0)
    print(f"gammaNNZ = {gammaNNZ}")
    beta = H.dot(gamma[1:])
    betaNNZ = sum(abs(beta)>0.0)
    print(f"betaNNZ = {betaNNZ}")
    if plot:
        plt.plot(gamma)
        plt.title('gamma')
        plt.show()
        plt.plot(beta)
        plt.title('beta')
        plt.show()

tAbsoluteStart = time.time()


path2data = 'data/trip_advisor/'

maxIter = 20000 

getOld = True
saveResults = True

run2fg = True
run1f = True
run2f=True
runCP = True
runFRB = True
runTseng=True

#np.random.seed(1)

loss = "log"
lams =        [1e-4]
#gamma2fembeds=[1e-5 , 1e-6  , 0.001,1.0]
#gamma2fembeds=[1e-6  , 0.001,1.0]
gamma2fs = [0.0001]
gamma1fs=[0.0001 , 0.0001  , 0.001,1.0]
gammabgs =    [0.0001 , 0.0001  , 0.0001,1.0]
betacps =     [1e6,1e6,10.0,1.0]
tsengs = [1.0,1.0,10.0,100.0]
frbs = [1.0,1.0,100.0,100.0]

#gamma1f_reg_embeds = [0.1,0.1,0.0001]




for i in range(len(lams)):

    lam = lams[i]
    print("\n\n")
    print(f"trying lam = {lam}")
    print(f"loss = {loss}")
    print("\n")

    mu = 0.5

    theGrad,G,Gt,theProx1,theProx2,proxg,proxfstar_4_tseng,theFunc,hfunc,theGradSmart,d,p,X,H,y\
        = setUpRareFeatureProblem(lam,mu,path2data,loss=loss)


    #print("1f")
    #init = algo.InitPoint([],np.zeros(d),np.zeros(d),np.zeros(p))
    #out1f = algo.PS1f_bt_comp(init,maxIter,G,theProx1,theProx2,
    #                      theGrad,Gt,theFunc,gamma = gamma1f,equalRhos=False,verbose=False)


    if runTseng :
        print("Tseng")
        init = algo.InitPoint([],[],np.zeros(d),np.zeros(p))
        outtseng = algo.tseng_product(theFunc, proxfstar_4_tseng, proxg, theGrad, init,verbose=True,
                                      getFuncVals=True,iter=maxIter//2, gamma1=tsengs[i],
                                      gamma2=tsengs[i],G=G,Gt=Gt,historyFreq=20)



    if runFRB :
        print("FRB")
        outfrb = algo.for_reflect_back(theFunc,proxfstar_4_tseng,proxg,theGrad,init,iter=maxIter,
                               gamma0=frbs[i],gamma1=frbs[i],G=G,Gt=Gt,verbose=True,
                               getFuncVals=True,historyFreq=20)


    if runCP :
        print("cp")
        stepIncAmount = 1.0
        #stepIncAmount = 1.1
        betacp = betacps[i]
        init = algo.InitPoint(np.zeros(d),np.zeros(p),[],[])
        outcp = algo.cpBT(theProx2, theGradSmart, proxg, theFunc, hfunc, init, iter=maxIter,
                      beta=betacps[i],stepInc=stepIncAmount,K=Gt,Kstar=G,verbose=True,historyFreq=20)
        print("cp-bt TOTAL running time: "+str(outcp.times[-1]))
        print("================")



    if loss == "log":
        loss2use = "logistic"
        backProcess = lp.BackwardLBFGS()
    else:
        loss2use = 2
        backProcess = lp.BackwardCG()

    embed = False


    if run2f :
        print("2f")
        gamma2f = gamma2fs[i]
        t0 = time.time()
        psObj = ps.ProjSplitFit(gamma2f)

        if embed:
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=lp.Forward2Backtrack(),
                      embed=regularizers.L1(scaling=(1-mu)*lam))
        else:
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=lp.Forward2Backtrack())
            psObj.addRegularizer(regularizers.L1(scaling=(1-mu)*lam),linearOp=H)

        (nbeta,ngamma) = H.shape
        shape = (ngamma-1,ngamma)
        G_for_ps = sl.LinearOperator(shape,matvec=lambda x: x[:-1],
                                     rmatvec = lambda x : np.concatenate((x,np.array([0]))))
        psObj.addRegularizer(regularizers.L1(scaling = mu*lam),linearOp=G_for_ps)
        psObj.run(nblocks=1,maxIterations=maxIter,verbose=True,keepHistory=True,historyFreq=20,
                          primalTol=0.0,dualTol=0.0)
        f_ps2f = psObj.getHistory()[0]
        t_ps2f = psObj.getHistory()[1]
        t1 = time.time()
        print(f"ps2f total running time {t1-t0}")



    if run1f :
        print("1f")
        gamma1f = gamma1fs[i]
        t0 = time.time()
        psObj = ps.ProjSplitFit(gamma1f)
        if embed:
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=lp.Forward1Backtrack(),
                      embed=regularizers.L1(scaling=(1-mu)*lam))
        else:
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=lp.Forward1Backtrack())
            psObj.addRegularizer(regularizers.L1(scaling=(1-mu)*lam),linearOp=H)

        (nbeta,ngamma) = H.shape
        shape = (ngamma-1,ngamma)
        G_for_ps = sl.LinearOperator(shape,matvec=lambda x: x[:-1],
                                     rmatvec = lambda x : np.concatenate((x,np.array([0]))))
        psObj.addRegularizer(regularizers.L1(scaling = mu*lam),linearOp=G_for_ps)
        psObj.run(nblocks=1,maxIterations=maxIter,verbose=True,keepHistory=True,historyFreq=20,
                          primalTol=0.0,dualTol=0.0)
        f_ps1f = psObj.getHistory()[0]
        t_ps1f = psObj.getHistory()[1]
        t1 = time.time()
        print(f"ps1f total running time {t1-t0}")
        #getSparsity(psObj.getSolution(),H,plot=True)

    runPSB_g = False
    if runPSB_g :
        print("psb_g")
        gammabg = gammabgs[i]
        t0 = time.time()
        psObj = ps.ProjSplitFit(gammabg)
        psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=backProcess)
        psObj.addRegularizer(regularizers.L1(scaling=(1-mu)*lam),linearOp=H)
        (nbeta,ngamma) = H.shape
        shape = (ngamma-1,ngamma)
        G_for_ps = sl.LinearOperator(shape,matvec=lambda x: x[:-1],
                                     rmatvec = lambda x : np.concatenate((x,np.array([0]))))
        psObj.addRegularizer(regularizers.L1(scaling = mu*lam),linearOp=G_for_ps)
        psObj.run(nblocks=10,maxIterations=maxIterG,verbose=True,keepHistory=True,historyFreq=100,
                          primalTol=0.0,dualTol=0.0)
        fpsbg = psObj.getHistory()[0]
        tpsbg = psObj.getHistory()[1]
        t1 = time.time()
        print(f"psb_g total running time {t1-t0}")


    embed = False
    if run2fg :
        print("ps2fg")
        gamma2f = gamma2fs[i]
        t0 = time.time()
        psObj = ps.ProjSplitFit(gamma2f)
        proc = lp.Forward2Backtrack()
        if embed:
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=proc,
                embed=regularizers.L1(scaling=(1-mu)*lam))
        else:
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=proc)
            psObj.addRegularizer(regularizers.L1(scaling=(1-mu)*lam),linearOp=H)


        (nbeta,ngamma) = H.shape
        shape = (ngamma-1,ngamma)
        G_for_ps = sl.LinearOperator(shape,matvec=lambda x: x[:-1],
                                     rmatvec = lambda x : np.concatenate((x,np.array([0]))))
        regStepsize=1.0
        blockAct = "greedy"
        psObj.addRegularizer(regularizers.L1(scaling = mu*lam,step=regStepsize),linearOp=G_for_ps)
        psObj.run(nblocks=10,maxIterations=maxIter,verbose=True,keepHistory=True,historyFreq=100,
                          primalTol=0.0,dualTol=0.0,blockActivation=blockAct)

        history_2fg = psObj.getHistory()
        z_2fg = psObj.getSolution()


        t1 = time.time()
        print(f"ps2fg total running time {t1-t0}")


    run1falt = False
    if run1falt:
        #this version of 1f does not use the second linear operator G_for_ps
        #but instead defines the regularizer in a different way.
        #The reason for G_for_ps is to deal with the intercept.

        gamma = gamma1f_reg_embeds[i]
        t0 = time.time()
        psObj = ps.ProjSplitFit(gamma)
        psObj.addData(X,y,2,linearOp=H,normalize=False,process=lp.Forward1Backtrack(),
                      embed=regularizers.L1(scaling=(1-mu)*lam))

        def prox(gamma,sigma):
            temp = (gamma[:-1]>sigma)*(gamma[:-1]-sigma)
            temp += (gamma[:-1]<-sigma)*(gamma[:-1]+sigma)
            return np.concatenate((temp,np.array([gamma[-1]])))

        def val(gamma):
            return np.linalg.norm(gamma[:-1],1)

        psObj.addRegularizer(regularizers.Regularizer(prox,val,scaling=lam*mu))
        psObj.run(nblocks=1,maxIterations=maxIter,verbose=False,keepHistory=True,historyFreq=1,
                  primalTol=0.0,dualTol=0.0)
        f1f_alt = psObj.getHistory()[0]
        t1f_alt = psObj.getHistory()[1]
        t1 = time.time()
        print(f"1f alt total running time {t1-t0}")



    if saveResults:
        import pickle

        if getOld:
            if loss == "log":
                with open('results/bnl_results_log_'+str(lam),'rb') as file:
                    cache = pickle.load(file)
            else:
                with open('results/bnl_results_'+str(lam),'rb') as file:
                    cache = pickle.load(file)
        else:

            cache = {}

        if run1f:
            cache['f_ps1f'] = f_ps1f
            cache['t_ps1f'] = t_ps1f

        if runFRB:
            cache['outfrb']= outfrb
        if runTseng:
            cache['outtseng']=outtseng
        if run2fg:
            if embed:
                if blockAct == "greedy":
                    cache['history_2fg']=history_2fg
                    cache['z_2fg'] = z_2fg
                elif blockAct == "random":
                    cache['history_2fr']=history_2fg
                    cache['z_2fr'] = z_2fg
                elif blockAct == "cyclic":
                    cache['history_2fc']=history_2fg
                    cache['z_2fc'] = z_2fg

            else:
                cache['history_2fg']=history_2fg
                cache['z_2fg'] = z_2fg


        #cache['t_ps2fembed_c']=t_ps2fembed_c
        #cache['f_ps2fembed_c']=f_ps2fembed_c
        if runPSB_g:
            cache['f_psbg']=fpsbg
            cache['t_psbg']=tpsbg
        if runCP:
            cache['outcp'] = outcp

        if run2f:
            cache['f_ps2f'] = f_ps2f
            cache['t_ps2f'] = t_ps2f


        if loss == "log":
            with open('results/bnl_results_log_'+str(lam),'wb') as file:
                pickle.dump(cache,file)

        else:
            with open('results/bnl_results_'+str(lam),'wb') as file:
                pickle.dump(cache,file)


tEnd = time.time()
print(f"total running time {tEnd-tAbsoluteStart}")
