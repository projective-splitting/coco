import sys
sys.path.append('../sphinx_projSplitFit/projSplitFit/')
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

maxIterG = 50000
maxIterCP = 10000

np.random.seed(1)
loss = "log"
lams =        [1e-4]
#gamma2fembeds=[1e-5 , 1e-6  , 0.001,1.0]
#gamma2fembeds=[1e-6  , 0.001,1.0]
gamma2fembeds = [0.0001]
#tsengs = [100.0]
#frbs = [100.0]
gamma1fembeds=[0.0001 , 0.0001  , 0.001,1.0]
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

    runTseng=True
    if runTseng :
        print("Tseng")
        init = algo.InitPoint([],[],np.zeros(d),np.zeros(p))
        outtseng = algo.tseng_product(theFunc, proxfstar_4_tseng, proxg, theGrad, init,verbose=True,
                                      getFuncVals=True,iter=maxIterCP//2, gamma1=tsengs[i],
                                      gamma2=tsengs[i],G=G,Gt=Gt,historyFreq=20)


    runFRB = True
    if runFRB :
        print("FRB")
        outfrb = algo.for_reflect_back(theFunc,proxfstar_4_tseng,proxg,theGrad,init,iter=maxIterCP,
                               gamma0=frbs[i],gamma1=frbs[i],G=G,Gt=Gt,verbose=True,
                               getFuncVals=True,historyFreq=20)

    runCP = True
    if runCP :
        print("cp")
        stepIncAmount = 1.0
        #stepIncAmount = 1.1
        betacp = betacps[i]
        init = algo.InitPoint(np.zeros(d),np.zeros(p),[],[])
        outcp = algo.cpBT(theProx2, theGradSmart, proxg, theFunc, hfunc, init, iter=maxIterCP,
                      beta=betacps[i],stepInc=stepIncAmount,K=Gt,Kstar=G,verbose=True,historyFreq=20)
        print("cp-bt TOTAL running time: "+str(outcp.times[-1]))
        print("================")


    if loss == "log":
        loss2use = "logistic"
        backProcess = lp.BackwardLBFGS()
    else:
        loss2use = 2
        backProcess = lp.BackwardCG()

    run2fembed=False
    if run2fembed :
        print("2f_embed")
        gamma2fembed = gamma2fembeds[i]
        t0 = time.time()
        psObj = ps.ProjSplitFit(gamma2fembed)

        psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=lp.Forward2Backtrack(),
                      embed=regularizers.L1(scaling=(1-mu)*lam))
        #psObj.addRegularizer(regularizers.L1(scaling=(1-mu)*lam),linearOp=H)
        (nbeta,ngamma) = H.shape
        shape = (ngamma-1,ngamma)
        G_for_ps = sl.LinearOperator(shape,matvec=lambda x: x[:-1],
                                     rmatvec = lambda x : np.concatenate((x,np.array([0]))))
        psObj.addRegularizer(regularizers.L1(scaling = mu*lam),linearOp=G_for_ps)
        psObj.run(nblocks=1,maxIterations=maxIterCP,verbose=True,keepHistory=True,historyFreq=20,
                          primalTol=0.0,dualTol=0.0)
        f_ps2fembed = psObj.getHistory()[0]
        t_ps2fembed = psObj.getHistory()[1]
        t1 = time.time()
        print(f"ps2fembed total running time {t1-t0}")


    if False :
        print("1f_embed")
        gamma1fembed = gamma1fembeds[i]
        t0 = time.time()
        psObj = ps.ProjSplitFit(gamma1fembed)
        psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=lp.Forward1Backtrack(),
                      embed=regularizers.L1(scaling=(1-mu)*lam))
        #psObj.addRegularizer(regularizers.L1(scaling=(1-mu)*lam),linearOp=H)
        (nbeta,ngamma) = H.shape
        shape = (ngamma-1,ngamma)
        G_for_ps = sl.LinearOperator(shape,matvec=lambda x: x[:-1],
                                     rmatvec = lambda x : np.concatenate((x,np.array([0]))))
        psObj.addRegularizer(regularizers.L1(scaling = mu*lam),linearOp=G_for_ps)
        psObj.run(nblocks=1,maxIterations=maxIter,verbose=True,keepHistory=True,historyFreq=20,
                          primalTol=0.0,dualTol=0.0)
        f_ps1fembed = psObj.getHistory()[0]
        t_ps1fembed = psObj.getHistory()[1]
        t1 = time.time()
        print(f"ps1fembed total running time {t1-t0}")
        getSparsity(psObj.getSolution(),H,plot=True)

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

    run2f_embed_g = True
    noEmbed = True
    if run2f_embed_g :
        print("ps2f_embed_g")
        gamma2fembed = gamma2fembeds[i]
        t0 = time.time()
        psObj = ps.ProjSplitFit(gamma2fembed)
        proc = lp.Forward2Backtrack()
        if noEmbed :
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=proc)
            psObj.addRegularizer(regularizers.L1(scaling=(1-mu)*lam),linearOp=H)
        else:
            psObj.addData(X,y,loss2use,linearOp=H,normalize=False,process=proc,
                        embed=regularizers.L1(scaling=(1-mu)*lam))

        (nbeta,ngamma) = H.shape
        shape = (ngamma-1,ngamma)
        G_for_ps = sl.LinearOperator(shape,matvec=lambda x: x[:-1],
                                     rmatvec = lambda x : np.concatenate((x,np.array([0]))))
        regStepsize=1.0
        blockAct = "greedy"
        psObj.addRegularizer(regularizers.L1(scaling = mu*lam,step=regStepsize),linearOp=G_for_ps)
        psObj.run(nblocks=10,maxIterations=maxIterG,verbose=True,keepHistory=True,historyFreq=100,
                          primalTol=0.0,dualTol=0.0,blockActivation=blockAct)

        history_2fg = psObj.getHistory()
        z_2fg = psObj.getSolution()


        t1 = time.time()
        print(f"ps2fembed_g total running time {t1-t0}")


    if False :
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
        f1f_reg_embed = psObj.getHistory()[0]
        t1f_reg_embed = psObj.getHistory()[1]
        t1 = time.time()
        print(f"1f reg embed total running time {t1-t0}")


    saveResults = True
    if saveResults:
        import pickle
        getOld = True
        if getOld:
            if loss == "log":
                with open('saved_results_log_'+str(lam),'rb') as file:
                    cache = pickle.load(file)
            else:
                with open('saved_results_'+str(lam),'rb') as file:
                    cache = pickle.load(file)
        else:

            cache = {}

        #cache['out1f'] = out1f
        #cache['out2f'] = out2f

        if runFRB:
            cache['outfrb']= outfrb
        if runTseng:
            cache['outtseng']=outtseng
        if run2f_embed_g:
            if noEmbed :
                cache['history_2fg_ne']=history_2fg
                cache['z_2fg_ne'] = z_2fg
            else:
                if blockAct == "greedy":
                    cache['history_2fg']=history_2fg
                    cache['z_2fg'] = z_2fg
                elif blockAct == "random":
                    cache['history_2fr']=history_2fg
                    cache['z_2fr'] = z_2fg
                elif blockAct == "cyclic":
                    cache['history_2fc']=history_2fg
                    cache['z_2fc'] = z_2fg

        #cache['t_ps2fembed_c']=t_ps2fembed_c
        #cache['f_ps2fembed_c']=f_ps2fembed_c
        if runPSB_g:
            cache['f_psbg']=fpsbg
            cache['t_psbg']=tpsbg
        if runCP:
            cache['outcp'] = outcp

        if run2fembed:
            cache['f_ps2fembed'] = f_ps2fembed
            cache['t_ps2fembed'] = t_ps2fembed


        if loss == "log":
            with open('saved_results_log_'+str(lam),'wb') as file:
                pickle.dump(cache,file)

        else:
            with open('saved_results_'+str(lam),'wb') as file:
                pickle.dump(cache,file)


tEnd = time.time()
print(f"total running time {tEnd-tAbsoluteStart}")
