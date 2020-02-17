'''
This file runs the rare feature selection experiment in the paper
"Single Forward Step Projective Splitting: Exploiting Cocoercivity", Patrick R. Johnstone
and Jonathan Eckstein, arXiv:1902.09025.
Various parameters can be set from the command line.
To run: (from the directory containing this file)
$python run_rare_feature.py
This runs with default parameters.
To see what parameters can be set from the command line, run
$python run_rare_feature.py -h
This code has been tested with python2.7 and python3.5.
'''

import numpy as np
import algorithms as algo
import time
from matplotlib import pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as ln
import argparse

tAbsolutestart = time.time()
parser = argparse.ArgumentParser(description='Rare Feature Selection Experiment')

parser.add_argument('--lam',type=float,default=1e-5,dest='lam',
                    help = 'reg parameter',metavar='lam')
parser.add_argument('--mu',type=float,default=0.5,dest='mu',
                    help = 'reg parameter',metavar='mu')
parser.add_argument('--iter',type=int,default=1000,dest='iter',
                    help = 'num iterations, same for all algorithms',metavar='iter')
parser.add_argument('--gamma1f',type=float,default=1e1,dest='gamma1f',
                    help = 'primal-dual tuning parameter for ps1fbt',metavar='gamma1f')
parser.add_argument('--gamma2f',type=float,default=1e2,dest='gamma2f',
                    help = 'primal-dual tuning parameter for ps2fbt',metavar='gamma2f')
parser.add_argument('--gammatg',type=float,default=1e0,dest='gammatg',
                    help = 'primal-dual tuning parameter for tseng-pd',metavar='gammatg')
parser.add_argument('--gammafrb',type=float,default=1e0,dest='gammafrb',
                    help = 'primal-dual tuning parameter for frb-pd',metavar='gammafrb')
parser.add_argument('--betacp',type=float,default=1e-1,dest='betacp',
                    help = 'primal-dual tuning parameter beta for cp-bt',metavar='betacp')

lam = parser.parse_args().lam
mu = parser.parse_args().mu
iter = parser.parse_args().iter
gamma1f = parser.parse_args().gamma1f
gamma2f = parser.parse_args().gamma2f
gammatg = parser.parse_args().gammatg
gammafrb = parser.parse_args().gammafrb
betacp = parser.parse_args().betacp

# the following matrices are loaded in scipy's sparse matrix format '.npz'
#The trip advisor data were kindly shared with us by Xiaohan Yan and Jacob Bien
#Yan, X., Bien, J.: Rare Feature Selection in High Dimensions. arXiv preprint
#arXiv:1803.06675 (2018).

#The trip advisor data in R and Python format is also available at
#https://github.com/yanxht/TripAdvisorData
#and we have included it in Python format in our repo here in
#data/trip_advisor/

S_train = sp.load_npz('data/trip_advisor/S_train.npz') # training matrix
S_test  = sp.load_npz('data/trip_advisor/S_test.npz')  # testing matrix
S_A     = sp.load_npz('data/trip_advisor/S_A.npz')     # this matrix is called H
                                                       # in the the paper
y_train = np.load('data/trip_advisor/y_train.npy')     # training labels
y_test  = np.load('data/trip_advisor/y_test.npy')      # testing labels

print("----------------------------------")
print("----------------------------------")
print("----------------------------------")

(n_train,_) = S_train.shape
(n_test,_) = S_test.shape

y = y_train
S = S_train
n = n_train

print("Adding all ones column to train/test matrix for offset/intercept...")
onesCol = np.ones([n,1])
onesCol = sp.csc_matrix(onesCol)
S = sp.hstack([onesCol,S],format='csc')

onesCol = np.ones([n_test, 1])
onesCol = sp.csc_matrix(onesCol)
S_test = sp.hstack([onesCol, S_test], format='csc')

print("The offset is replicated in gamma.")
print("To H, we append a column and row consisting of all zeros, except for")
print("   a one in the upper left corner")
(p,d) = S_A.shape
zerosCol = np.zeros([p,1])
zerosCol = sp.csc_matrix(zerosCol)
S_A = sp.hstack([zerosCol,S_A],format='csc')
zerosButOneRow = np.zeros([1,d+1])
zerosButOneRow[0]=1.0
zerosButOneRow = sp.csc_matrix(zerosButOneRow)
S_A = sp.vstack([zerosButOneRow,S_A],format='csc')

p = p+1 # to reflect the added offset variable
d = d+1 # to reflect the added offset variable


Stranspose = S.T
S_A_t = S_A.T

print("n is "+str(n))
print("n_train is "+str(n_train))
print("n_test is "+str(n_test))
print("d is "+str(d))
print("p is "+str(p))
print("density of S: "+str(S.count_nonzero()/(n*float(p))))
print("density of S_A: "+str(S_A.count_nonzero()/(p*float(d))))
print("lambda is "+str(lam))
print("mu is "+str(mu))

#################################################################################
# create plug-in functions for gradient, prox, objective func evals, and matrix mults

def theGrad(x):
    return (1/float(n))*S_A_t.dot(Stranspose.dot(S.dot(S_A.dot(x))-y))

def G(x):
    return S_A.dot(x)

def Gt(x):
    return S_A_t.dot(x)

def theProx1(x,rho):
    '''
    prox corresponding to the ell1 composed with the matrix G.
    Doesn't apply to the offset/intercept x[0].
    '''
    xthresh = algo.proxL1_v2(x[1:len(x)], rho * lam * (1 - mu))
    return np.concatenate([np.array([x[0]]),xthresh])

def theProx2(x,rho):
    '''
    prox corresponding to the ell1 norm on gamma.
    The first element of x, x[0], corresponds to the offset and so is not
    thresholded (not subject to the ell_1 penalty. The last element x[-1]
    is the root of the node, x_{-r}, and so is also not thresholded.
    '''
    xthresh = algo.proxL1_v2(x[1:len(x)-1],rho*lam*mu)
    return np.concatenate([np.array([x[0]]),xthresh,np.array([x[-1]])])

def proxg(x,tau):
    '''
    used in Tseng-pd, frb-pd, and cp-bt
    '''
    out = algo.projLInf(x, lam * (1 - mu))
    out[0]=0.0
    return out

def proxfstar_4_tseng(x,tau):
    '''
    used in tseng-pd and frb-pd
    '''
    out = algo.projLInf(x,lam*mu)
    out[-1]=0.0
    out[0]=0.0
    return out

def theFunc(x):
    '''
    evaluate the objective function
    '''
    Ax = S_A.dot(x)
    return lam*(1-mu)*np.linalg.norm(Ax[1:len(Ax)],1)\
            + lam*mu*np.linalg.norm(x[1:len(x)-1],1)\
            + (1/(2*float(n)))*np.linalg.norm(y - S.dot(S_A.dot(x)),2)**2

def hfunc(x):
    '''
    Smart func evaluation also saves the matrix multiply for later use.
    Used in cp-bt
    '''
    Matx = S.dot(S_A.dot(x))
    return [ (1/(2*float(n)))*np.linalg.norm(y - Matx,2)**2,Matx]

def theGradSmart(Matx):
    '''
    Smart gradient exploits previously computed matrix multiply. Used in cp-bt.
    '''
    return (1/float(n))*S_A_t.dot(Stranspose.dot(Matx-y))

################## Run Algorithms ##############################################
print("================")
print("================")
print("================")
print("running 1fbt...")
init = algo.InitPoint([],np.zeros(d),np.zeros(d),np.zeros(p))
out1f = algo.PS1f_bt_comp(init,iter,G,theProx1,theProx2,
                          theGrad,Gt,theFunc,gamma = gamma1f)
print("1f TOTAL running time: "+str(out1f.times[-1]))
print("================")

print("running 2fbt...")
init = algo.InitPoint([],[],np.zeros(d),np.zeros(p))
out2f = algo.PS2f_bt_comp(init,iter,G,theProx1,theProx2,theGrad,Gt,
                          theFunc,gamma=gamma2f)
print("2f TOTAL running time: "+str(out2f.times[-1]))
print("================")

print("running cp-bt...")
init = algo.InitPoint(np.zeros(d),np.zeros(p),[],[])
outcp = algo.cpBT(theProx2, theGradSmart, proxg, theFunc, hfunc, init, iter=iter,
                  beta=betacp,stepInc=1.1,K=Gt,Kstar=G)
print("cp-bt TOTAL running time: "+str(outcp.times[-1]))
print("================")

print("running tseng...")
init = algo.InitPoint([],[],np.zeros(d),np.zeros(p))
outtseng = algo.tseng_product(theFunc, proxfstar_4_tseng, proxg, theGrad, init,
                              iter=iter, gamma1=gammatg,gamma2=gammatg,G=G,Gt=Gt)
print("TOTAL tseng time: "+str(outtseng.times[-1]))
print("================")

print("running frb...")
outfrb = algo.for_reflect_back(theFunc,proxfstar_4_tseng,proxg,theGrad,init,iter=iter,
                               gamma0=gammafrb,gamma1=gammafrb,G=G,Gt=Gt)
print("frb running time: "+str(outfrb.times[-1]))
print("================")

tol = 1e-3

def getMSE(x):
    MSEtrain = (1/(float(n_train))) * np.linalg.norm(y_train - S.dot(S_A.dot(x)), 2) ** 2
    MSEtest = (1/(float(n_test))) * np.linalg.norm(y_test - S_test.dot(S_A.dot(x)), 2) ** 2
    return [MSEtrain,MSEtest]


print("================")
print("================")
print("================")
print("getting Mean Square Errors (MSE)...")
[MSEtrain1f,MSEtest1f] = getMSE(out1f.x2)
print("training MSE 1f: "+str(MSEtrain1f))
print("testing MSE 1f: " + str(MSEtest1f))
[MSEtrain2f,MSEtest2f] = getMSE(out2f.x2)
print("training MSE 2f: "+str(MSEtrain2f))
print("testing MSE 2f: " + str(MSEtest2f))
[MSEtraincp, MSEtestcp] = getMSE(outcp.y)
print("training MSE cp: " + str(MSEtraincp))
print("testing MSE cp: " + str(MSEtestcp))
[MSEtraintg, MSEtesttg] = getMSE(outtseng.x )
print("training MSE tseng: " + str(MSEtraintg))
print("testing MSE tseng: " + str(MSEtesttg))
[MSEtrainfrb, MSEtestfrb] = getMSE(outfrb.x)
print("training MSE frb: " + str(MSEtrainfrb))
print("testing MSE frb: " + str(MSEtestfrb))



tAbsolutefinish = time.time()
print("total running time: "+str(tAbsolutefinish-tAbsolutestart))
print("================")
print("================")
print("================")
print("plotting...")
print("================")
print("================")
print("================")


opt = min(np.concatenate([np.array(out1f.fx2), np.array(out2f.fx2), np.array(outcp.f), np.array(outtseng.f),
                             np.array(outfrb.f)]))

markFreq = 2000
markerSz = 10
print("plotting relative error to optimality of funtion values")
print("optimal value estimated as lowest returned by any algorithm")
plt.semilogy(out1f.times,(np.array(out1f.fx2)-opt)/opt)
plt.semilogy(out2f.times,(np.array(out2f.fx2)-opt)/opt,'-o',markevery = markFreq,markersize =markerSz)
plt.semilogy(outfrb.times, (np.array(outfrb.f)-opt)/opt,'D-',markevery = markFreq,markersize =markerSz,color='brown')
plt.semilogy(outcp.times,(np.array(outcp.f)-opt)/opt,'rs-',markevery = markFreq,markersize =markerSz)
plt.semilogy(outtseng.times,(np.array(outtseng.f)-opt)/opt,'mx-',markevery = markFreq,markersize =markerSz)

fonts = 15
plt.xlabel('time (s)',fontsize = fonts)
plt.legend(['ps1fbt','ps2fbt','frb-pd','cp-bt','Tseng-pd'])
plt.title('relative error to optimality of function values')
plt.grid()
plt.show()

print("================")
print("================")
print("================")
print("plotting raw function values...")
plt.plot(out1f.times,np.array(out1f.fx2))
plt.plot(out2f.times,np.array(out2f.fx2))
plt.plot(outfrb.times,np.array(outfrb.f))
plt.plot(outcp.times,np.array(outcp.f))
plt.plot(outtseng.times,np.array(outtseng.f))
plt.xlabel('times (s)')
plt.title('raw function values')
plt.legend(['ps1fbt','ps2fbt','frb-pd','cp-bt','Tseng-pd'])
plt.show()
print("================")
print("================")
print("================")
print("plotting step sizes...")
print("================")
print("================")
print("================")

markFreq = 1000
plt.plot(out1f.rhos)
plt.plot(out2f.rhos,':')
plt.xlabel('iterations')
plt.title('backtracked stepsizes: rare features')
plt.legend(['1f','2f'])
plt.show()


print("plotting fz versus fx2 for ps1fbt and ps2fbt")
print("================")
print("================")
print("================")
zcomp = True
if zcomp:
    algo.compareZandX(out1f,"1f")
    algo.compareZandX(out2f,"2f")
