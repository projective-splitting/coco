import numpy as np
import rare_feature as rf
import algorithms as algo
import time
from matplotlib import pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as ln
import sys
'''
Feb 2020 plan
- Ideally would like to  use the same algorithm package, so have to deal with
the inconsistencies there.
er.
- As with the other run scripts, we need to replace the params.txt input with an
  argpass call.
'''
tAbsolutestart = time.time()

S_train = sp.load_npz('data/trip_advisor/S_train.npz')
S_test  = sp.load_npz('data/trip_advisor/S_test.npz')
S_A     = sp.load_npz('data/trip_advisor/S_A.npz')
y_train = np.load('data/trip_advisor/y_train.npy')
y_test  = np.load('data/trip_advisor/y_test.npy')

dir = sys.argv[-1]
params = rf.getParams(dir)
print "----------------------------------"

(n_train,_) = S_train.shape
(n_test,_) = S_test.shape

y = y_train
S = S_train
n = n_train

print "adding variable for offset..."
onesCol = np.ones([n,1])
onesCol = sp.csc_matrix(onesCol)
S = sp.hstack([onesCol,S],format='csc')
print "S shape now: "+str(S.shape)

onesCol = np.ones([n_test, 1])
onesCol = sp.csc_matrix(onesCol)
S_test = sp.hstack([onesCol, S_test], format='csc')
print "S_test shape now: "+str(S_test.shape)

(p,d) = S_A.shape
zerosCol = np.zeros([p,1])
zerosCol = sp.csc_matrix(zerosCol)
S_A = sp.hstack([zerosCol,S_A],format='csc')
zerosButOneRow = np.zeros([1,d+1])
zerosButOneRow[0]=1.0
zerosButOneRow = sp.csc_matrix(zerosButOneRow)
S_A = sp.vstack([zerosButOneRow,S_A],format='csc')
print "S_A shape now: "+str(S_A.shape)
p = p+1
d = d+1


Stranspose = S.T
S_A_t = S_A.T

print "n is "+str(n)
print "n_train is "+str(n_train)
print "n_test is "+str(n_test)
print "d is "+str(d)
print "p is "+str(p)
print "density of S: "+str(S.count_nonzero()/(n*float(p)))
print "density of S_A: "+str(S_A.count_nonzero()/(p*float(d)))

#normalize columns of S
doNormalize = int(params['doNormalize'])
if doNormalize:
    print "normalizing columns of S..."
    invCols = []
    for i in range(p):
        invCols.append(ln.norm(S[:,i])**(-1))
    invCols = sp.csc_matrix(np.diag(invCols))
    S = S.dot(invCols)

    if doConcat==False:
        # the factors which you used to normalize the train columns must agian
        # be used to normalize the features of the test data
        # so we again use invCols...
        #S_train = S
        #invCols = []
        #allZeroCols = 0
        #for i in range(p):
        #    normOfCol = ln.norm(S_test[:, i])
        #    if normOfCol>0:
        #        invCols.append(normOfCol ** (-1))
        #    else:
        #        allZeroCols += 1
        #        invCols.append(1.0)
        #invCols = sp.csc_matrix(np.diag(invCols))
        S_test = S_test.dot(invCols)


else:
    print "not normalizing cols..."


# create plug-in functions for gradient, prox, objective func evals, and matrix mults
lam = float(params['lam'])
alpha = float(params['alpha'])
print "lambda is "+str(lam)
print "alpha is "+str(alpha)
def theGrad(x):
    return (1/float(n))*S_A_t.dot(Stranspose.dot(S.dot(S_A.dot(x))-y))

def G(x):
    return S_A.dot(x)

def Gt(x):
    return S_A_t.dot(x)

def theProx1(x,rho):
    '''
    prox corresponding to the ell1 composed with A (H) regularizer
    '''
    xthresh = algo.proxL1_v2(x[1:len(x)], rho * lam * (1 - alpha))
    return np.concatenate([np.array([x[0]]),xthresh])

def theProx2(x,rho):
    '''
    prox corresponding to the ell1 norm on gamma.
    The first element of x, x[0], corresponds to the offset and so is not
    thresholded (not subject to the ell_1 penalty. The last element x[-1]
    is the root of the node, x_{-r}, and so is also not thresholded.
    '''
    xthresh = algo.proxL1_v2(x[1:len(x)-1],rho*lam*alpha)
    return np.concatenate([np.array([x[0]]),xthresh,np.array([x[-1]])])

def proxg(x,tau):
    out = algo.projLInf(x, lam * (1 - alpha))
    out[0]=0.0
    return out

def proxgOld(x):
    '''
    XX remove me XX
    '''
    out = algo.projLInf(x, lam * (1 - alpha))
    out[0]=0.0
    return out


def proxfstar_4_tseng(x,tau):
    out = algo.projLInf(x,lam*alpha)
    out[-1]=0.0
    out[0]=0.0
    return out

def theFunc(x):
    Ax = S_A.dot(x)
    return lam*(1-alpha)*np.linalg.norm(Ax[1:len(Ax)],1)\
            + lam*alpha*np.linalg.norm(x[1:len(x)-1],1)\
            + (1/(2*float(n)))*np.linalg.norm(y - S.dot(S_A.dot(x)),2)**2



iter = int(params['iter'])

print("running 1fbt...")
init = algo.InitPoint([],np.zeros(d),np.zeros(d),np.zeros(p))
gamma1f = 1e1
out1f = algo.PS1f_bt_comp(init,iter,G,theProx1,theProx2,
                          theGrad,Gt,theFunc,gamma = gamma1f)

print "1f TOTAL running time: "+str(out1f.times[-1])





print("running 2fbt...")
equalRhos2f = int(params['equalRhos2f'])
rho12f = float(params['rho12f'])
rho22f = float(params['rho22f'])
deltabt = float(params['deltabt'])
stepDec = float(params['stepDec'])
gamma2f = float(params['gamma2f'])
stepUp2f = float(params['stepUp2f'])
#[x2f,f2f,t2f,rhos2s2f,gradNorms2f,phis2f] = rf.ps2fbt(d,p,iter,G,rho12f,theProx1,rho22f,theProx2,theGrad,equalRhos2f,Gt,
#                      gamma2f,theFunc,deltabt,stepDec,stepUp2f,plotSteps)

init = algo.InitPoint([],[],np.zeros(d),np.zeros(p))
out2f = algo.PS2f_bt_comp(init,iter,G,theProx1,theProx2,theGrad,Gt,
                          theFunc,gamma=gamma2f)

print "2f TOTAL running time: "+str(out2f.times[-1])


saveResults = True


print "running cp-bt..."
proxfstar = theProx2
gradH = theGrad
tau = float(params['tau'])
Kstarop = G
Kop = Gt
mu = float(params['mu'])
delta_cp = float(params['delta_cp'])
beta = float(params['beta'])
stepIncCP = float(params['stepIncCP'])
plotStepsCP = int(params['plotStepsCP'])

def hfunc(x):
    Matx = S.dot(S_A.dot(x))
    return [ (1/(2*float(n)))*np.linalg.norm(y - Matx,2)**2,Matx]

def theGradSmart(Matx):
    return (1/float(n))*S_A_t.dot(Stranspose.dot(Matx-y))


#[xcp,fcp,tcp] = rf.malitskyBT(d,p,iter,proxfstar, theGradSmart, tau, proxg,theFunc,
#                              Kop,Kstarop,delta_cp,beta,mu,hfunc,stepIncCP,
#                              plotStepsCP)

init = algo.InitPoint(np.zeros(d),np.zeros(p),[],[])
outcp = algo.cpBT(theProx2, theGradSmart, proxg, theFunc, hfunc, init, iter=iter,
                  beta=beta,stepInc=1.1,K=Gt,Kstar=G)





print "cp-bt TOTAL running time: "+str(outcp.times[-1])





print "running tseng..."
alpha_tg = float(params['alpha_tg'])
theta_tg = float(params['theta_tg'])
stepIncrease_tg = float(params['stepIncrease_tg'])
proxgstar_tg = proxg
stepDecrease_tg = float(params['stepDecrease_tg'])
gamma_tg_1 = float(params['gamma_tg_1'])
gamma_tg_2 = float(params['gamma_tg_2'])
gammatg = gamma_tg_1
#[xtseng,ftseng,ttseng] = rf.tseng_product(d, p, iter, alpha_tg, theta_tg, theFunc,
                                          #stepIncrease_tg, proxfstar_4_tseng, proxgstar_tg,
                                          #theGrad,stepDecrease_tg,gamma_tg_1,gamma_tg_2,G,
                                          #Gt)

init = algo.InitPoint([],[],np.zeros(d),np.zeros(p))
outtseng = algo.tseng_product(theFunc, proxfstar_4_tseng, proxg, theGrad, init,
                              iter=iter, gamma1=gammatg,gamma2=gammatg,G=G,Gt=Gt)


print "TOTAL tseng time: "+str(outtseng.times[-1])



print "running frb..."
proxgstar_tg = proxgOld
x0 = np.zeros(d)
gammafrb = float(params['gamma_frb'])
lam_frb = float(params['lam_frb'])
stepIncrease_frb = float(params['stepIncrease_frb'])
delta_frb = float(params['delta_frb'])
stepDecrease_frb = float(params['stepDecrease_frb'])

#[xfrb,ffrb,tfrb] = rf.for_reflect_back(p,proxfstar_4_tseng,proxgstar_tg,theGrad,iter,x0,gamma_frb,gamma_frb,lam_frb,stepIncrease_frb,
#                                  delta_frb,stepDecrease_frb,theFunc,True,G,Gt)

outfrb = algo.for_reflect_back(theFunc,proxfstar_4_tseng,proxg,theGrad,init,iter=iter,
                               gamma0=gammafrb,gamma1=gammafrb,G=G,Gt=Gt)


print "frb running time: "+str(outfrb.times[-1])




tol = 1e-3


print "++++++++++++++++"


def getMSE(x):
    MSEtrain = (1/(float(n_train))) * np.linalg.norm(y_train - S.dot(S_A.dot(x)), 2) ** 2
    MSEtest = (1/(float(n_test))) * np.linalg.norm(y_test - S_test.dot(S_A.dot(x)), 2) ** 2
    return [MSEtrain,MSEtest]

def getMSEalt(v):
    MSEtrain = (1/(float(n_train))) * np.linalg.norm(y_train - S.dot(v), 2) ** 2
    MSEtest = (1/(float(n_test))) * np.linalg.norm(y_test - S_test.dot(v), 2) ** 2
    return [MSEtrain, MSEtest]

doConcat = False
if doConcat==False:
    print "getting testing MSE..."
    [MSEtrain1f,MSEtest1f] = getMSE(out1f.x2)
    [MSEtrain1falt, MSEtest1falt] = getMSEalt(out1f.x1)
    print "training errors 1f: "+str(MSEtrain1f)+','+str(MSEtrain1falt)
    print "testing errors 1f: " + str(MSEtest1f) + ',' + str(MSEtest1falt)
    [MSEtrain2f,MSEtest2f] = getMSE(out2f.x2)
    print "training error 2f: "+str(MSEtrain2f)
    print "testing error 2f: " + str(MSEtest2f)
    [MSEtraincp, MSEtestcp] = getMSE(outcp.y)
    print "training error cp: " + str(MSEtraincp)
    print "testing error cp: " + str(MSEtestcp)
    [MSEtraintg, MSEtesttg] = getMSE(outtseng.x )
    print "training error tseng: " + str(MSEtraintg)
    print "testing error tseng: " + str(MSEtesttg)
    [MSEtrainfrb, MSEtestfrb] = getMSE(outfrb.x)
    print "training error frb: " + str(MSEtrainfrb)
    print "testing error frb: " + str(MSEtestfrb)



tAbsolutefinish = time.time()
print "running time: "+str(tAbsolutefinish-tAbsolutestart)
print "----------------------------------"


opt = min(np.concatenate([np.array(out1f.fx2), np.array(out2f.fx2), np.array(outcp.f), np.array(outtseng.f),
                             np.array(outfrb.f)]))



markFreq = 2000
markerSz = 10

#g = plt.figure(2)
plt.semilogy(out1f.times,(np.array(out1f.fx2)-opt)/opt)
plt.semilogy(out2f.times,(np.array(out2f.fx2)-opt)/opt,'-o',markevery = markFreq,markersize =markerSz)



plt.semilogy(outfrb.times, (np.array(outfrb.f)-opt)/opt,'D-',markevery = markFreq,markersize =markerSz,color='brown')

plt.semilogy(outcp.times,(np.array(outcp.f)-opt)/opt,'rs-',markevery = markFreq,markersize =markerSz)
plt.semilogy(outtseng.times,(np.array(outtseng.f)-opt)/opt,'mx-',markevery = markFreq,markersize =markerSz)


fonts = 15
plt.xlabel('time (s)',fontsize = fonts)
plt.legend(['ps1fbt','ps2fbt','frb-pd','cp-bt','Tseng-pd'])


plt.grid()
xlim1 = int(params['xlim1'])
plt.xlim([0,xlim1])
plt.show()

if plotSteps:
    print "plotting steps..."
    markFreq = 1000
    plt.plot(phis1f)
    plt.title('phi 1f')
    plt.show()
    plt.plot(phis2f)
    plt.title('phi 2f')
    plt.show()
    plt.plot(out1f.rhos)
    plt.plot(out2f.rhos,':')
    plt.xlabel('iterations')
    plt.title('discovered stepsizes: rare features')
    plt.legend(['1f','2f'])
    plt.show()


zcomp = True
if zcomp:
    algo.compareZandX(out1f,"1f")
    algo.compareZandX(out2f,"2f")
