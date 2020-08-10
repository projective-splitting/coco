
import numpy as np
import scipy.sparse as sp

import sys
sys.path.append('../')
#sys.path.append('/home/pj222/gitFolders/proj_split_v2.0/group_logistic_regression/')

def logit(score):

    pos = np.log(1 + np.exp(score))
    pos2 = (~np.isinf(pos))*np.nan_to_num(pos)
    neg = score + np.log(1+np.exp(-score))
    neg2 = (~np.isinf(neg)) * np.nan_to_num(neg)
    coef = 0.5*np.ones(len(pos))
    coef = coef+0.5*np.isinf(pos)+0.5*np.isinf(neg)
    secondTerm = coef*(pos2+neg2)
    logAns = score - secondTerm
    return [np.exp(logAns),secondTerm]

def LRgrad(A,x,y):
    Ax = A.dot(x)
    score = -y*Ax
    [logItScore,_] = logit(score)
    return -A.T.dot(y*logItScore)

def LRfunc(A,x,y):
    Ax = A.dot(x)
    score = -y * Ax
    return sum(np.log(1+np.exp(score)))

def LRgradNew(A,Ax,y):
    score = -y*Ax
    [logItScore,_] = logit(score)
    return -A.T.dot(y*logItScore)

import algorithms as algo

def setUpRareFeatureProblem(lam,mu,path2Data,loss=2):
    S_train = sp.load_npz(path2Data+'S_train.npz') # training matrix
    S_A     = sp.load_npz(path2Data+'S_A.npz')     # this matrix is called H
                                                           # in the the paper
    y_train = np.load(path2Data+'y_train.npy')     # training labels

    (n_train,_) = S_train.shape

    y = y_train
    S = S_train
    n = n_train

    if loss == "log":
        y_train_class = 2*(y_train==5)-1


    #print("Adding all ones column to train/test matrix for offset/intercept...")
    onesCol = np.ones([n,1])
    onesCol = sp.csc_matrix(onesCol)
    S = sp.hstack([onesCol,S],format='csc')

    #print("The offset is replicated in gamma.")
    #print("To H, we append a column and row consisting of all zeros, except for")
    #print("a one in the upper left corner")
    (p,d) = S_A.shape
    zerosCol = np.zeros([p,1])
    zerosCol = sp.csc_matrix(zerosCol)
    S_A = sp.hstack([zerosCol,S_A],format='csc')
    zerosButOneRow = np.zeros([1,d+1])
    zerosButOneRow[0,0]=1.0

    zerosButOneRow = sp.csc_matrix(zerosButOneRow)
    S_A = sp.vstack([zerosButOneRow,S_A],format='csc')

    p = p+1 # to reflect the added offset variable
    d = d+1 # to reflect the added offset variable


    Stranspose = S.T
    S_A_t = S_A.T

    #print("n is "+str(n))
    #print("n_train is "+str(n_train))
    #print("d is "+str(d))
    #print("p is "+str(p))
    #print("density of S: "+str(S.count_nonzero()/(n*float(p))))
    #print("density of S_A: "+str(S_A.count_nonzero()/(p*float(d))))

    #################################################################################
    # create plug-in functions for gradient, prox, objective func evals, and matrix mults

    if loss == 2:
        def theGrad(x):
            return (1/float(n))*S_A_t.dot(Stranspose.dot(S.dot(S_A.dot(x))-y))

        def theFunc(x):
            '''
            evaluate the objective function
            '''
            Ax = S_A.dot(x)
            loss = (1.0/(2*float(n)))*np.linalg.norm(y - S.dot(Ax),2)**2

            return lam*(1-mu)*np.linalg.norm(Ax[1:len(Ax)],1)\
                    + lam*mu*np.linalg.norm(x[1:len(x)-1],1)\
                    + loss


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
    else:
        def theGrad(gamma):
            '''
            gradient wrt gamma
            '''
            beta = S_A.dot(gamma)
            return (1/float(n))*S_A.T.dot(LRgrad(S,beta,y_train_class))

        def theFunc(gamma):
            beta = S_A.dot(gamma)
            loss = (1/float(n))*LRfunc(S,beta,y_train_class)
            return loss +\
                        + lam*mu*np.linalg.norm(gamma[1:len(gamma)-1],1)\
                        + lam*(1-mu)*np.linalg.norm(beta[1:len(beta)],1)

        def hfunc(x):
            Matx = S.dot(S_A.dot(x))
            score = -y_train_class * Matx
            return [(1.0/n)*sum(np.log(1+np.exp(score))),Matx]

        def theGradSmart(Matx):
            return (1.0/n)*S_A.T.dot(LRgradNew(S,Matx,y_train_class))

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


    S_train_orig = sp.load_npz(path2Data+'S_train.npz') # training matrix
    S_A_orig     = sp.load_npz(path2Data+'S_A.npz')     # this matrix is called H
    y_train_orig = np.load(path2Data+'y_train.npy')     # training labels
    if loss == "log":
        y_train_orig = y_train_class

    return theGrad,G,Gt,theProx1,theProx2,proxg,proxfstar_4_tseng,theFunc,hfunc,\
             theGradSmart,d,p,S_train_orig,S_A_orig,y_train_orig
