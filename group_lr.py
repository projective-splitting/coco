import numpy as np
import time
from matplotlib import pyplot as plt


#=============== functions for assessing training performance #===============
def training_error_rate(A,xhat,y):
    predictions = np.sign(A.dot(xhat))
    scores = A.dot(xhat)


    return float(sum(predictions!=y))/len(y)

def nnz_groups(Partition,x):
    group_norms = np.array([np.linalg.norm(x[part]) for part in Partition])
    return group_norms


#=============== useful prox, projection, and the logit for logreg calculations

def proxL1(a,rholam):
    x = (a> rholam)*(a-rholam)
    x+= (a<-rholam)*(a+rholam)
    return x

def projLInf(x,thresh):
    return (x>=-thresh)*(x<=thresh)*x + (x>thresh)*thresh - (x<-thresh)*thresh

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

#========== group norm subroutines =============
def block_thresh(a,P,mus):
    # P is a non-overlapping partition of [1,...,d]
    # mus is the same length as P
    x = np.copy(a)
    for i in range(len(P)):
        x[P[i]]=norm_thresh(a[P[i]],mus[i])
    return x


def norm_thresh(a,mu):
    norma = np.linalg.norm(a,2)
    if(norma<=mu):
        return 0*a
    else:
        coef = 1-mu/norma
        return coef*a

#=========== gradients ==========================

def LRgrad(A,x,y):
    Ax = A.dot(x)
    score = -y*Ax
    [logItScore,_] = logit(score)
    return -A.T.dot(y*logItScore)

def LRgrad_smart(A,x,y):
    Ax = A.dot(x)
    score = -y*Ax
    [logItScore,_] = logit(score)
    return [-A.T.dot(y*logItScore),Ax]

def LRgrad_smart_alt(A,y,Ax):
    score = -y*Ax
    [logItScore,_] = logit(score)
    return -A.T.dot(y*logItScore)

#============ objective function evals ===================

#def LRfunc(A,x,y):
#    Ax = A.dot(x)
#    score = -y * Ax
#    return sum(np.log(1+np.exp(score)))

#f=LRfunc(A,x,y) => [f,_] = LRfunc_smart(A,x,y)

def LRfunc_smart(A,x,y):
    Ax = A.dot(x)
    score = -y * Ax
    [logItScore,secondTerm]=logit(score)
    return [sum(secondTerm),Ax]

def LRfunc_smart_alt(y, Ax):
    score = -y * Ax
    return sum(np.log(1+np.exp(score)))



#========== create all plugin functions used in test_group_lr.py =============

def create_all_funcs(A,y,Partitions,lam2s,lam_L1):

    # gradients

    def the_grad_smart_for_mal(Ax):
        return LRgrad_smart_alt(A, y, Ax)

    def theGrad(x):
        return LRgrad(A,x,y)

    def theGrad_smart(x):
        return LRgrad_smart(A,x,y)

    # objective function evaluators

    def the_func_smart_for_mal(x):
        return LRfunc_smart(A, x, y)

    def theFunc(x):
        [f,_] = LRfunc_smart(A,x,y)
        for j in range(len(Partitions)):
            f += lam2s[j]*np.linalg.norm(x[Partitions[j]],2)
        f+=lam_L1*np.linalg.norm(x[0:len(x)-1],1)

        return f

    def lrFunc(x):
        [f,_] = LRfunc_smart(A,x,y)
        return f

    def lrFunc_smart_alt(x,Ax):
        f = LRfunc_smart_alt(y,Ax)
        return f

   # proxes
    def prox_L1(x,rho):
        xthresh = proxL1(x[0:len(x)-1], rho * lam_L1)
        return np.concatenate([xthresh,np.array([x[-1]])])

    def proxg_for_mal(x,tau):
        return projLInf(x, lam_L1)

    def proxfstar4tseng(a, alpha):
        return a - alpha * block_thresh(a / alpha, Partitions, lam2s / alpha)

    def proxgstar4tseng(a, alpha):
        return a - alpha * prox_L1(a / alpha, alpha ** (-1))


    def group_prox(x,rho):
        return block_thresh(x,Partitions,rho*lam2s)

    return [theGrad,theGrad_smart,the_grad_smart_for_mal,theFunc,lrFunc_smart_alt,lrFunc,
            the_func_smart_for_mal,proxg_for_mal,proxfstar4tseng,
            proxgstar4tseng,prox_L1,group_prox]
