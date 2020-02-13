'''
This file runs the group logistic regression experiment in the paper
"Single Forward Step Projective Splitting: Exploiting Cocoercivity", Patrick R. Johnstone
and Jonathan Eckstein, arXiv:1902.09025.
Various parameters can be set from the command line.
To run:(from the directory containing this file)
$python run_group_lr.py
This runs with default parameters.
To see what parameters can be set from the command line, run
$python run_portfolio.py -h
This code has been tested with python2.7 and python3.5 and requires numpy.
'''
import numpy as np
from matplotlib import pyplot as plt
import time as time
import group_lr as gp
import algorithms as algo
import argparse

tActualStart = time.time()

parser = argparse.ArgumentParser(description='Group Sparse LR Experiment')

parser.add_argument('--lam1',type=float,default=1e-1,dest='lam1',
                    help = 'reg parameter for ell 1 norm',metavar='lam1')
parser.add_argument('--lam2',type=float,default=1e-1,dest='lam2',
                    help = 'reg parameter for group norm',metavar='lam2')
parser.add_argument('--iter',type=int,default=2000,dest='iter',
                    help = 'number of iterations (all algorithms the same)',metavar='iter')
parser.add_argument('--stepIncrease',type=float,default=1.1,dest='stepIncrease',
                    help = 'stepsize increase factor (all algorithm the same)',metavar='step_increase')
parser.add_argument('--dataset',default='colitis',dest='dataset',
                    help = 'which dataset, either colitis or breastCancer',metavar='dataset')
parser.add_argument('--gamma1f',type=float,default=1.0,dest='gamma1f',
                    help = 'tuning parameter for ps1fbt',metavar='gamma1f')
parser.add_argument('--gamma2f',type=float,default=1.0,dest='gamma2f',
                    help = 'tuning parameter for ps2fbt',metavar='gamma2f')
parser.add_argument('--gammatg',type=float,default=1e3,dest='gammatg',
                    help = 'tuning parameter for Tseng-pd',metavar='gammatg')
parser.add_argument('--gammafrb',type=float,default=1e3,dest='gammafrb',
                    help = 'tuning parameter for FRB',metavar='gammafrb')
parser.add_argument('--betacp',type=float,default=1e1,dest='betacp',
                    help = 'tuning parameter for cp-bt',metavar='betacp')

lam1 = parser.parse_args().lam1
lam2 = parser.parse_args().lam2
iter = parser.parse_args().iter
step_increase = parser.parse_args().stepIncrease
dataset = parser.parse_args().dataset
gamma1f = parser.parse_args().gamma1f
gamma2f = parser.parse_args().gamma2f
gammatg = parser.parse_args().gammatg
gammafrb = parser.parse_args().gammafrb
betacp = parser.parse_args().betacp



if dataset =='colitis':
    print("Colitis dataset")
    dataFolder = 'data/colitis/'
else:
    print("Breast Cancer dataset")
    dataFolder = 'data/breast_cancer/'


print("Loading data...")
A = np.load(dataFolder+'A_stripped.npy',allow_pickle=True)
y = np.load(dataFolder+'y.npy',allow_pickle=True)
Partitions = np.load(dataFolder+'Partition.npy',allow_pickle=True)
print("Data loaded successfully.")


[n,d] = A.shape

print("number of genes: "+str(d))
print("number of groups: "+str(len(Partitions)))
print("av genes per group: "+str(sum([len(parts) for parts in Partitions])/float(len(Partitions))))



print("normalizing columns of A to unit norm")
col_norms = np.linalg.norm(A, axis=0)
invdiago = np.diag(1/col_norms)
A = A.dot(invdiago)

# adding a column of all ones to deal with the offset
AT = np.concatenate([A.T, np.ones([1, n])])
A = AT.T

lam2s = lam2*np.ones(len(Partitions))


# call subroutine create the plug-in functions for gradients, proxes,
# and function evals
[theGrad,theGrad_smart,the_grad_smart_for_cp,theFunc,lrFunc_smart,lrFunc,
        the_func_smart_for_cp,proxg_for_cp,proxfstar4tseng,
        proxgstar4tseng,prox_L1,group_prox]\
         = gp.create_all_funcs(A,y,Partitions,lam2s,lam1)


def print_results(alg,x,time2run):
    print("===results summary for " +alg+"===")
    print("running time "+str(time2run))
    print(alg+" training error: "+str(gp.training_error_rate(A, x, y)))
    print(alg+" nnz: "+str(sum(abs(x)>1e-5)))
    group_norms = gp.nnz_groups(Partitions,x)
    print(alg+" nnz groups: "+str(sum(group_norms>1e-5)))
    print("=== end results summary===")


print("running ada3op")
init = algo.InitPoint([],[],np.zeros(d+1),[])
out3op = algo.adap3op(d+1,group_prox,theGrad_smart,lrFunc,theFunc,prox_L1,lrFunc_smart,
            init, stepIncrease = step_increase,iter=iter,lip_const = lam1)
print_results("ada3op",out3op.x,out3op.times[-1])



print("running 1fbt...")
init =algo.InitPoint(np.zeros(d+1),np.zeros(d+1),np.zeros(d+1),np.zeros(d+1))
out1f = algo.PS1f_bt(theFunc,prox_L1,group_prox,theGrad,d+1,init,gamma = gamma1f,
                     stepIncrease = step_increase,iter=iter)
print_results("ps1fbt",out1f.x1,out1f.times[-1])


print("running 2fbt...")
init =algo.InitPoint([],[],np.zeros(d+1),np.zeros(d+1))
out2f = algo.PS2f_bt(d+1,theFunc,theGrad,prox_L1,group_prox,init,iter=iter,
                     gamma=gamma2f,stepIncrease = step_increase)

print_results("ps2fbt",out2f.x1,out2f.times[-1])

print("running cp-bt")
init =algo.InitPoint(np.zeros(d+1),np.zeros(d+1),[],[])
outcp = algo.cpBT(d+1, group_prox, the_grad_smart_for_cp, proxg_for_cp,
                  theFunc, the_func_smart_for_cp, init, iter=iter,beta=betacp,
                  stepInc=step_increase)
print_results("cp-bt",outcp.y,outcp.times[-1])

print("Running Tseng-pd")
init =algo.InitPoint([],[],np.zeros(d+1),np.zeros(d+1))
outTseng = algo.tseng_product(d+1, theFunc, proxfstar4tseng, proxgstar4tseng,
                              theGrad, init, stepIncrease=step_increase,
                               gamma1=gammatg,gamma2=gammatg,iter=iter)
print_results("tseng-pd",outTseng.x,outTseng.times[-1])

print("running FRB...")
init =algo.InitPoint([],[],np.zeros(d+1),np.zeros(d+1))
outFRB = algo.for_reflect_back(theFunc,proxfstar4tseng,proxgstar4tseng,theGrad,init,iter=iter,gamma0=gammafrb,
                     gamma1=gammafrb,stepIncrease=step_increase)

print_results("frb-pd",outFRB.x,outFRB.times[-1])


tendFinal = time.time()
print("===============")
print("total running time: "+str(tendFinal - tActualStart))

print("plotting...")

plot_steps = True
if plot_steps:    
    markFreq=100
    markerSz = 10
    plt.semilogy(out1f.rhos)
    plt.semilogy(out2f.rhos,':')


    plt.legend(['ps1fbt','ps2fbt'])
    plt.title('discovered stepsizes: group log reg')
    plt.xlabel('iterations')
    plt.grid()

    plt.show()




opt = min(np.concatenate([np.array(out3op.f),np.array(out1f.fx1),np.array(out2f.fx1),np.array(outTseng.f),
                             np.array(outcp.f),np.array(outFRB.f)]))


plot_raw = True
if(plot_raw):

    plt.plot(out1f.times,out1f.fx1)
    plt.plot(out2f.times,out2f.fx1)
    plt.plot(out3op.times,out3op.f)
    plt.plot(outcp.times,outcp.f)
    plt.plot(outTseng.times,outTseng.f)
    plt.plot(outFRB.times,outFRB.f)

    plt.plot(out3op.times,opt*np.ones(len(out3op.times)))
    plt.legend(['1fbt','2fbt','3opBT','cpbt','tseng','frb'])
    plt.xlabel('time (s) excluding time to evaluate objective')
    plt.title("Objective Function values versus elapsed time")
    plt.show()

logPlot = True

if(logPlot):
    markFreq = 500
    markerSz = 10
    plt.semilogy(out1f.times,(np.array(out1f.fx1) - opt)/opt)
    plt.semilogy(out2f.times,(np.array(out2f.fx1) - opt)/opt,'-o',markevery = markFreq,markersize =markerSz)

    plt.semilogy(out3op.times, (np.array(out3op.f) - opt)/opt,'-v',markevery = markFreq,markersize =markerSz)

    plt.semilogy(outcp.times, (np.array(outcp.f) - opt) / opt,'s-',markevery = markFreq,markersize =markerSz)
    plt.semilogy(outTseng.times, (np.array(outTseng.f) - opt) / opt,'x-',markevery = markFreq,markersize =markerSz)
    plt.semilogy(outFRB.times,(np.array(outFRB.f) - opt) / opt,'D-',markevery = markFreq,markersize =markerSz)


    plt.legend(['ps1fbt','ps2fbt','ada3op','cp-bt','tseng-pd','frb'],fontsize='large')
    #plt.ylabel('relative error objective function gap')
    plt.xlabel('time (s) excluding time to evaluate objective',fontsize='large')
    plt.grid(True)
    plt.show()


#compare z versus fx1 for ps1f and ps2f
zcomp = True
if zcomp:
    algo.compareZandX(out1f,"1f")
    algo.compareZandX(out2f,"2f")



# get result returns the first iterate s.t. f stays below tol
def getResult(f,tol):
    f_flip = np.flip(f,0)
    result = (1.0 * (f_flip < tol)).argmin()
    result = len(f) - result
    return result

tol = 1e-3

iter_3opt = getResult((np.array(out3op.f) - opt)/opt,tol)
iter_tseng = getResult((np.array(outTseng.f) - opt) / opt,tol)
iter_cp = getResult((np.array(outcp.f) - opt) / opt,tol)
iter_1f = getResult((np.array(out1f.fx1) - opt)/opt,tol)
iter_2f = getResult((np.array(out2f.fx1) - opt)/opt,tol)
iter_frb = getResult((np.array(outFRB.f) - opt) / opt,tol)


print("=================================")
print("=================================")
print("=================================")
print("<<<<<<<<<<< Results >>>>>>>>>>>>>")
if iter_3opt<len(out3op.times):
    totalT3op = out3op.times[iter_3opt]
    print("3op time: "+str(totalT3op))
    print("3op iteration: " + str(iter_3opt))
else:
    print("3opt no made it")
print("=================================")
if iter_1f<len(out1f.times):
    totalT1f = out1f.times[iter_1f]
    print("1f time: "+str(totalT1f))
    print("1f iteration: " + str(iter_1f))
else:
    print("1f no made it")
print("=================================")
if iter_2f<len(out2f.times):
    totalT2f = out2f.times[iter_2f]
    print("2f time: "+str(totalT2f))
    print("2f iteration: "+str(iter_2f))
else:
    print("2f no made it")
print("=================================")
if iter_tseng<len(outTseng.times):
    totalTtseng = outTseng.times[iter_tseng]
    print("tseng time: "+str(totalTtseng))
    print("tseng iteration: "+str(iter_tseng))
else:
    print("tseng no made it")
print("=================================")
if iter_cp<len(outcp.times):
    totalTcp = outcp.times[iter_cp]
    print("cp time: "+str(totalTcp))
    print("cp iteration: "+str(iter_cp))
else:
    print("cp no made it")
print("=================================")
if iter_frb<len(outFRB.times):
    totalTfrb = outFRB.times[iter_frb]
    print("frb time: " + str(totalTfrb))
    print("frb iteration: " + str(iter_frb))
else:
    print("frb did not make it")

print("=================================")
print("=================================")
print("=================================")

print("<<<<<<<<<<<         >>>>>>>>>>>>>")
