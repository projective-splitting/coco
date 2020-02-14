'''
This file runs the portfolio experiment in the paper
"Single Forward Step Projective Splitting: Exploiting Cocoercivity", Patrick R. Johnstone
and Jonathan Eckstein, arXiv:1902.09025.
Various parameters can be set from the command line.
To run:(from the directory containing this file)
$python run_portfolio.py
This runs with default parameters.
To see what parameters can be set from the command line, run
$python run_portfolio.py -h
This code has been tested with python2.7 and python3.5.
'''
import algorithms as algo
import numpy as np
from matplotlib import pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser(description='Portfolio Experiment')
parser.add_argument('--dimension',type=int,default=1000,dest='dimension',
                    help = 'problem dimension/number of variables',metavar='d')
parser.add_argument('--deltar',type=float,default=0.5,dest='deltar',
                    help = 'constant used to generate r',metavar='deltar')
parser.add_argument('--iter',type=int,default=1000,dest='iterations',
                    help = 'number of iterations to run all algorithms',metavar='iter')
parser.add_argument('--gamma1f',type=float,default=0.01,dest='gamma1f',
                    help = 'primal-dual constant for ps1f',metavar='gamma1f')
parser.add_argument('--gammafrb',type=float,default=1.0,dest='gammafrb',
                    help = 'primal-dual constant for frb',metavar='gammafrb')
parser.add_argument('--gamma2f',type=float,default=0.1,dest='gamma2f',
                    help = 'primal-dual constant for ps2f',metavar='gamma2f')
parser.add_argument('--betacp',type=float,default=1.0,dest='betacp',
                    help = 'primal-dual constant for cp-bt',metavar='betacp')
parser.add_argument('--gammatg',type=float,default=1.0,dest='gammatg',
                    help = 'primal-dual constant for Tseng-pd',metavar='gammatg')
parser.add_argument('--verbose',type=int,default=0,dest='verbose',
                    help = 'set to 1 to print iteration number, else 0',metavar='verbose')
parser.add_argument('--runCVX',type=int,default=0,dest='runCVX',
                    help = 'set to 1 to run cvxpy',metavar='runCVX')

# d is the problem dimension, size of the vector x in the Markovitz optimization problem
# default to d = 1000
d = parser.parse_args().dimension
deltar = parser.parse_args().deltar
iter = parser.parse_args().iterations
verbose = parser.parse_args().verbose

# These parameters are important tuning parameters which effect practical performance
# see table 1 on page 30 of the arXiv paper for how we set it for the problems
# in that paper
gamma1f = parser.parse_args().gamma1f
gamma_frb = parser.parse_args().gammafrb
gamma2f = parser.parse_args().gamma2f
betacp = parser.parse_args().betacp
gamma_tg = parser.parse_args().gammatg
runCVX = parser.parse_args().runCVX

tBegin = time.time()

# randomly generate vector m of mean investment returns (between 0 and 100)
# appearing in the constraints
m = np.random.random_sample(d) * 100.0


# r appears in the constraint as the minimum expected return
#deltar = 1 implies r = average(m)
r = deltar*sum(m)/(d)

trand = time.time()

print(str(d)+" dimension problem")
print("creating matrix...")
# Q is a psd matrix
Q = np.random.normal(0,1,[d,d])
Q = Q.dot(Q.T)
Q = Q/d


print("time to generate random data: "+str(time.time()-trand))

#----------------------------------------------------
#----------------------------------------------------
#----------------------------------------------------
# plug-in functions for computing proxes, gradients, etc
#----------------------------------------------------
def proxfstar4Tseng(a,alpha):
    return a - alpha*algo.projSimplex(a/alpha)

def proxgstar4Tseng(a,alpha):
    return a - alpha*algo.projHplane(a/alpha,m,r)

def theProx1(t,rho):
    #simplex projection
    return algo.projSimplex(t)

def theProx2(t, rho):
    #hyperplane projection
    return algo.projHplane(t,m,r)

def hyper_resid(x):
    #hyperplane residual
    return -min([0,m.dot(x)-r])

def theGrad(x):
    return Q.dot(x)

def theGradSmart(x):
    # many of our algorithms are implemented to take advantage of
    # matrix multiplications in computed previous gradients, for example in
    # future func evaluations (eg: cp-bt).
    # for portfolio this makes no difference as the gradient is the matrix
    # multiply, but for consistency with other problems, we define the smart
    # gradient in this way.
    Qx = Q.dot(x)
    return [Qx,Qx]

def theFunc(x):
    return 0.5*x.T.dot(Q.dot(x))

def theFuncSmart(x,Qx):
    # smart func makes use of the previously computed matrix multiply
    return 0.5*x.T.dot(Qx)

def theFuncHelpful(x):
    # smart func evaluation also returns the matrix multiply for future use.
    Qx = Q.dot(x)
    return [0.5*x.T.dot(Q.dot(x)),Qx]

def theGradEasy(Qx):
    # This grad evaluation uses the matrix multiply which was already computed.
    # Used in cp-bt.
    return Qx

def prox2cp(a):
    return algo.projHplaneCP(a,m,r)

def proxg4cp(a,tau):
    return a - tau*prox2cp(a/tau)

def print_results(alg, t, x, mults):
    # function to print after an algorithm runs
    print("==================================")
    print(alg + " running time (excluding function computes for plots): " +str(t))
    print(alg + " simplex constraint (should be 1): " + str(sum(x)))
    print(alg + " nonnegativity constraint (should be positive): " + str(min(x)))
    print(alg + " hyperplane constraint (should be positive): " + str(x.dot(m) - r))
    print(alg + " total matrix multiplications: " + str(mults[-1]))

#----------------------------------------------------
#----------------------------------------------------
#----------------------------------------------------
# ...Running algorithms..............................
#----------------------------------------------------

print("number of iterations (same for each method): " + str(iter))


# set runCVX to true to run cvx. Only good for d<=1000 on my machine otherwise too slow
# also, you must have CVXPY properly installed

if(runCVX):
    print("running cvx...")
    tstart_cvx = time.time()

    [cvxopt,xopt] = algo.runCVX_portfolio(d,Q,m,r)
    tend_cvx = time.time()
    print("CVX runtime: "+str(tend_cvx-tstart_cvx))
else:
    print("not running cvx")
print("=========================")
print("Running projective splitting one-forward-step backtrack... (ps1fbt)")

# initPoint returns an initial point to start the algorithm.
# initPoint is called before every algorithm to ensure we get new memory for
# each new variable

initPoint = algo.InitPoint(np.ones(d)/d,np.ones(d)/d,np.ones(d)/d,np.zeros(d))
out1f = algo.PS1f_bt(theFunc,theProx1,theProx2,theGrad,initPoint,gamma=gamma1f,
                    hyper_resid=hyper_resid,verbose=verbose,iter=iter)

print_results("1fbt", out1f.times[-1], out1f.x1, out1f.grad_evals)

print("=========================")
print("running forward reflected backward (primal-dual)... (frb-pd)")
x0 = np.ones(d)/d
init = algo.InitPoint([],[],np.ones(d)/d,np.zeros(d))
outfrb = algo.for_reflect_back(theFunc,proxfstar4Tseng,proxgstar4Tseng,theGrad,init,
                                gamma0=gamma_frb,gamma1=gamma_frb,iter=iter,
                                hyper_resid=hyper_resid,verbose=verbose)

print_results("frb", outfrb.times[-1],outfrb.x, outfrb.grad_evals)

print("=========================")
print("Running adaptive three operator splitting (ada3op)")
init = algo.InitPoint([],[],np.ones(d)/d,[])
out3op = algo.adap3op(theProx1,theGradSmart,theFunc,theFunc,theProx2,theFuncSmart,
                    init,hyper_resid=hyper_resid,verbose=verbose,iter=iter)
print_results("ada3op", out3op.times[-1], out3op.x,out3op.func_evals+out3op.grad_evals)

print("=========================")
print("Running projective splitting two-forward-step with backtrack... (ps2fbt)")
init = algo.InitPoint([],[],np.ones(d)/d,np.zeros(d))
out2f = algo.PS2f_bt(theFunc,theGrad,theProx1,theProx2,init,gamma=gamma2f,
                    hyper_resid=hyper_resid,verbose=verbose,iter=iter)
print_results("2fbt", out2f.times[-1], out2f.x1, out2f.grad_evals)

print("=========================")
print("Running Chambolle-Pock Primal-Dual splitting (cp-bt)...")

init = algo.InitPoint(np.ones(d)/d,np.ones(d)/d,[],[])
outcp = algo.cpBT(theProx1, theGradEasy, proxg4cp, theFunc, theFuncHelpful, init = init,
                  hyper_resid=hyper_resid,beta=betacp,iter=iter,
                  verbose=verbose)

print_results("cp-bt", outcp.times[-1], outcp.y, outcp.func_evals)

print("=========================")
print("running Tseng-pd...")

init = algo.InitPoint([],[],np.ones(d)/d,np.ones(d)/d)

outTseng = algo.tseng_product(theFunc, proxfstar4Tseng, proxgstar4Tseng,
                              theGrad,init,hyper_resid=hyper_resid,iter=iter,
                              gamma1=gamma_tg,gamma2=gamma_tg,verbose=verbose)

print_results("tseng-pd", outTseng.times[-1], outTseng.x, outTseng.grad_evals)

print("total Runtime for all algorithms: "+str(time.time()-tBegin))
print("=========================")
print("plotting...")



plt.plot(out1f.rhos)
plt.plot(out2f.rhos,':')
plt.xlabel('iterations')
plt.title('discovered stepsizes via backtracking for portfolio')
plt.legend(['1f','2f'])
plt.show()


#Select f^* (the optimal function value) to be equal to the lowest function value returned
# by any algorithm which has a feasible iterate
opt2compare = []
minConstrViol = 1e-9 # allow constraint violations less than minConstrViol

def upd_opts(constr,f):
    if constr<minConstrViol:
        opt2compare.append(f)

if runCVX:
    opt2compare.append(cvxopt)

upd_opts(out1f.constraints[-1],out1f.fx1[-1])
upd_opts(out2f.constraints[-1],out2f.fx1[-1])
upd_opts(out3op.constraints[-1],out3op.f[-1])
upd_opts(outfrb.constraints[-1],outfrb.f[-1])
upd_opts(outcp.constraints[-1],outcp.f[-1])
upd_opts(outTseng.constraints[-1],outTseng.f[-1])

if(len(opt2compare)==0):
    print("All algorithms still infeasible!")
    print("Setting opt to 1")
    opt = 1.0
else:
    opt = min(np.array(opt2compare))



plt.semilogy(out1f.constraints)
plt.semilogy(out2f.constraints)
plt.semilogy(out3op.constraints)
plt.semilogy(outfrb.constraints)
plt.semilogy(outcp.constraints)
plt.semilogy(outTseng.constraints)
plt.ylabel('constraint violations')
plt.xlabel('iteration')
plt.legend(['1fbt','2fbt','ada3op','frb-pd','cp-bt','tseng-pd'])
plt.show()



opt1 = opt
if(abs(opt)<1e-7):
    print("opt val is 0, so not doing relative error plot")
    opt2 = 1.0
else:
    opt2 = opt

plt.semilogy(abs(np.array(out1f.fx1)-opt1)/opt2 )
plt.semilogy(abs(np.array(out2f.fx1)-opt1)/opt2)
plt.semilogy(abs(np.array(out3op.f) - opt1) / opt2)
plt.semilogy(abs(np.array(outcp.f)-opt1)/opt2)
plt.semilogy(abs(np.array(outTseng.f) - opt1) / opt2)
plt.semilogy(abs(np.array(outfrb.f)-opt1)/opt2)

plt.xlabel('Number of Iterations')
plt.ylabel('Relative Objective Optimality Gap')
plt.grid()
plt.legend(['ps1fbt', 'ps2fbt', 'ada3op','cp-bt','tseng-pd','frb-pd'])
plt.show()


# normal (non-log) plot of function vals vs iteration number


plt.plot(out1f.fx1)
plt.plot(out2f.fx1)
plt.plot(out3op.f)
plt.plot(outfrb.f)
plt.plot(outcp.f)
plt.plot(outTseng.f)
plt.legend(['ps1fbt', 'ps2fbt', 'ada3op','frb','cpbt','tseng'])
plt.xlabel("iteration")
plt.ylabel("function values")
plt.title('function values')

plt.show()

# semilogy plot vs number of matrix multiplications


opt1 = opt
if(abs(opt)<1e-7):
    print("opt val is 0, so not doing relative error plot")
    opt2 = 1.0
else:
    opt2 = opt

mults3op = np.array(out3op.grad_evals[0:iter]) + np.array(out3op.func_evals[0:iter])
plt.semilogy(out1f.grad_evals[0:iter],[abs(out1f.fx1[i] - opt1) / opt2 for i in range(iter)])
plt.semilogy(out2f.grad_evals[0:iter],[abs(out2f.fx1[i] - opt1) / opt2 for i in range(iter)],':')
plt.semilogy(mults3op, [abs(out3op.f[i] - opt1) / opt2 for i in range(iter)],'--')
plt.semilogy(outcp.func_evals,abs(np.array(outcp.f) - opt1) / opt2)
plt.semilogy(outTseng.grad_evals, abs(np.array(outTseng.f) - opt1) / opt2)
plt.semilogy(outfrb.grad_evals, abs(np.array(outfrb.f) - opt1) / opt2)

plt.legend(['ps1fbt', 'ps2fbt', 'ada3op','cp-bt','tseng-pd','frb-pd'])
plt.xlabel('Matrix Multiplies')
plt.ylabel('Relative Objective Optimality Gap')
plt.grid()
plt.show()

opt1 = opt
if(abs(opt)<1e-7):
    print("opt val is 0, so not doing relative error plot")
    opt2 = 1.0
else:
    opt2 = opt
plt.semilogy(out1f.times,abs(np.array(out1f.fx1)-opt1)/opt2)
plt.semilogy(out2f.times, abs(np.array(out2f.fx1) - opt1) / opt2,':')
plt.semilogy(out3op.times, abs(np.array(out3op.f) - opt1) / opt2,'--')
plt.semilogy(outcp.times,abs(np.array(outcp.f) - opt1) / opt2)
plt.semilogy(outTseng.times, abs(np.array(outTseng.f) - opt1) / opt2)
plt.semilogy(outfrb.times, abs(np.array(outfrb.f) - opt1) / opt2)

plt.grid()
plt.xlabel('time (s)')
plt.ylabel('Relative Objective Optimality Gap')
plt.legend(['ps1fbt','ps2fbt','ada3op','cp-bt','tseng-pd','frb-pd'])
plt.show()

# ce_xx combines the constraint violations and the function suboptimality into
# one measure for algorithm xx, as in c(x) defined in (60) on page 29 of the arXiv
# paper.

fx1_1fbt = np.array(out1f.fx1)
ce_1f = np.array(out1f.constraints) + (fx1_1fbt>opt)*(fx1_1fbt-opt)/opt

fx1_2fbt = np.array(out2f.fx1)
ce_2f = np.array(out2f.constraints) + (fx1_2fbt > opt) * (fx1_2fbt - opt)/opt

fx_3obt = np.array(out3op.f)
ce_3o = np.array(out3op.constraints) + (fx_3obt > opt) * (fx_3obt - opt)/opt

fcpBT = np.array(outcp.f)
ce_cpBT = np.array(outcp.constraints) + (fcpBT > opt) * (fcpBT - opt)/opt

f_tseng = np.array(outTseng.f)
ce_tseng = np.array(outTseng.constraints) + (f_tseng > opt) * (f_tseng - opt)/opt

f_frb = np.array(outfrb.f)
ce_frb = np.array(outfrb.constraints) + (f_frb> opt) * (f_frb - opt)/opt

# plot of log of c(x) for each method versus elapsed running time

plt.semilogy(out1f.times,ce_1f)
plt.semilogy(out2f.times,ce_2f)
plt.semilogy(out3op.times,ce_3o)
plt.semilogy(outcp.times,ce_cpBT)
plt.semilogy(outTseng.times,ce_tseng)
plt.semilogy(outfrb.times,ce_frb)

plt.xlabel('time (s)')
plt.ylabel('combined optimality+constraint criterion')
plt.grid()

plt.legend(['ps1fbt', 'ps2fbt', 'ada3op','cp-bt','tseng-pd','frb-pd'])

plt.show()

def getResult(ce,tol):
    ce_flip = np.flip(ce,0)
    result = (1.0 * (ce_flip < tol)).argmin()
    result = len(ce) - result
    return result

print("Printing results summary as in Table 2 (page 31)")

ce_tol = 1e-5
# find the first iteration where the algorithm produces an iterate below ce_tol
result_1f = getResult(ce_1f,ce_tol)
result_2f = getResult(ce_2f, ce_tol)
result_3o = getResult(ce_3o, ce_tol)
result_cpBT = getResult(ce_cpBT, ce_tol)
result_tseng = getResult(ce_tseng, ce_tol)
result_frb = getResult(ce_frb, ce_tol)
print("\n\n\n")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXX Results XXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("first iteration where the algorithm produces an iterate with error below "+str(ce_tol))
print("and stays below this tolerance for all subsequent iterations")
print("Iterations 1f: "+str(result_1f))
print("Iterations 2f: " + str(result_2f))
print("Iterations 3op: " + str(result_3o))
print("Iterations cp-bt: " + str(result_cpBT))
print("Iterations tseng: " + str(result_tseng))
print("Iterations frb: " + str(result_frb))
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

if result_1f<len(out1f.times):
    print("Time 1f: " + str(out1f.times[result_1f]))
else:
    print("1f greater than "+str(iter)+" iterations")

if result_2f < len(out2f.times):
    print("Time 2f: " + str(out2f.times[result_2f]))
else:
    print("2f greater than "+str(iter)+" iterations")
if result_3o < len(out3op.times):
    print("Time 3op: " + str(out3op.times[result_3o]))
else:
    print("3op greater than "+str(iter)+" iterations")

if result_cpBT < len(outcp.times):
    print("Time cp-bt: " + str(outcp.times[result_cpBT]))
else:
    print("cp-bt greater than "+str(iter)+" iterations")

if result_tseng < len(outTseng.times):
    print("Time tseng: " + str(outTseng.times[result_tseng]))
else:
    print("tseng greater than "+str(iter)+" iterations")

if result_frb < len(outfrb.times):
    print("Time frb: " + str(outfrb.times[result_frb]))
else:
    print("frb greater than "+str(int(iter))+" iterations")

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


#compare z versus fx1 for ps1f and ps2f
print("Comparing performance of fx1 and fz for ps1fbt and ps2fbt")
algo.compareZandX(out1f,"1f")
algo.compareZandX(out2f,"2f")
