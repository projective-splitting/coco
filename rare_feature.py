import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
import group_lr as gl

import time





def project2Hplane(Az,x1,y1,w,z,x2,y2,Atw,gamma,Amat,Atranspose):
    phi = (Az - x1).T.dot(y1 - w) + (z - x2).T.dot(y2 + Atw)
    Aty1 = Atranspose(y1)
    gradz = Aty1 + y2
    Ax2 = Amat(x2)
    gradw = x1 - Ax2
    #gammaTry = (np.linalg.norm(gradz,2)+1.0)/(np.linalg.norm(gradw,2)+1.0)

    #p = 2.0
    #gamma = gammaTry**(p)
    #print "trial gamma is: " + str(gamma)
    normGradSq = gamma ** (-1) * np.linalg.norm(gradz) ** 2 \
                 + np.linalg.norm(gradw) ** 2
    if normGradSq > 0:
        z = z - gamma ** (-1) * (phi / normGradSq) * gradz
        w = w - (phi / normGradSq) * gradw
    else:
        print "gradient of hyperplane is 0"


    return [z,w,normGradSq,phi]

def ps1fbt(d,p,iter,Amat,rho1,theProx1,alpha2,rho2,theProx2,theGrad,delta,equalRhos,Atranspose,
           gamma,theFunc,stepUp):
    # d is number of nodes in the tree
    # the root node corresponds to the last variable in the d-dimensional vector
    # p is number of features

    z = np.zeros(d)
    w = np.zeros(p)

    x2 = np.zeros(d)
    b2 = theGrad(x2)
    y2 = b2
    thetaHat = np.zeros(d)
    wHat = b2

    times = [0]

    fx2 = []
    rho2s = []
    normGradSqs = []
    phis = []
    for k in range(iter):
        tstartiter = time.time()
        if k%100==0:
            print "iter: "+str(k)
        if equalRhos:
            rho1 = rho2

        Az = Amat(z)
        t1 = Az + rho1 * w
        x1 = theProx1(t1, rho1)
        y1 = (1 / rho1) * (t1 - x1)

        Atw = Atranspose(w)
        phi2 = (z - x2).T.dot(y2 + Atw)
        [x2,y2,b2,rho2] = bt1f(alpha2,x2,z,Atw,stepUp*rho2,b2,theProx2,theGrad,y2,phi2,
                               thetaHat,wHat,delta)
        rho2s.append(rho2)
        #print rho2

        [z,w,normGradSq,phi] = project2Hplane(Az,x1,y1,w,z,x2,y2,Atw,gamma,Amat,Atranspose)
        phis.append(phi)
        normGradSqs.append(normGradSq)
        tfinishiter = time.time()
        times.append(times[-1]+tfinishiter-tstartiter)
        fx2.append(theFunc(x2))



    return [x2,fx2,np.array(times[1:len(times)])-times[1],x1,normGradSqs,rho2s,phis,w]

def bt1f(alpha,x,z,Atw,rho,b,theProx,theGrad,y,phi,thetaHat,wHat,delta):
    w = -Atw
    keepBTing = True
    C1termA = (1 - alpha) * np.linalg.norm(x - thetaHat) + \
              alpha * np.linalg.norm(z - thetaHat)
    C1termB = np.linalg.norm(w - wHat)
    while keepBTing:
        t = (1 - alpha) * x + alpha * z - rho * (b - w)
        xplus = theProx(t,rho)
        aplus = (1/rho)*(t-xplus)
        bplus = theGrad(xplus)
        yhat = aplus + b
        yplus = aplus + bplus

        phi_plus = (z-xplus).T.dot(yplus - w)
        C1 = np.linalg.norm(xplus - thetaHat) - C1termA - rho*C1termB
        C2 = phi_plus - (rho/(2*alpha))*(np.linalg.norm(yplus-w)**2 \
             + alpha*np.linalg.norm(yhat-w)**2)
        C3 = (1-alpha)*(phi - (rho/(2*alpha))*np.linalg.norm(y - w)**2)

        if (C1<=0) & (C2>=C3):
            keepBTing = False
        else:
            rho = rho*delta

    return [xplus,yplus,bplus,rho]


def ps2fbt(d,p,iter,Amat,rho1,theProx1,rho2,theProx2,theGrad,equalRhos,Atranspose,gamma,
           theFunc,deltabt,stepDec,stepUp,plotSteps):
    '''
    2f version, i.e. two forward step, as in the MAPR paper. However, this version
    uses the prox-grad type update which we didn't have at the time of submitting to
    mapr, so this is not MAPR-friendly. Instead, to implement vanilla 2f, you could
    use greedy but with a partition size of 1.
    '''
    # d is number of nodes in the tree
    # the root node corresponds to the last variable in the d-dimensional vector
    # p is number of features

    z = np.zeros(d)
    w = np.zeros(p)
    rho2s = []
    fx2 = []
    times = [0]
    gradNorms = []
    phis = []

    for k in range(iter):
        if k%100==0:
            print "iter: "+str(k)
        t0iter = time.time()
        if equalRhos:
            rho1 = rho2

        Az = Amat(z)
        t1 = Az + rho1 * w
        x1 = theProx1(t1, rho1)
        y1 = (1 / rho1) * (t1 - x1)

        Atw = Atranspose(w)
        [x2,y2,rho2] = bt2f(z,Atw,stepUp*rho2,theProx2,theGrad,deltabt,stepDec)
        rho2s.append(rho2)
        #print rho2

        [z,w,normGradSq,phi] = project2Hplane(Az,x1,y1,w,z,x2,y2,Atw,gamma,Amat,Atranspose)
        gradNorms.append(normGradSq)
        phis.append(phi)
        tenditer = time.time()
        times.append(times[-1]+tenditer-t0iter)
        fx2.append(theFunc(x2))



    return [x2,fx2,np.array(times[1:len(times)])-times[1],rho2s,gradNorms,phis]

def bt2f(z,Atw,rho,theProx,theGrad,deltabt,stepDec):
    w = -Atw
    keepBting = True
    Bz=  theGrad(z)
    while keepBting:
        t = z-rho*(Bz-w)
        xnew = theProx(t,rho)
        a = (1/rho)*(t-xnew)
        b = theGrad(xnew)
        ynew = a+b
        if (deltabt*np.linalg.norm(z-xnew)**2 - (z - xnew).dot(ynew - w) <=0):
            keepBting = False
        else:
            rho = rho*stepDec

    return [xnew, ynew, rho]


def malitskyBT(d,p,iter,proxfstar, gradH, tau, proxg,Func,Kop,Kstarop,delta,beta,
               mu,hfunc,stepInc,plotSteps):

    y = np.zeros(d)
    x = np.zeros(p)
    fy = []
    Kstary = Kstarop(y)
    yFunc = hfunc(y)

    theta = 1
    times = [0]
    taus = []
    t0iter = time.time()

    for k in range(iter):
        if k%100==0:
            print "iter: "+str(k)


        xnext = proxg(x - tau * Kstary,tau)

        taunext = min([tau*np.sqrt(1+theta),stepInc*tau])
        doLine = True
        gradHy = gradH(y)


        while(doLine):

            theta = taunext/tau
            sigmanext = beta*taunext
            xbar = xnext + theta*(xnext - x)
            Kxbar = Kop(xbar)
            ynext = proxfstar(y+sigmanext*(Kxbar - gradHy),sigmanext)
            Kstarynext = Kstarop(ynext)
            yFuncNext = hfunc(ynext)
            C1 = taunext*sigmanext*np.linalg.norm(Kstarynext - Kstary)**2
            C2 = 2*sigmanext*(yFuncNext-yFunc - gradHy.T.dot(ynext - y))
            C3 = delta*np.linalg.norm(ynext - y)**2
            if(C1+C2<=C3):
                doLine = False
            else:
                taunext = taunext*mu



        taus.append(taunext)
        tau = taunext
        x = xnext
        y = ynext
        Kstary = Kstarynext
        yFunc = yFuncNext

        if k%10==0:
            t1iter = time.time()
            times.append(times[-1]+t1iter-t0iter)
            fy.append(Func(y))
            t0iter = time.time()

    if plotSteps:
        plt.plot(taus)
        plt.title('cp steps')
        plt.show()

    print "cp running time, not including func evals: "+str(times[-1])
    return [y,fy,np.array(times[1:len(times)])-times[1]]


def tseng_product(d, p,iter, alpha, theta, theFunc, stepIncrease, proxfstar, proxgstar, gradh,stepDecrease,gamma1,gamma2,Amat,Atranspose):
    # Tseng applied to the primal-dual product-space form
    # this instance is applied to min_x f(Amat*x) + g(x) + h(x)
    # Let p = (w_1,w_2,x), this opt prob is equivalent to finding 0\in B p + A p
    # where Bp = [subf* w_1,subg* w_2,0] and B = [-Amat*x,-x,Amat^T* w_1+w_2+gradh x]
    # note B is Lipschitz monotone but obvs not cocoercive
    # for group lasso, g corresponds to the ell_1 norm (remember the offset is not included),
    # f corresponds to the group lasso norm.
    # proxf and proxg are already defined.
    # So we use Moreau's decomposition to evaluation proxfstar and proxgstar
    # variable metric version: we actually implement a variable metric version because the standard one is very slow
    # on group-LR and portfolio.
    # the updates are now pbar = (I+alpha P B)^{-1}(I-alpha P A) p, p^+ = p - alpha P(B pbar - B p)
    # the stepsize check is alpha^2 ||Apbar - Ap||^2_P <= theta*||pbar - p||_{P^{-1}}^{2}
    # where we will use P = diag(gamma1,gamma2,1)
    # so that ||t||_{P^{-1}}^2 = gamma1^{-1}||t_1||^2+gamma2^{-1}||t_2||^2 + ||t_3||^2

    x = np.zeros(d)
    w1 = np.zeros(p)
    w2 = np.zeros(d)
    Fx = []
    times = [0]
    tstartiter = time.time()

    for k in range(iter):
        if k%100==0:
            print k


        # compute Atild p
        Ap1 = -Amat(x)
        Ap2 = -x
        Ap3 = Atranspose(w1) + w2 + gradh(x)
        keepBT = True
        alpha = alpha * stepIncrease
        while keepBT:

            pbar = theBigProx(w1 - gamma1*alpha * Ap1, w2 - gamma2*alpha*Ap2,
                              x-alpha*Ap3, proxfstar,proxgstar)
            Apbar1 = -Amat(pbar[2])
            Apbar2 = -pbar[2]
            Apbar3 = Atranspose(pbar[0]) + pbar[1] + gradh(pbar[2])

            totalNorm \
                = np.sqrt(gamma1*np.linalg.norm(Apbar1 - Ap1) ** 2 + \
                         gamma2*np.linalg.norm(Apbar2 - Ap2) ** 2 + \
                         np.linalg.norm(Apbar3 - Ap3) ** 2)
            totalNorm2 \
                = np.sqrt(gamma1**(-1)*np.linalg.norm(pbar[0] - w1) ** 2 + gamma2**(-1)*np.linalg.norm(pbar[1] - w2) ** 2 +
                          np.linalg.norm(pbar[2] - x) ** 2)

            if (alpha * totalNorm <= theta * totalNorm2):
                keepBT = False
            else:
                alpha = stepDecrease * alpha

        w1 = pbar[0] - gamma1 * alpha * (Apbar1 - Ap1)
        w2 = pbar[1] - gamma2 * alpha * (Apbar2 - Ap2)
        x = pbar[2] - alpha * (Apbar3 - Ap3)
        if k%10==0:
            tenditer = time.time()
            times.append(times[-1]+tenditer-tstartiter)
            Fx.append(theFunc(x))
            tstartiter = time.time()



    return [x, Fx, np.array(times[1:len(times)])-times[1]]

def theBigProx(a, b, c, proxfstar,proxgstar):
    out1 = proxfstar(a)
    out2 = proxgstar(b)
    out3 = c
    return [out1, out2, out3]

def for_reflect_back(p,proxfstar,proxgstar,gradh,iter,x0,gamma0,gamma1,lam,stepIncrease,delta,stepDecrease,theFunc,verbose,Amat,Atranspose):
    #basically apply the forward-reflected-backward method to the same primal-dual product-space inclusion
    #as we did for Tseng-pd.
    #as with Tseng-pd we use the variable metric: diag(gamma1,gamma2,1)
    #note that the backtrack check condition is now lambda||B p^{k+1} - B p^k||<=0.5*delta||p^{k+1}-p^k||_{P^{-1}}
    d = len(x0)
    x = x0
    w0 = np.zeros(p)
    w1 = np.zeros(d)

    B0 = -Amat(x)
    B1 = -x
    B2 = Atranspose(w0) + w1 + gradh(x)

    B0old = B0
    B1old = B1
    B2old = B2
    F = []
    times = [0]
    lamOld = lam
    lamMax = lam
    tstartIter = time.time()

    for k in range(iter):
        if (k%100==0) & verbose:
            print k



        doBackTrack = True



        while doBackTrack:
            toProx0 = w0 - gamma0*lam*B0 - gamma0*lamOld*(B0 - B0old)
            toProx1 = w1 - gamma1*lam*B1 - gamma1*lamOld*(B1 - B1old)
            toProx2 = x -         lam*B2 -        lamOld*(B2 - B2old)

            phat = theBigProx(toProx0, toProx1, toProx2, proxfstar, proxgstar)

            Bhat0 = -Amat(phat[2])
            Bhat1 = -phat[2]
            Bhat2 = Atranspose(phat[0]) + phat[1] + gradh(phat[2])


            normLeft = np.linalg.norm(Bhat0-B0)**2 + np.linalg.norm(Bhat1-B1)**2 + np.linalg.norm(Bhat2-B2)**2
            normRight = gamma0**(-1)*np.linalg.norm(phat[0]-w0)**2 + gamma1**(-1)*np.linalg.norm(phat[1]-w1)**2 + np.linalg.norm(phat[2]-x)**2

            if lam*np.sqrt(normLeft)<=0.5*delta*np.sqrt(normRight):
                doBackTrack = False
            else:
                lam = lam*stepDecrease

        B0old = B0
        B1old = B1
        B2old = B2

        B0 = Bhat0
        B1 = Bhat1
        B2 = Bhat2

        w0 = phat[0]
        w1 = phat[1]
        x = phat[2]
        lamOld = lam
        lam = stepIncrease * lam
        #print "lam is: "+str(lam)
        #print "lamOld is: "+str(lamOld)
        if k%10==0:

            tendIter = time.time()
            times.append(times[-1] + tendIter-tstartIter)


            F.append(theFunc(x))
            tstartIter = time.time()

    print "frb running time (excluding func evals): "+str(times[-1])
    return [x,F,np.array(times[1:len(times)])-times[1]]



def getParams(folder):
    fh = open(folder + 'params.txt', 'r')
    params = {}
    for row in fh:
        #print row
        split_entry = row.split(' = ')
        if len(split_entry)>1:
            params[split_entry[0]] = split_entry[1]
    return params
