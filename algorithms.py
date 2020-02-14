import numpy as np
from matplotlib import pyplot as plt
import time

class Results:
    '''
    A results struct to return from each function
    '''
    pass

class InitPoint:
    def __init__(self,x1,x2,z,w):
        self.x1 = x1
        self.x2 = x2
        self.z = z
        self.w = w

###################################################
#       Algorithms #
###################################################


def PS1f_bt(theFunc,theProx1,theProx2,theGrad,init,iter=1000,alpha=0.1,
            rho1=1.0,rho2=1.0,gamma=1.0,hyper_resid=-1,stepDecrease=0.7,stepIncrease=1.0,
            equalRho=True,verbose=True):
    '''
    The algorithm proposed in this paper. projective splitting with one-forward-step.
    This variant features backtracking on the smooth loss.
    parameters:
        d: dimension of the optimization variable/vector
        iter: number of iterations to run the method. Note that there is no stopping criterion
              implemented, so you must set iter to terminate the method.
        alpha: averaging parameter
        rho1: stepsize
        rho2: stepsize
        theProx1: a function with two arguments, the first is the input to the prox,
                  the second is the stepsize. It is the prox of f1
        theGrad: a function of one argument, returns the gradient of the smooth term
        theProx2: a function with two arguments, the first is the input to the prox,
                  the second is the stepsize. It is the prox of f2
        theFunc: returns the function value of the entire objective
        gamma: primal-dual parameter
        hyper_resid: (portfolio optimiz) function of one parameter, computes the error/residual w.r.t.
        the hyperplane constraint
        stepDecrease: how much to decrease the stepsize by in the backtracking linesearch, eg: 0.7
        equalRho: set stepsizes to be equal (non-backtracked equal to backtracked)
        verbose: whether to print out iteration number
        init: Initial point. A structure with four fields: x1, x2, z, and w.
    '''

    x1 = init.x1
    x2 = init.x2
    z = init.z
    w = init.w

    b1 = theGrad(x1)
    y1 = b1
    yhat = np.copy(w)

    phi_1 = (z - x1).T.dot(y1 - w)

    thetaHat = np.copy(x1)
    wHat = b1

    fx1 = []
    fz = []
    rhos = []
    grad_evals = []
    constraintsErr = []
    grad_evalsSofar = 1
    times = [0]


    for k in range(iter):
        if (k%100==0) & verbose:
            print(k)

        tstartiter = time.time()

        if stepIncrease>1.0:
            maxFac = 1 + alpha*(np.linalg.norm(yhat-w)**2)/(np.linalg.norm(y1-w)**2)
            factor = min([maxFac,stepIncrease])
            rho1 = factor*rho1

        [x1,y1,b1,rho1,grad_evalsNew,yhat] = bt_1f(alpha,x1,z,w,rho1,b1,theProx1,theGrad,
                                                   y1,phi_1,thetaHat,wHat,stepDecrease)
        grad_evalsSofar +=grad_evalsNew
        grad_evals.append(grad_evalsSofar)

        if equalRho:
            rho2 = rho1

        t2 = z - rho2 * w
        x2 = theProx2(t2,rho2)
        y2 = (1 / rho2) * (t2 - x2)

        phi = (z - x1).T.dot(y1 - w) + (z - x2).T.dot(y2 + w)
        gradz = y1 + y2
        gradw = x1 - x2
        normGradsq = gamma ** (-1) * np.linalg.norm(gradz) ** 2 + np.linalg.norm(gradw) ** 2

        if (normGradsq > 1e-20):
            z = z - gamma ** (-1) * (phi / normGradsq) * gradz
            w = w - (phi / normGradsq) * gradw

        phi_1 = (z - x1).T.dot(y1 - w)

        tendIter = time.time()
        times.append(times[-1]+tendIter-tstartiter)

        fx1.append(theFunc(x1))
        fz.append(theFunc(z))
        rhos.append(rho1)

        if hyper_resid != -1:
            simplError = abs(sum(x1)-1.0)
            posErr = -min([min(x1),0])
            hyperErr = hyper_resid(x1)
            constraintsErr.append(simplError+posErr+hyperErr)

    out = Results()
    out.fz = fz
    out.fx1 = fx1
    out.z = z
    out.x1 = x1
    out.x2 = x2
    out.rhos = rhos
    out.grad_evals = grad_evals
    out.times = np.array(times[1:len(times)])-times[1]
    out.constraints = constraintsErr

    return out

def bt_1f(alpha,x1,z,w,rho,b_old,theProx1,theGrad,y_old,phiOld,thetaHat,wHat,stepDecrease):
    '''
        backtracking procedure for ps1f. Internal subroutine of PS1f_bt().
    '''

    keepBTing = True
    C1termA = (1 - alpha) * np.linalg.norm(x1 - thetaHat) + \
                alpha * np.linalg.norm(z - thetaHat)
    C1termB = np.linalg.norm(w-wHat)
    grad_evals = 0
    while(keepBTing):
        tnew = (1 - alpha) * x1 + alpha * z - rho * (b_old - w)
        xnew = theProx1(tnew, rho)
        a1 = (1 / rho) * (tnew - xnew)
        bnew = theGrad(xnew)
        grad_evals+=1
        yhat = a1 + b_old
        ynew = a1 + bnew

        phi = (z - xnew).dot(ynew - w)

        C1 = np.linalg.norm(xnew - thetaHat) - C1termA - rho*C1termB
        C2 = phi  - (rho/(2*alpha))*(np.linalg.norm(ynew - w)**2 \
              + alpha*np.linalg.norm(yhat - w)**2)
        C3 = (1-alpha)*(phiOld - (rho/(2*alpha))*np.linalg.norm(y_old - w)**2)

        if( (C2>= C3) & (C1 <= 0) ):
            # backtracking successful
            keepBTing = False
        else:
            rho = rho*stepDecrease

    return [xnew,ynew,bnew,rho,grad_evals,yhat]




def PS1f_bt_comp(init,iter,G,theProx1,theProx2,theGrad,Gt,theFunc,
                 rho1=1.0,rho2=1.0,alpha2=0.1,stepDecrease=0.7,equalRhos=True,
                 gamma=1.0,stepIncrease=1.0):
    '''
    The algorithm proposed in this paper. projective splitting with one-forward-step.
    Essentially the same as PS1f_bt() defined above, however this variant allows for composition
    with a linear operator as in the rare feature selection problem. So it solves
    min_x f_1(Gx)+f_2(x)+h_2(x)
    '''
    # d is number of nodes in the tree
    # the root node corresponds to the last variable in the d-dimensional vector
    # p is number of features

    z = init.z
    w = init.w
    d = len(z)
    p = len(w)

    x2 = init.x2
    b2 = theGrad(x2)
    y2 = b2
    thetaHat = np.copy(z)
    wHat = b2
    yhat = np.copy(wHat)

    times = [0]

    fx2 = []
    fz = []
    rho2s = []

    for k in range(iter):
        tstartiter = time.time()
        if k%100==0:
            print "iter: "+str(k)
        if equalRhos:
            rho1 = rho2

        Gz = G(z)
        t1 = Gz + rho1 * w
        x1 = theProx1(t1, rho1)
        y1 = (1 / rho1) * (t1 - x1)

        Gtw = Gt(w)
        phi2 = (z - x2).T.dot(y2 + Gtw)

        if stepIncrease>1.0:
            maxFac = 1 + alpha2*(np.linalg.norm(yhat+Gtw)**2)/(np.linalg.norm(y2+Gtw)**2)
            factor = min([maxFac,stepIncrease])
            rho2 = factor*rho2

        [x2,y2,b2,rho2,_,yhat] = bt_1f(alpha2, x2,z,-Gtw,rho2,b2,theProx2,theGrad,y2,phi2,
                                thetaHat,wHat,stepDecrease)

        rho2s.append(rho2)

        phi = (Gz - x1).T.dot(y1 - w) + (z - x2).T.dot(y2 + Gtw)
        Gty1 = Gt(y1)
        gradz = Gty1 + y2
        Gx2 = G(x2)
        gradw = x1 - Gx2
        normGradSq = gamma ** (-1) * np.linalg.norm(gradz) ** 2 \
                     + np.linalg.norm(gradw) ** 2

        if normGradSq > 0:
            z = z - gamma ** (-1) * (phi / normGradSq) * gradz
            w = w - (phi / normGradSq) * gradw


        tfinishiter = time.time()
        times.append(times[-1]+tfinishiter-tstartiter)
        fx2.append(theFunc(x2))
        fz.append(theFunc(z))

    out = Results()
    out.fx2 = fx2
    out.fz = fz
    out.times = np.array(times[1:len(times)])-times[1]
    out.x2 = x2
    out.x1 = x1
    out.rhos = rho2s


    return out




def PS2f_bt(theFunc,theGrad,theProx1,theProx2,init,iter=1000,rho1=1.0,rho2=1.0,
            gamma=1.0,Delta=1.0,hyper_resid=-1,stepDecrease=0.7,stepIncrease = 1.0,
            equalRho_2f=True,verbose=True):
    '''
    projective splitting with two forward steps and backtracking
    same parameters as described in PS1f_bt() with the addition of Delta
    which defaults to 1.0 and is used in the backtracking procedure
    '''
    z = init.z
    w = init.w

    fx1 = []
    fz = []
    gradEvals = []
    gradsRunning = 0
    times = [0]
    rhosBT = []
    constraintErr = []

    for k in range(iter):
        if (k%100==0) & verbose:
            print(k)
        tstartiter = time.time()
        rho1 = rho1*stepIncrease
        [x1,y1,rho1,gradsNew] = bt_2f(z,rho1,w,theProx1,theGrad,Delta,stepDecrease)
        gradsRunning +=gradsNew
        gradEvals.append(gradsRunning)
        rhosBT.append(rho1)

        if equalRho_2f:
            rho2 = rho1

        t2 =  z - rho2 * w
        x2 = theProx2(t2,rho2)
        y2 = (1 / rho2) * (t2 - x2)

        phi = (z - x1).T.dot(y1 - w) + (z - x2).T.dot(y2 + w)
        gradz = y1 + y2
        gradw = x1 - x2
        normGradsq = gamma ** (-1) * np.linalg.norm(gradz) ** 2 + np.linalg.norm(gradw) ** 2

        if (normGradsq > 0):
            z = z - gamma ** (-1) * (phi / normGradsq) * gradz
            w = w - (phi / normGradsq) * gradw

        tenditer = time.time()

        times.append(times[-1]+tenditer-tstartiter)
        fx1.append(theFunc(x1))
        fz.append(theFunc(z))
        if hyper_resid != -1:
            simplError = abs(sum(x1) - 1.0)
            posErr = -min([min(x1), 0])
            hyperErr = hyper_resid(x1)
            constraintErr.append(simplError+posErr+hyperErr)


    out = Results()
    out.fz = fz
    out.fx1 = fx1
    out.z = z
    out.x1 = x1
    out.x2 = x2
    out.grad_evals = gradEvals
    out.times = np.array(times[1:len(times)])-times[1]
    out.constraints = constraintErr
    out.rhos = rhosBT
    return out



def PS2f_bt_comp(init,iter,G,theProx1,theProx2,theGrad,Gt,theFunc,
                 stepDec=0.7,stepUp=1.0,rho1 = 1.0,rho2=1.0,
                 equalRhos=True,gamma=1.0,deltabt=1.0):
    '''
    This is the same as PS2f_bt() above except it allows for composition with a linear term
    as in the rare feature selection problem and so can solve problems of the form
    min_x f_1(G_1x)+f_2(x)+h_2(x)
    '''
    # d is number of nodes in the tree
    # the root node corresponds to the last variable in the d-dimensional vector
    # p is number of features

    z = init.z
    w = init.w
    d = len(z)
    p = len(w)
    rho2s = []
    fx2 = []
    fz = []
    times = [0]

    for k in range(iter):
        if k%100==0:
            print "iter: "+str(k)
        t0iter = time.time()
        if equalRhos:
            rho1 = rho2

        Gz = G(z)
        t1 = Gz + rho1 * w
        x1 = theProx1(t1, rho1)
        y1 = (1 / rho1) * (t1 - x1)

        Gtw = Gt(w)
        [x2,y2,rho2,_] = bt_2f(z,stepUp*rho2,-Gtw,theProx2,theGrad,deltabt,stepDec)
        rho2s.append(rho2)

        phi = (Gz - x1).T.dot(y1 - w) + (z - x2).T.dot(y2 + Gtw)
        Gty1 = Gt(y1)
        gradz = Gty1 + y2
        Gx2 = G(x2)
        gradw = x1 - Gx2

        normGradSq = gamma ** (-1) * np.linalg.norm(gradz) ** 2 \
                     + np.linalg.norm(gradw) ** 2
        if normGradSq > 0:
            z = z - gamma ** (-1) * (phi / normGradSq) * gradz
            w = w - (phi / normGradSq) * gradw



        tenditer = time.time()
        times.append(times[-1]+tenditer-t0iter)
        fx2.append(theFunc(x2))
        fz.append(theFunc(z))

    out = Results()
    out.fz = fz
    out.fx2 = fx2
    out.x1 = x1
    out.x2 = x2
    out.times = np.array(times[1:len(times)])-times[1]
    out.rhos = rho2s

    return out


def bt_2f(z,rho, w,theProx1,theGrad,Delta,stepDecrease):

    keepBTing = True
    Bz = theGrad(z)
    gradsNew = 1
    while(keepBTing):
        t = z - rho * (Bz - w)
        xnew = theProx1(t, rho)
        a = (1 / rho) * (t - xnew)
        b = theGrad(xnew)
        gradsNew += 1
        ynew = a + b

        if(Delta*np.linalg.norm(z-xnew)**2 - (z - xnew).dot(ynew - w) <=0):
            #backtracking complete
            keepBTing = False
        else:
            rho = stepDecrease*rho

    return [xnew,ynew,rho,gradsNew]



def adap3op(proxg,gradf,f_smooth,theFunc,proxh,f_smooth_smart,init,stepIncrease = 1.0,
            lip_const = 0.0,iter=1000,gamma=1.0,tau=0.7,hyper_resid=-1,verbose=True):
    '''
        Adaptive version of three operator splitting (ada3po in the paper)
    '''

    z = init.z
    u = np.copy(init.z)

    Fx = []
    Qt = 0.0
    fx = 0.0
    gradEvals = [0]
    funcEvals = [0]
    times = [0]
    constraintErr = []

    for k in range(iter):
        if (k%100==0) & verbose:
            print(k)
        tstartiter = time.time()
        [gradfz,Az] = gradf(z)
        gradEvals.append(gradEvals[-1]+1)

        fz = f_smooth_smart(z,Az)




        if stepIncrease>1.0:
            deltat = Qt - fx
            gamma1 = np.sqrt(gamma ** 2 + gamma * deltat * (2 * lip_const) ** (-2))
            gamma = min([gamma1,stepIncrease*gamma])

        newfEvals = 0
        doLine = True
        while(doLine):
            x = proxg(z - gamma*u - gamma*gradfz,gamma)
            Qt = fz + gradfz.T.dot(x - z)+(1/(2*gamma))*np.linalg.norm(x-z)**2
            fx = f_smooth(x)
            newfEvals += 1
            if(fx<=Qt):
                doLine = False
            else:
                gamma = gamma*tau

        z = proxh(x+gamma*u,gamma)
        u = u + (x-z)/gamma
        tenditer = time.time()

        funcEvals.append(funcEvals[-1]+newfEvals)
        if hyper_resid != -1:
            simplError = abs(sum(x) - 1.0)
            posErr = -min([min(x), 0])
            hyperErr = hyper_resid(x)
            constraintErr.append(simplError+posErr+hyperErr)

        times.append(times[-1]+tenditer-tstartiter)
        Fx.append(theFunc(x))


    out = Results()
    out.x = x
    out.f = Fx
    out.times = np.array(times[1:len(times)])-times[1]
    out.grad_evals = gradEvals[1:len(gradEvals)]
    out.func_evals = funcEvals[1:len(funcEvals)]
    out.constraints = constraintErr
    return out




def cpBT(proxfstar, gradH, proxg, Func, hfunc, init, iter=1000, tau=1.0,delta=0.99,
         beta=1.0,mu=0.7,hyper_resid=-1,stepInc=1.0,verbose=True,K=-1,Kstar=-1):
    '''
        Chambolle-Pock Primal Dual splitting with backtracking
        this algorithm solves the following optimization problem:
        min_y {gstar(-K^*y) + fstar(y) + h(y)}
        where h is Lipschitz diffentiable and the other two functions are proximable
        and K is a linear operator. K is controlled by the input K and Kstar
        plug-in routines. If these are set to -1, no linear operator is present,
        i.e. K is the identity.
        For example in our application to portfolio optimization, we will set
        -> fstar(y) = ind(simplex)
        -> gstar(-y) = ind(hyperplane constraint, <m,y> >= r), which means gstar(t) = ind(hyperplane constraint <m,t> <= -r)
        -> h(y) = 0.5*y^T*Q*y
        # TODO:
        - Handle the matrix, as in rare features
    '''


    y = init.x1
    x = init.x2
    fy = []
    [yFunc, Ay] = hfunc(y)

    if K ==-1:
        Kstary = y
    else:
        Kstary = Kstar(y)

    theta = 1
    func_evals = [1]
    grad_evals = [0]
    times = [0]
    constraintErr = []


    for k in range(iter):
        if (k%100==0) & verbose:
            print(k)

        tstartIter = time.time()
        xnext = proxg(x - tau * Kstary, tau)
        taunext = min([tau*np.sqrt(1+theta),stepInc*tau])

        doLine = True
        gradHy = gradH(Ay)
        grad_evals.append(grad_evals[-1]+1)

        newFuncEvals = 0
        while(doLine):

            theta = taunext/tau
            sigmanext = beta*taunext
            xbar = xnext + theta*(xnext - x)
            if K == -1:
                Kxbar = xbar
            else:
                Kxbar = K(xbar)

            ynext = proxfstar(y+sigmanext*(Kxbar - gradHy),sigmanext)
            [yFuncNext,Ay] = hfunc(ynext)
            if K ==-1:
                KstarYnext = ynext
            else:
                KstarYnext = Kstar(ynext)

            newFuncEvals += 1

            C1 = taunext*sigmanext*np.linalg.norm(KstarYnext - Kstary)**2
            C2 = 2*sigmanext*(yFuncNext-yFunc - gradHy.T.dot(ynext - y))
            C3 = delta*np.linalg.norm(ynext - y)**2
            if(C1+C2<=C3):
                doLine = False
            else:
                taunext = mu*taunext


        tau = taunext
        x = xnext
        y = ynext
        Kstary = KstarYnext
        yFunc = yFuncNext

        tendIter = time.time()
        times.append(times[-1]+tendIter-tstartIter)
        func_evals.append(func_evals[-1]+newFuncEvals)

        fy.append(Func(y))
        if hyper_resid != -1:
            simplError = abs(sum(y) - 1.0)
            posErr = -min([min(y), 0])
            hyperErr = hyper_resid(y)
            constraintErr.append(simplError + posErr + hyperErr)



    out = Results()
    out.y = y
    out.f = fy
    out.func_evals = func_evals[1:len(func_evals)]
    out.grad_evals = grad_evals[1:len(func_evals)]
    out.times = np.array(times[1:len(times)])-times[1]
    out.constraints = constraintErr
    return out

def tseng_product(theFunc, proxfstar, proxgstar, gradh, init, iter=1000, alpha=1.0,
                  theta=0.99, stepIncrease=1.0, stepDecrease=0.7,hyper_resid=-1,
                  gamma1=1.0,gamma2=1.0,verbose=True,G=[],Gt=[]):
    '''
    Tseng applied to the primal-dual product-space form
    this instance is applied to min_x f(x) + g(x) + h(x)
    Let p = (w_1,w_2,x), this opt prob is equivalent to finding 0\in B p + A p
    where Bp = [subf* w_1,subg* w_2,grad h x] and B = [-x,-x,w_1+w_2+gradh x]
    note B is Lipschitz monotone but obvs not cocoercive
    for portfolio, f corresponds to the simplex, g corresponds to the hyperplane.
    proxf and proxg are projections onto these sets.
    So we use moreau's decomposition to evaluation proxfstar and proxgstar
    note that the linesearch exit condition must become: alpha||Axbar - Ax||_P <= delta||xbar-x||_{P^{-1}}
    '''

    x = init.z
    w1 = init.w
    w2 = np.copy(x)


    Fx = []
    grad_evals = [0]
    times = [0]
    constraintErr = []


    for k in range(iter):
        if (k%100==0) & verbose:
            print(k)

        tstartiter = time.time()
        # compute Ap
        if G == []:
            Ap1 = -x
        else:
            Ap1 = -G(x)

        Ap2 = -x
        if G == []:
            Ap3 = w1 + w2 + gradh(x)
        else:
            Ap3 = Gt(w1) + w2 + gradh(x)

        newGrads = 1
        keepBT = True
        alpha = alpha * stepIncrease
        while (keepBT):

            pbar = theBigProx(w1 - gamma1 * alpha * Ap1, w2 - gamma2 * alpha * Ap2,
                              x - alpha * Ap3, proxfstar,proxgstar,alpha,gamma1,gamma2)
            if G == []:
                Apbar1 = -pbar[2]
            else:
                Apbar1 = - G(pbar[2])

            Apbar2 = -pbar[2]
            if G == []:
                Apbar3 = pbar[0] + pbar[1] + gradh(pbar[2])
            else:
                Apbar3 = Gt(pbar[0]) + pbar[1] + gradh(pbar[2])

            newGrads += 1
            totalNorm \
                = np.sqrt(gamma1*np.linalg.norm(Apbar1 - Ap1) ** 2 +
                          gamma2*np.linalg.norm(Apbar2 - Ap2) ** 2 +
                          np.linalg.norm(Apbar3 - Ap3) ** 2)
            totalNorm2 \
                = np.sqrt(gamma1**(-1)*np.linalg.norm(pbar[0] - w1) ** 2 +
                          gamma2**(-1)*np.linalg.norm(pbar[1] - w2) ** 2 +
                          np.linalg.norm(pbar[2] - x) ** 2)

            if (alpha * totalNorm <= theta * totalNorm2):
                keepBT = False
            else:
                alpha = stepDecrease * alpha

        w1 = pbar[0] - gamma1 * alpha * (Apbar1 - Ap1)
        w2 = pbar[1] - gamma2 * alpha * (Apbar2 - Ap2)
        x = pbar[2] - alpha * (Apbar3 - Ap3)
        tenditer = time.time()
        times.append(times[-1]+tenditer-tstartiter)
        Fx.append(theFunc(x))
        grad_evals.append(grad_evals[-1]+newGrads)

        if hyper_resid!=-1:
            simplError = abs(sum(x) - 1.0)
            posErr = -min([min(x), 0])
            hyperErr = hyper_resid(x)
            constraintErr.append(simplError + posErr + hyperErr)




    out = Results()
    out.x = x
    out.f = Fx
    out.grad_evals = grad_evals[1:len(grad_evals)]
    out.times = np.array(times[1:len(times)])-times[1]
    out.constraints = constraintErr
    return out


def theBigProx(a, b, c, proxfstar,proxgstar,alpha,gamma1,gamma2):
    '''
        internal function for tseng_product()
    '''
    out1 = proxfstar(a, gamma1*alpha)
    out2 = proxgstar(b, gamma2*alpha)
    out3 = c
    return [out1, out2, out3]



def for_reflect_back(theFunc,proxfstar,proxgstar,gradh,init,iter=1000,gamma0=1.0,
                     gamma1=1.0,lam=1.0,stepIncrease=1.0,delta=0.99,
                     stepDecrease=0.7,hyper_resid=-1,verbose=True,G=[],Gt=[]):
    '''
        Apply the forward-reflected-backward method to the same primal-dual product-space inclusion
        as we did for Tseng-pd.
        as with Tseng-pd we use the variable metric: diag(gamma1,gamma2,1)
        note that the backtrack check condition is now lambda||B p^{k+1} - B p^k||<=0.5*delta||p^{k+1}-p^k||_{P^{-1}}
    '''



    x = init.z
    w0 = init.w
    w1 = np.copy(x)

    if G == []:
        B0 = -x
    else:
        B0 = -G(x)

    B1 = -x
    if G == []:
        B2 = w0 + w1 + gradh(x)
    else:
        B2 = Gt(w0) + w1 + gradh(x)


    B0old = B0
    B1old = B1
    B2old = B2
    F = []
    times = [0]
    grad_evals = [1]
    lamOld = lam
    lamMax = lam


    constraintErr = []

    for k in range(iter):
        if (k%100==0) & verbose:
            print(k)
        tstartIter = time.time()

        doBackTrack = True
        newGrads = 0
        while doBackTrack:
            toProx0 = w0 - gamma0*lam*B0 - gamma0*lamOld*(B0 - B0old)
            toProx1 = w1 - gamma1*lam*B1 - gamma1*lamOld*(B1 - B1old)
            toProx2 = x -         lam*B2 -        lamOld*(B2 - B2old)

            phat = theBigProx(toProx0, toProx1, toProx2, proxfstar, proxgstar,
                              lam, gamma0, gamma1)

            if G == []:
                Bhat0 = -phat[2]
            else:
                Bhat0 = -G(phat[2])

            Bhat1 = -phat[2]
            if G == []:
                Bhat2 = phat[0] + phat[1] + gradh(phat[2])
            else:
                Bhat2 = Gt(phat[0]) + phat[1] + gradh(phat[2])

            newGrads += 1

            normLeft = np.linalg.norm(Bhat0-B0)**2 + np.linalg.norm(Bhat1-B1)**2\
                        + np.linalg.norm(Bhat2-B2)**2
            normRight = gamma0**(-1)*np.linalg.norm(phat[0]-w0)**2\
                        + gamma1**(-1)*np.linalg.norm(phat[1]-w1)**2\
                        + np.linalg.norm(phat[2]-x)**2

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

        tendIter = time.time()
        times.append(times[-1] + tendIter-tstartIter)
        F.append(theFunc(x))
        if hyper_resid!=-1:
            simplError = abs(sum(x) - 1.0)
            posErr = -min([min(x), 0])
            hyperErr = hyper_resid(x)
            constraintErr.append(simplError + posErr + hyperErr)


        grad_evals.append(grad_evals[-1]+newGrads)
        multsIter = 0

    out = Results()
    out.x = x
    out.f = F
    out.times = np.array(times[1:len(times)])-times[1]
    out.grad_evals = grad_evals[1:len(grad_evals)]
    out.constraints = constraintErr

    return out

############################################################################
#       additional routines
############################################################################

def proxL1(a,rho, lam):
    '''
    proximal operator for lambda||x||_1 with stepsize rho
    aka soft thresholding operator
    '''
    rholam = rho * lam
    x = (a> rholam)*(a-rholam)
    x+= (a<-rholam)*(a+rholam)
    return x

def proxL1_v2(a,thresh):
    '''
    variant of the above
    '''
    x = (a> thresh)*(a-thresh)
    x+= (a<-thresh)*(a+thresh)
    return x

def projLInf(x,thrsh):
    return (x>=-thrsh)*(x<=thrsh)*x + (x>thrsh)*thrsh - (x<-thrsh)*thrsh

def projSimplex(a):
    '''
    Project onto the simplex
    Implementation based on ``projection on a simplex" 2011, Chen and Ye.
    '''

    n = len(a)
    y = np.sort(a)
    ti = 0
    s = 0
    flag = False
    for i in range(n-1,0,-1):
        s += y[i]
        ti = (s-1)/(n - i)
        if(ti>= y[i-1]):
            that = ti
            flag = True
            break

    if(flag==False):
        s += y[0]
        that = (s-1)/n

    return posPart(a - that)

def posPart(y):
    '''
    used in project to simplex function above
    '''
    return (y>=0)*y


def projHplane(a,m,r):
    '''
    project to the hyperplane defined in the portfolio problem.
    '''
    mTa = m.T.dot(a)
    if(mTa < r):
        lamda = (r - mTa)/(np.linalg.norm(m,2)**2)
        return (a + lamda*m)
    else:
        return a

def projHplaneCP(a,m,r):
    '''
    project to the hyperplane defined in the portfolio problem.
    cp-bt version projects onto <m,a><=-r
    '''

    mTa = m.T.dot(a)
    if(mTa > -r):
        lamda = (-r - mTa)/(np.linalg.norm(m,2)**2)
        return (a + lamda*m)
    else:
        return a

def runCVX_portfolio(d,Q,m,r):
    '''
    run CVX for the portfolio problem.
    '''
    print("setting up cvx...")
    x_cvx = cvx.Variable(d)
    f = 0.5*cvx.quad_form(x_cvx,Q)
    constraints = [x_cvx>=0,cvx.sum_entries(x_cvx)==1.0,m.T*x_cvx>=r]
    prob = cvx.Problem(cvx.Minimize(f),constraints)
    print("solving with cvx...")
    prob.solve(verbose=True)
    opt = prob.value
    xopt = x_cvx.value

    return [opt,xopt]


def compareZandX(out,alg):
    plt.plot(out.times,out.fz,out.times,out.fx1)
    plt.xlabel("time (s)")
    plt.ylabel("function value")
    plt.title(alg)
    plt.legend(['fz','fx1'])
    plt.show()
