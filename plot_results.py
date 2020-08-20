import numpy as np
import pickle
from matplotlib import pyplot as plt
import scipy.sparse as sp


for lam in [1e-4]:
    loss = "log"
    if loss == "log":
        with open('saved_results_log_'+str(lam),'rb') as file:
            cache = pickle.load(file)
    else:
        with open('saved_results_'+str(lam),'rb') as file:
            cache = pickle.load(file)

    outcp = cache.get('cp')
    if outcp is None:
        outcp = cache['outcp']

    f_psbg = cache['f_psbg']
    t_psbg = cache['t_psbg']

    print("================")
    print("================")
    print("================")
    print("classification errors...")




    getClassErr = True
    if getClassErr :
        path2Data = 'data/trip_advisor/'
        S_train = sp.load_npz(path2Data+'S_train.npz')
        S_test = sp.load_npz(path2Data+'S_test.npz')

        S_A     = sp.load_npz(path2Data+'S_A.npz')
        y_train = np.load(path2Data+'y_train.npy')
        y_test = np.load(path2Data+'y_test.npy')

        print(f"z ps2f_g sparsity = {sum(abs(cache['z_2fg'])>1e-5)}")
        print(f"Hz ps2f_g sparsity = {sum(abs(S_A.dot(cache['z_2fg'][1:]))>1e-5)}")

        y_train_class = 2*(y_train==5)-1
        y_test_class = 2*(y_test==5)-1
        def getClassify(x,algo):
            train_pred = np.sign(x[0] + S_train.dot(S_A.dot(x[1:])))
            train_err = sum(train_pred != y_train_class)/len(y_train_class)
            test_pred = np.sign(x[0] + S_test.dot(S_A.dot(x[1:])))
            test_err = sum(test_pred != y_test_class)/len(y_test_class)
            print(f"{algo} training error = {train_err}")
            print(f"{algo} testing error = {test_err}")

        getClassify(cache['z_2fg'],"ps2f_g")
        getClassify(outcp.y,"cp")
        getClassify(cache['outtseng'].x,"tseng")
        getClassify(cache['outfrb'].x,"FRB")

    print("================")
    print("================")
    print("================")
    print("objective function values...")


    history_2fg = cache['history_2fg']
    f_2fg = history_2fg[0]
    t_2fg = history_2fg[1]
    plt.plot(t_2fg,f_2fg)
    if lam == 1e-8:
        markFreq=20
    else:
        markFreq=200
    plt.plot(t_psbg,f_psbg,marker='o',markevery=markFreq)
    markFreq=50
    plt.plot(outcp.times,np.array(outcp.f),marker='s',markevery=markFreq)

    plt.plot(cache['outfrb'].times,cache['outfrb'].f,markevery=markFreq,marker='v')
    plt.plot(cache['outtseng'].times,cache['outtseng'].f,markevery=markFreq,marker='d')

    plt.plot(cache['history_2fg_ne'][1],cache['history_2fg_ne'][0])

    fntSze=14
    plt.xlabel('times (s)',fontsize=fntSze)
    plt.title('objective function values',fontsize=fntSze)

    plt.legend(['psf-g','psb-g','cp-bt','frb','tseng-pd','noembed'],fontsize=fntSze)
    plt.xlim((0,600))
    plt.grid()
    plt.xlabel('times (s)')


    if lam == 1e-8:
        prob = "dense"
    elif lam == 1e-6:
        prob = "med"
    else:
        prob = "sparse"
    figname = 'figs/raw_'+prob+'.pdf'
    #plt.savefig(figname,format='pdf')

    plt.show()
    #plt.close()
    print("greedy compare...")

    history_2fr = cache['history_2fr']
    f_2fr = history_2fr[0]
    t_2fr = history_2fr[1]
    history_2fc = cache['history_2fc']
    f_2fc = history_2fc[0]
    t_2fc = history_2fc[1]

    plt.plot(t_2fg,f_2fg)
    markFreq=50
    plt.plot(cache['t_ps2fembed'],cache['f_ps2fembed'],'k>-',markevery=markFreq)
    markFreq=100
    plt.plot(t_2fr,f_2fr,'m*-',markevery=markFreq)
    plt.plot(t_2fc,f_2fc,'ch-',markevery=markFreq)

    plt.legend(['psf-g','psf-1','psf-r','psf-c'],fontsize=fntSze)
    plt.xlim((0,600))
    plt.title('objective function values',fontsize=fntSze)
    plt.xlabel('times (s)',fontsize=fntSze)
    plt.grid()
    figname = 'figs/block_compare_'+prob+'.pdf'
    #plt.savefig(figname,format='pdf')
    plt.show()
    #plt.close()

if False :
    opt = min(np.concatenate([np.array(outcp.f),f_psbg,cache['f_ps2fembed_g']]))

    markFreq = 2000
    markerSz = 10
    print("plotting relative error to optimality of funtion values")
    print("optimal value estimated as lowest returned by any algorithm")
    # only plot out to half the total number of iterations to reduce the distortion
    # caused by the inaccuracy of the estimate for opt.
    #plt.semilogy(out1f.times,(np.array(out1f.fx2)-opt)/opt)
    #plt.semilogy(out2f.times[0:int(len(out2f.times)/2)],(np.array(out2f.fx2[0:int(len(out2f.times)/2)])-opt)/opt,'-o',markevery = markFreq,markersize =markerSz)
    #plt.semilogy(outfrb.times[0:int(len(outfrb.times)/2)], (np.array(outfrb.f[0:int(len(outfrb.times)/2)])-opt)/opt,'D-',markevery = markFreq,markersize =markerSz,color='brown')
    plt.semilogy(outcp.times,(np.array(outcp.f)-opt)/opt,'rs-',markevery = markFreq,markersize =markerSz)
    #plt.semilogy(outtseng.times[0:int(len(outtseng.times)/2)],(np.array(outtseng.f[0:int(len(outtseng.times)/2)])-opt)/opt,'mx-',markevery = markFreq,markersize =markerSz)
    #plt.semilogy(t_ps2fg,(f_ps2fg-opt)/opt)
    plt.semilogy(t_psbg,(f_psbg-opt)/opt)
    #plt.semilogy(cache['t_ps1fembed'],(cache['f_ps1fembed']-opt)/opt)
    #plt.semilogy(cache['t_ps2fembed'],(cache['f_ps2fembed']-opt)/opt)
    plt.semilogy(cache['t_ps2fembed_g'],(cache['f_ps2fembed_g']-opt)/opt)

    fonts = 15
    plt.xlabel('time (s)',fontsize = fonts)
    #plt.legend(['ps1fbt','ps2fbt','frb-pd','cp-bt','Tseng-pd'])
    plt.legend(['cp','psbg','2f_embed_g'])
    plt.title('relative error to optimality of function values')
    plt.grid()
    plt.show()
