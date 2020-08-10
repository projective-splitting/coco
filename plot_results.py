import numpy as np
import pickle
from matplotlib import pyplot as plt

lam = 1e-8
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


#plt.plot(outcp.y)
#plt.title('cp output')
#plt.show()
print("================")
print("================")
print("================")
print("classification errors...")


print("================")
print("================")
print("================")
print("plotting raw function values...")


plt.plot(outcp.times,np.array(outcp.f))
plt.plot(t_psbg,f_psbg)

history_2fg = cache['history_2fg']
f_2fg = history_2fg[0]
t_2fg = history_2fg[1]
plt.plot(t_2fg,f_2fg)

plt.plot(cache['outfrb'].times,cache['outfrb'].f)
plt.plot(cache['outtseng'].times,cache['outtseng'].f)

plt.xlabel('times (s)')
plt.title('raw function values')

plt.legend(['cp','psb_g','2fembed_g','frb','tseng']) # MAPR plan
plt.show()

plt.plot(cache['t_ps2fembed'],cache['f_ps2fembed'])
plt.plot(cache['t_ps2fembed_g'],cache['f_ps2fembed_g'])
plt.plot(cache['t_ps2fembed_r'],cache['f_ps2fembed_r'])
plt.plot(cache['t_ps2fembed_c'],cache['f_ps2fembed_c'])
plt.legend(['2f_embed','2f_embed_g','2f_embed_r','2f_embed_c'])
plt.show()
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
