
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import math as m

import sys
sys.path.append('..')
from freddi import Freddi, State, EvolutionResults

DAY = 86400


# In[2]:

#print(Freddi.__doc__)
#print('#############################')
#dir(State)


# In[3]:

from matplotlib import pylab as plt
get_ipython().magic('matplotlib inline')


# In[4]:

_Xi_max = 40
_T_iC = 1e8


rout = 1.7e11
Mx = 2e33*9.4
GM = 6.673e-8 * Mx
Mdotout = 0
Mdotin = 1e19
Cirr = 2.9e-4
alpha = 1.2
kerr =  0.4


default_kwargs = dict(wind=b'no',F0=Mdotin*np.sqrt(GM*rout), Mdotout=Mdotout, Mx=Mx, rout = rout, alpha = alpha, kerr = kerr,
            initialcond=b'quasistat', powerorder=1, opacity=b'OPAL', 
            Cirr = Cirr, time=35*DAY, tau=1*DAY, Nx=10000, gridscale=b'linear')

def run(**input_kwargs):
    kwargs = default_kwargs.copy()
    kwargs.update(input_kwargs)
    fr = Freddi(**kwargs)
    return fr
    

fr = run()
r = fr.evolve()

frw = run(wind=b'__Woods_Which_Shields__', windparams=[_Xi_max, _T_iC], alpha = 0.7, time=35*DAY)
result = frw.evolve()





frh = run(boundcond = b'Tirr', Thot = 1e4)
rh  = frh.evolve()

frwh = run(wind=b'__Woods_Which_Shields__', windparams=[_Xi_max, _T_iC], alpha = 0.7, boundcond = b'Tirr', kerr = kerr, Thot = 1e4)
resulth = frwh.evolve()


# In[5]:

a = np.genfromtxt('asu.tsv', names = True)


# In[6]:

x = (a['tend']/2 + a['tbegin']/2) - (2452443.31221716/2 + 2452443.32503197/2)
y = a['dotM']*1e18
yerr = [a['dotM']-a['b_dotM'],a['B_dotM']- a['dotM']]


# In[7]:

plt.figure(figsize = (10,6))
plt.title(r'Wind off: $\alpha = 1.2$, Wind on:  $\alpha = 0.7$, ; Woods Approx Case', fontsize=16)
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'$t$, days after peak', fontsize=18)
plt.ylabel(r'$\dotM$, g/s', fontsize=22)
#plt.plot(r.t / DAY + 3.5, r.Mdot_in, label='Wind off')
plt.plot(rh.t / DAY + 3.5, rh.Mdot_in, label='Wind off, hot radius ')
plt.errorbar(x, y, yerr, fmt='x', color = 'k', label='Observe')
#plt.plot(result.t / DAY + 3.5, result.Mdot_in, label='Wind on')
plt.plot(resulth.t / DAY + 3.5, resulth.Mdot_in, label='Wind on, hot radius ')
#plt.plot(xise.t / DAY + 3.5, xise.Mdot_in, label='Wind on, hot radius')
plt.axhline(np.exp(-1)*resulth.Mdot_in[0], ls='-.', color='k', lw=0.5, label='$\mathrm{e}^{-1}$')
plt.legend()
plt.grid()


# In[8]:

Mopt = 2e33*2.5
q = Mx/Mopt
period = 1.116*DAY
a = (((period*period)/(4*m.pi*m.pi))* 6.673e-8 *(Mx + Mopt))**(1/3)
rA = 0.8*(0.49*q**(2/3)/(0.6*q**(2/3) + m.log(1+q**(1/3))))*a
plt.figure(figsize = (10,6))
plt.title(r'Wind off: $\alpha = 0.7$, Wind on: $\alpha = 1.2$;  Woods Case', fontsize=16)
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'$t$, days after peak', fontsize=18)
plt.ylabel(r'$R_{hot}/R_{tid}$', fontsize=22)
plt.plot(rh.t / DAY, rh.last_R/rA, label='Wind off, hot radius')
plt.plot(resulth.t / DAY, resulth.last_R/rA, label='Wind on, hot radius')
plt.legend()
plt.grid()


# In[40]:

m_P = 1.6726e-24
k_B = 1.3807e-16
mu = 0.61
Ric = ((GM*mu*m_P)/(k_B*_T_iC))
#plt.xscale('log')
SMTH = -resulth.windC[0,:]*GM*GM/(4*m.pi*(resulth.h[0,:])*(resulth.h[0,:])*(resulth.h[0,:]))
plt.xlabel(r'$R/R_{IC}$,')
plt.ylabel(r'$d\dot{M}/dA$, g/(s*cm$^{2}$)')
plt.plot(resulth.R[-1,:]/Ric, SMTH)
plt.grid()
#plt.savefig('four.pdf')


# In[42]:

C_ch = ((k_B*_T_iC)/(m_P*mu))**(0.5)

c_G = (GM / Ric )**0.5

from scipy import integrate
from math import cos, sin

i = 70 * m.pi / 180

r = resulth.R[0,:]
n = (SMTH / (C_ch * m_P * mu))*(1 - cos(i))

integrate.simps(n, r)


# In[43]:

sigma_Toms = 6.65e-25
Rin = 0.2 * Ric
N_H = ( resulth.Mdot_wind * (1-cos(i)) / (4 * m.pi * Rin * C_ch * m_P * mu) )

TaU = N_H * sigma_Toms 


# In[44]:

plt.xlabel(r'$t$, дней после пика', fontsize=12)
plt.ylabel(r'$\tau$', fontsize=18)
plt.plot(resulth.t / DAY, TaU)
plt.grid()


# In[45]:

#plt.yscale('log')
#plt.xscale('log')
SMWH = -result.windC[-1,:]*GM*GM/(4*m.pi*(result.h[-1,:])*(result.h[-1,:])*(result.h[-1,:]))*2*m.pi*result.R[-1,:]*result.R[-1,:]
plt.xlabel(r'$R/R_{IC}$,')
plt.ylabel(r'$(\partial \Sigma_{wind}/ \partial t)*r^2  $, g/s') 
plt.plot(result.R[-1,:]/1.00244e+11, SMWH)
plt.grid()
#plt.savefig('four.pdf')


# In[46]:

plt.figure()
#plt.title(r'в расчете на $\dotM_{\rm acc}/(M_{\rm x} + M_{\rm opt})$', fontsize=14)
plt.xlabel(r'$\dotM_{\rm wind}/\dotM_{\rm acc}$', fontsize=12)
plt.ylabel(r'$(\dot{P}/P)$', fontsize=18)
Mopt = 1e33
Macc1 = 1e15
t1 = 40*365*DAY
Macc2 = 1e18
t2 = 40*DAY
deltaM = Macc1*t1 + Macc2*t2
Macc = deltaM/(t1 + t2)
q = Mx/Mopt
rA = 0.9*(0.49*q**(2/3)/(0.6*q**(2/3) + m.log(1+q**(1/3))))
k = np.linspace(0, 10, 101)
P = ((Macc/(Mx + Mopt))*(3.0*k*((Mopt + Mx)/Mopt - Mopt/Mx - ((Mopt + Mx)**(3/2))/(Mopt*(Mx)**(0.5))*rA**(1/2) - 1/2) + 3.0*(Mx/Mopt- Mopt/Mx)))
plt.plot(k, P, label='$M_{x}/M_{opt} = 20$ ')
Mopt = 2e33
q = Mx/Mopt
rA = 0.9*(0.49*q**(2/3)/(0.6*q**(2/3) + m.log(1+q**(1/3))))
P = ((Macc/(Mx + Mopt))*(3.0*k*((Mopt + Mx)/Mopt - Mopt/Mx - ((Mopt + Mx)**(3/2))/(Mopt*(Mx)**(0.5))*rA**(1/2) - 1/2) + 3.0*(Mx/Mopt- Mopt/Mx)))
plt.plot(k, P, label='$M_{x}/M_{opt} = 10$')
Mopt = 5e33
q = Mx/Mopt
rA = 0.9*(0.49*q**(2/3)/(0.6*q**(2/3) + m.log(1+q**(1/3))))
P = ((Macc/(Mx + Mopt))*(3.0*k*((Mopt + Mx)/Mopt - Mopt/Mx - ((Mopt + Mx)**(3/2))/(Mopt*(Mx)**(0.5))*rA**(1/2) - 1/2) + 3.0*(Mx/Mopt- Mopt/Mx)))
plt.plot(k, P, label='$M_{x}/M_{opt} = 4$ ')
plt.legend()
plt.grid()


