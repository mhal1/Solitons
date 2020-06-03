print("Python version 3.6.4")

# Project 3: Dynamics of Solitons

import math

print("Math version for python 3.6.4")

import numpy as np

print("Numpy version: " + str(np.__version__))

import matplotlib.pyplot as plt

import matplotlib.axes as ax

import matplotlib.animation as animation

from matplotlib.colors import LogNorm

from math import log10, floor

print("matplotlib version for python 3.6.4")

# a) program to propagate the pulse using Runge Kutta 4th order discretisation

# Parameters of the problem

alpha = 9.0

dx = 0.15

dt = dx**3

x = np.arange(0,40,dx)

#conditions for different results-----------------------------------------------------------------

wavebreak = False

plotcmap = False

plotcmapcol = False

plotsoliton = False

animate = False

velplot = False

stabilityplot = False

plotcmapwb = False

shockwave = False

diff = False

plotdiff = False

notdiff = True

#periodic boundary conditions imposed by shifting the list ---------------------------------------

def shift(List, positions):

    return List[positions:] + List[:positions]

#function for the derivative

def f(u,dx):
    
    
    #this is the du/dt
    
    if shockwave == True or notdiff == True:
        
        #shockwave term with no non-linear term

        return -0.25*(1/dx)*((np.array(shift(u,1)))**2-((np.array(shift(u,-1)))**2))

    elif diff == True:

        D = 2.0
    
        return -0.25*(1/dx)*((np.array(shift(u,1)))**2-((np.array(shift(u,-1)))**2)) + D*(1/(dx**2))*(np.array(shift(u,1)) - 2*np.array(u) + np.array(shift(u,-1)))

    else:

        #discretisation for KdeV
        
        return -0.25*(1/dx)*((np.array(shift(u,1)))**2-((np.array(shift(u,-1)))**2)) - 0.5*(1/(dx)**3)*(np.array(shift(u,2)) - 2*np.array(shift(u,1)) + 2*np.array((shift(u,-1))) - np.array((shift(u,-2))))

# functions for evaluating Runge-Kutta

def k1(a,dt,dx):
    
    return dt*f(list(a),dx)

def k2(a,dt,dx):
    
    return dt*f(list(np.array(a) + 0.5*k1(a,dt,dx)),dx)

def k3(a,dt,dx):
    
    return dt*f(list(np.array(a) + 0.5*k2(a,dt,dx)),dx)

def k4(a,dt,dx):
    
    return dt*f(list(np.array(a) + k3(a,dt,dx)),dx)

# this simply plots sin wave with only positive values--------------------------------------------------------------------------

def s():

    a = np.arange(0,80,0.1)

    #sine wave plotted with amplitude much less than period
    
    b = 5*np.sin((a)/20)

    c = list((b > 0) * b)

    #this simply fills list with zeros to match x axis plot in animation

    d = list(np.zeros(100)) + c + list(np.zeros(100))

    return d

def u(x,t,alpha):

    #standard initial wave form centered at x = 20 
    
    return 12*(alpha**2)*(1/math.cosh(alpha*(x - 20 - 4*(alpha**2)*t)))**2

#initial condition calculated with the analytic solution

uvalues1 = np.array([u(i,0,alpha) for i in x])

#save the initial condition to be used to reset the check for stability

uo = uvalues1
    
if wavebreak == False:
    
    uvalues = uvalues1
    
t = []

tt = 0

n = 100

dxl = np.logspace(-10, 0.2, n)

dtl = np.logspace(-10, 0.2, n)

def sta(dx,dt,uvalues):
    
    global uo
    
    if max(uvalues) < 0.99*max(uo) or max(uvalues) > 1.01*max(uo) or np.isnan(max(uvalues)):
        
        return 0
    
    else:
        
        return 1

def round_sig(x, sig=2):

    if x == 0:
        
        return 0
    
    else:
        
        return round(x, sig-int(floor(log10(abs(x))))-1)

if stabilityplot == True:
    
    steps = 0
    
    smap = []
    
    stability = []
    
    counter = 0
    
    for k in dxl:
        
        for l in dtl:
            
            counter += 1
            
            print(counter)
            
            while steps <= 100:
                
                c = uvalues #list
                
                u = np.array(c) + (1/6)*(k1(c,l,k) + 2*(k2(c,l,k) + k3(c,l,k)) + k4(c,l,k)) #array
                
                # Store the new solution
                
                uvalues = list(u)

                outcome = sta(k,l, uvalues)

                if outcome == 0:

                    #exit loop

                    steps = 101
                
                steps += 1

            #reset intial conditions

            uvalues = uo

            stability.append(outcome)
                
            steps = 0
            
        smap.append(stability)

        stability = []

    #plot heat map

    fig1, ax1 = plt.subplots()

    locs, labels = plt.xticks()

    locs, labels = plt.yticks()

    plt.xticks([round_sig(x) for x in np.linspace(0,n,n/10)], [round_sig(x) for x in np.logspace(-10, 0.2, n/10)])

    plt.yticks([round_sig(x) for x in np.linspace(0,n,n/10)], [round_sig(x) for x in np.logspace(-10, 0.2, n/10)])

    aa = ax1.imshow(smap, origin='lower', cmap='rainbow')
    
    cbar = fig1.colorbar(aa, ax=ax1, extend='both')
    
    ax1.set_title("Stability for alpha = " + str(alpha))
    
    ax1.set_xlabel("dt")
    
    ax1.set_ylabel("dx")
    
    plt.show()

# Function needed to initiate the animation --------------------------------------------------------------------------

def init():
    
    line.set_ydata(uvalues)
    
    return line,

# Calculation for animation --------------------------------------------------------------------------

def update(i):
    
    global unext, uvalues, dx, dt

    c = uvalues #list

    unext = np.array(c) + (1/6)*(k1(c,dt,dx) + 2*(k2(c,dt,dx) + k3(c,dt,dx)) + k4(c,dt,dx)) #array
    
    # Update the plot
    
    line.set_ydata(list(unext))  # update the data
    
    text.set_text(r't = {:3}'.format(i*dt))
    
    plt.text(list(uvalues).index(max(uvalues)), max(uvalues), r't = {}'.format(0.0))
    
    # Store the new solution
    
    uvalues = list(unext)
    
    return line, text

#running the animation ----------------------------------------------------------------------------

if animate == True:
    
    # Preparing the plots
    
    fig, ax = plt.subplots()
    
    line_ini = ax.plot(x, uvalues, 'r')
    
    line, = ax.plot(x, uvalues)
    
    plt.title("Soliton propagation")
    
    plt.text(max(x) - 20, max(uvalues), r'alpha = {:3.2}'.format(alpha))
    
    text = plt.text(10, max(uvalues), r't = {}'.format(0.0))
    
    plt.xlabel('x (m)')
    
    plt.ylabel('u')
    
    # Start the animation (and therefore the calculation)
    
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0,100000000,1), init_func=init, blit=False, interval=1)

    plt.show()

#for velocity calc -------------------------------------------------------------------------------

def velocityplot():

    global dt, dx, x

    v = []

    h = []

    #list of alphas for velocity plot

    alpha = np.linspace(0.1,3.0,12)

    #set time want to run each soliton for

    t = 0.06

    #counter for time in propagation

    j = 1

    for l in alpha:

        uvalues = np.array([u(i,0,l) for i in x])

        #initial value of peak

        xo = list(uvalues).index(max(uvalues))

        while j*dt <= t:

            c = uvalues #list

            unext = np.array(c) + (1/6)*(k1(c,dt,dx) + 2*(k2(c,dt,dx) + k3(c,dt,dx)) + k4(c,dt,dx)) #array

            uvalues = list(unext)

            j += 1

        #now append velocity and alpha (height 12alpa^2)

        v.append(dx*(list(uvalues).index(max(uvalues)) - xo)/(j*dt))

        j = 1

        h.append(12*(l**(2)))

    x2 = np.array(h)

    y2 = np.array(v)

    A = np.vstack([x2, np.ones(len(x2))]).T
    
    m, c = np.linalg.lstsq(A, y2, rcond=None)[0]

    # Polynomial Regression
    
    def polyfit(x, y, degree):
        
        results = {}

        coeffs = np.polyfit(x, y, degree)

        # Polynomial Coefficients
         
        results['polynomial'] = coeffs.tolist()

        # r-squared
        
        p = np.poly1d(coeffs)
        
        # fit values, and mean
        
        yhat = p(x) 
        
        ybar = np.sum(y)/len(y)
        
        ssreg = np.sum((yhat-ybar)**2)
        
        sstot = np.sum((y - ybar)**2)
        
        results['determination'] = ssreg / sstot

        return results

    r_value = polyfit(x2,y2,1)['determination']

    plt.figure()
    
    plt.scatter(x2,y2, s =6)
    
    plt.plot(x2, m*x2 + c, 'r', label='Fitted line,' + '\n' +  'm = ' + str(m) + '\n' + 'r = ' +str(r_value) + '\n' + 'intercept = ' + str(c))

    plt.legend()

    plt.grid(True)

    plt.title(" Velocity vs Height ")

    plt.xlabel("Height (m)")

    plt.ylabel("Velocity (m/s)")

    plt.xlim(0, max(x2))

    plt.ylim(0,max(y2))

    plt.show()

if velplot == True:
    
    velocityplot()

#plotting soliton propagation --------------------------------------------------------------------------

def plotsol(alpha,t,f):

    global x, dt, dx

    j = 0

    uvalues = np.array([u(i,0,alpha) for i in x])

    plt.figure()

    plt.plot(x,uvalues)

    plt.title("Soliton propagation with alpha = " + str(alpha))
    
    plt.xlabel("x (m)")
    
    plt.ylabel("u")

    plt.text(list(uvalues).index(max(uvalues))*dx , max(uvalues), r't = {:3.2}'.format(j*dt))

    i = 1
    
    while j*dt <= t:

        c = uvalues #list

        unext = np.array(c) + (1/6)*(k1(c,dt,dx) + 2*(k2(c,dt,dx) + k3(c,dt,dx)) + k4(c,dt,dx)) #array

        uvalues = list(unext)

        tt = j*dt

        if tt > i*f:
        
            plt.plot(x,unext)
            
            plt.text(list(unext).index(max(unext))*dx , max(unext), r't = {:3.2}'.format(j*dt))
            
            i += 1

        j += 1

    plt.show()

#alpha = 2, run with (2.0,2.1,0.5)

#alpha = 3, run with (3.0,1,0.3)

if plotsoliton == True:
    
    plotsol(3.0,1,0.3)

#heat map ----------------------------------------------------------------------------------------

def umap(alpha,t,x):
    
    global dt, dx

    j = 0

    umap = []

    tg = []

    uvalues = np.array([u(i,0,alpha) for i in x])

    while j*dt <= t:

        c = uvalues #list

        unext = np.array(c) + (1/6)*(k1(c,dt,dx) + 2*(k2(c,dt,dx) + k3(c,dt,dx)) + k4(c,dt,dx)) #array

        uvalues = list(unext)

        umap.append(uvalues)

        tg.append(j*dt)

        j += 1

    fig1, ax1 = plt.subplots()
    
    aa = ax1.imshow(umap, extent = [0,len(x)*dx,0,len(tg)*dt], origin='lower', cmap='rainbow')
    
    cbar = fig1.colorbar(aa, ax=ax1, extend='both')

    if plotdiff == True:

        ax1.set_title("Soliton diffusion shockwave")

    elif notdiff == True:

        ax1.set_title("Soliton shockwave")

    else:

        ax1.set_title("Soliton colour map with alpa = " +str(alpha))
    
    ax1.set_xlabel("x (m)")
    
    ax1.set_ylabel("t (s)")
    
    plt.show()

if plotcmap == True:
    
    umap(1.0,30,np.arange(0,30,dx))

#heat map for collisions--------------------------------------------------------------------------

def umapcol(alpha, alpha2, t):
    
    global dt, dx

    x = np.arange(0,30,dx)

    j = 0

    umap = []

    tg = []

    uvalues1 = np.array([u(i,0,alpha) for i in x])

    #add another soliton
    
    uvalues2 = np.array([u(i-5,0,alpha2) for i in x])

    uvalues = list(uvalues1 + uvalues2)
    
    while j*dt <= t:

        c = uvalues #list

        unext = np.array(c) + (1/6)*(k1(c,dt,dx) + 2*(k2(c,dt,dx) + k3(c,dt,dx)) + k4(c,dt,dx)) #array

        uvalues = list(unext)

        umap.append(uvalues)

        tg.append(j*dt)

        j += 1

    fig1, ax1 = plt.subplots()
    
    aa = ax1.imshow(umap, extent = [0,len(x)*dx,0,len(tg)*dt], origin='lower', cmap='rainbow')
    
    cbar = fig1.colorbar(aa, ax=ax1, extend='both')
    
    ax1.set_title("Soliton colour map with alpa = " +str(alpha) + ", " + str(alpha2))
    
    ax1.set_xlabel("x (m)")
    
    ax1.set_ylabel("t (s)")
    
    plt.show()

if plotcmapcol == True:
    
    umapcol(1.0,1.2,30)
    
#wavebreaking will be checked with sine wave ------------------------------------------------------

def wavebreakplot(t,f):
    
    global x, dt, dx

    j = 0

    ppos = [50,90]

    uvalues = s()

    x = np.arange(0,100,0.1)

    plt.figure()

    plt.plot(x,uvalues)

    plt.title("Wave breaking with a sine wave")
    
    plt.xlabel("x (m)")
    
    plt.ylabel("u")

    plt.text(list(uvalues).index(max(uvalues))*dx - 200*dx, max(uvalues), r't = {:3.2}'.format(j*dt))

    i = 1

    k = 0
    
    while j*dt <= t:

        c = uvalues #list

        unext = np.array(c) + (1/6)*(k1(c,dt,dx) + 2*(k2(c,dt,dx) + k3(c,dt,dx)) + k4(c,dt,dx)) #array

        uvalues = list(unext)

        tt = j*dt

        if tt > i*f:
        
            plt.plot(x,unext)
            
            plt.text(ppos[k], max(unext), r't = {:3.2}'.format(j*dt))

            k += 1
            
            i += 8

        j += 1

    plt.show()

if wavebreak == True:
    
    wavebreakplot(20,2)

#heat map ----------------------------------------------------------------------------------------

def umapwb(t):
    
    global dt, dx

    x = np.arange(0,100,dx)

    j = 0

    umap = []

    tg = []

    uvalues = s()

    while j*dt <= t:

        c = uvalues #list

        unext = np.array(c) + (1/6)*(k1(c,dt,dx) + 2*(k2(c,dt,dx) + k3(c,dt,dx)) + k4(c,dt,dx)) #array

        uvalues = list(unext)

        umap.append(uvalues)

        tg.append(j*dt)

        j += 1

    fig1, ax1 = plt.subplots()
    
    aa = ax1.imshow(umap, extent = [0,len(x)*dx,0,len(tg)*dt], origin='lower', cmap='rainbow')
    
    cbar = fig1.colorbar(aa, ax=ax1, extend='both')
    
    ax1.set_title("Wave breaking colour map")
    
    ax1.set_xlabel("x (m)")
    
    ax1.set_ylabel("t (s)")
    
    plt.show()

if plotcmapwb == True:

    umapwb(60)

#shock wave with soliton ------------------------------------------------------------------------

def shockwaveplot(t,f):
    
    global x, dt, dx

    j = 0

    x = np.arange(0, 100, dx)

    uvalues = np.array([u(i,0,1.0) for i in x])

    plt.figure()

    plt.plot(x,uvalues)

    plt.title("Shock wave with a soliton")
    
    plt.xlabel("x (m)")
    
    plt.ylabel("u")

    i = 1
    
    while j*dt <= t:

        c = uvalues #list

        unext = np.array(c) + (1/6)*(k1(c,dt,dx) + 2*(k2(c,dt,dx) + k3(c,dt,dx)) + k4(c,dt,dx)) #array

        uvalues = list(unext)

        tt = j*dt

        if tt > i*f:
        
            plt.plot(x,unext)
            
            i += 19

        j += 1

    plt.show()

if shockwave == True:
    
    shockwaveplot(50,2)


#make this shock wave with diffusion ------------------------------------------------------------------------

#animate the diffusive term

def shockwaveplotdiff(t,f):
    
    global x, dt, dx

    j = 0

    x = np.arange(0, 100, dx)

    uvalues = np.array([u(i,0,1.0) for i in x])

    plt.figure()

    plt.plot(x,uvalues)

    plt.title("Shock wave with a soliton")
    
    plt.xlabel("x (m)")
    
    plt.ylabel("u")

    i = 1
    
    while j*dt <= t:

        c = uvalues #list

        unext = np.array(c) + (1/6)*(k1(c,dt,dx) + 2*(k2(c,dt,dx) + k3(c,dt,dx)) + k4(c,dt,dx)) #array

        uvalues = list(unext)

        tt = j*dt

        if tt > i*f:
        
            plt.plot(x,unext)
            
            i += 40

        j += 1

    plt.show()

if diff == True and plotdiff == False:
    
    shockwaveplotdiff(40,0.2)

if diff == True and plotdiff == True:

    umap(1.0,60,np.arange(0,60,dx))

if notdiff == True:

    umap(1.0,40,np.arange(0,60,dx))

    

    
    



    

    


    

    
