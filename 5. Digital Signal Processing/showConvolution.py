import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['text.usetex'] = True

def showConvolution(t0, f1, f2):
    # Calculate the overall convolution result using Simpson integration
    convolution = np.zeros(len(t))
    for n, t_ in enumerate(t):
        prod = lambda tau: f1(tau) * f2(t_-tau)
        convolution[n] = scipy.integrate.simps(prod(t), t)

    # Create the shifted and flipped function
    f_shift = lambda t: f2(t0-t)
    prod = lambda tau: f1(tau) * f2(t0-tau)

    # Plot the curves
    plt.gcf().clear() 

    plt.subplot(3, 1, 1)
    plt.gca().set_ymargin(0.05)
    plt.plot(t, f1(t), label=r'$f_1(t)$')
    plt.plot(t, f2(t), label=r'$f_2(t)$')
    plt.grid(True)
    plt.legend(fontsize=10)
    
    plt.subplot(3, 1, 2)
    plt.gca().set_ymargin(0.05)
    plt.plot(t, f1(t), label=r'$f_1(\tau)$')
    plt.plot(t, f_shift(t), label=r'$f_2(t_0-\tau)$')
    plt.fill(t, prod(t), color='r', alpha=0.5, edgecolor='black', hatch='//') 
    plt.plot(t, prod(t), 'r-', label=r'$f_1(\tau)f_2(t_0-\tau)$')
    plt.grid(True)
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$x(\tau)$') 
    plt.legend(fontsize=10) 
    plt.text(-4, 0.6, '$t_0=%.2f$' % t0, bbox=dict(fc='white')) 

    # plot the convolution curve
    plt.subplot(3, 1, 3)
    plt.gca().set_ymargin(0.05) 
    plt.plot(t, convolution, label='$(f_1*f_2)(t)$')

    # recalculate the value of the convolution integral at the current time-shift t0
    current_value = scipy.integrate.simps(prod(t), t)
    plt.plot(t0, current_value, 'ro')  # plot the point
    plt.grid(True) 
    plt.xlabel('$t$')
    plt.ylabel('$(f_1*f_2)(t)$') 
    plt.legend(fontsize=10)
    plt.show()

Fs = 50  # our sampling frequency for the plotting
T = 5    # the time range we are interested in
t = np.arange(-T, T, 1/Fs)  # the time samples
f1 = lambda t: np.maximum(0, 1-abs(t))
f2 = lambda t: (t>0) * np.exp(-2*t)

t0 = np.arange(-2.0,2.0, 0.05)

fig = plt.figure(figsize=(8,3))
anim = animation.FuncAnimation(fig, showConvolution, frames=t0, fargs=(f1,f2),interval=80)

#anim.save('animation.mp4', fps=30) # fps = frames per second

plt.show()