import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def weighted_currents(b, theta, N, f, alpha):
    # signal parameters
    lam = 3e8/f
    k = 2*np.pi/lam

    # array parameters
    d = lam/2
    n = np.arange(1, N+1)

    # calculate Chebyshev-Dolf equation
    u = k*d*np.cos(theta)-alpha
    # Prevent x values from being outside the valid range for arccos
    x = b*np.cos(u/2) 

    T = np.zeros((N, len(theta)), dtype=complex)
    for i in range(N):
        T[i] = np.cos(n[i] * np.arccos(x.astype(complex)))  

    # return the weights
    return T

# Choose parameters
theta = np.linspace(0.01, 2*np.pi, 1000)
N_el = 6
N = N_el - 1
f = 1e9
alpha = 0

# Initial value of b
b_init = 1.1  # Set initial b to a value greater than 1

# Create the figure and axes for the plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plt.subplots_adjust(bottom=0.25)

# Run weighted_currents to get initial weights
weights = weighted_currents(b_init, theta, N, f, alpha)

# Calculate the array factor
AF = np.sum(weights, axis=0)

# Rectangular window plot
line, = ax.plot(theta, 20*np.log10(np.abs(AF)/np.max(np.abs(AF))), label='Array Factor')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Array Factor [dB]')
ax.set_title('Array Factor of a Uniform Linear Array')
ax.set_ylim([-100, 0])
ax.grid()

# Polar plot
fig_polar = plt.figure(figsize=(8, 6))
ax_polar = fig_polar.add_subplot(111, projection='polar')
ax_polar.plot(theta, np.abs(AF)/np.max(np.abs(AF)))
ax_polar.set_title('Array Factor of a Uniform Linear Array')

# Slider axis
ax_slider = plt.axes([0.1, 0.01, 0.8, 0.03], facecolor='lightgoldenrodyellow')

# Slider widget
slider = Slider(ax_slider, 'b', 1.0, 5.0, valinit=b_init, valstep=0.01)

# Update function for the slider
def update(val):
    b = slider.val
    weights = weighted_currents(b, theta, N, f, alpha)
    AF = np.sum(weights, axis=0)
    
    # Update the rectangular plot
    line.set_ydata(20*np.log10(np.abs(AF)/np.max(np.abs(AF))))
    
    # Update the polar plot
    ax_polar.clear()
    ax_polar.plot(theta, np.abs(AF)/np.max(np.abs(AF)))
    ax_polar.set_title('Array Factor of a Uniform Linear Array')
    
    fig.canvas.draw_idle()
    fig_polar.canvas.draw_idle()

# Connect the slider to the update function
slider.on_changed(update)

# Show the plots
plt.show()
