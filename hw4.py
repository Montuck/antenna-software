import numpy as np

# given
l = 4e-2 #antenna length
v = 1 #source voltage
R_g = 50 #source impedance
r = 500 #distance
R_l = 50 #load impedance
f = 2e9 #signal frequency

# constants
eta = 377 #impedance of air
mu = (np.pi)*4e-7 #permeability

# adjust the frequency
omega = 2*np.pi*f #sig in radians

# calculate the wave vector
lam = 3e8/f #wavelength
k = 2*np.pi / lam #wave vector

# input impedance
R_rad = eta * (k * l)**2 / (6*np.pi) #radiation resistance
print("Radiation Resistance: ", R_rad) #debug statement
R_loss = 0  #loss resistance
X = 0 #reactance
Z_in = R_rad + R_loss + 1j*X #input impedance
print("Impedance: ", Z_in) #debug statement

# calculate the current
I = v/(Z_in+R_g) #input current
print("Current:", I) #answer statement

# calculate the electric field
E_inc = -1j*omega*mu*np.exp(-1j*k*r)*l*I / (4*np.pi*r) #incident electric field
#E_inc = eta*I*l/(2*np.pi*r*lam)
#E_inc = Z_in * I / l
print("Electric Field:", E_inc) #answer

# calculate the open circuit voltage
V_oc = E_inc * l #open circuit voltage
print("Open circuit Voltage:", V_oc) #answer statement

# calculate the power
P = np.real(V_oc**2 / (2 * (R_l+Z_in))) #calculated power
print("Circuit Power:", P) #answer statement

print("Power (dBm): ", 10*np.log10(P*10**3)) #answer in dB

# Friis transmission formula
S = abs(E_inc)**2 / (2*eta) #power density
Gt = 1.5 #gain of the hertzian dipole
Gr = Gt #gain at receiver
Pt = S / (Gt / (4*np.pi*r**2)) #transmitted power
FPL = (lam / (4*np.pi*r))**2 #freespace path loss
Pr = 10*np.log10(Pt*10**3) + Gt + Gr + 10*np.log10(FPL) #receive power
print("Friis receive power (dBm): ", Pr) #answer statement