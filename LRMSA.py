########### IMPORTS #############
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

########################################### VARIABLES ############################################
# Define antenna parameters
f = 300e6                 # Frequency in Hz
c = 3e8                 # Speed of light in m/s
lam = c / f             # Wavelength in m
k = 2 * np.pi / lam     # Wave number
eta = 377               # Intrinsic impedance of free space
Pel = 1                 # Radiated power
r = 1                   # Observation distance
l = lam / 2             # Half-wave dipole length
Im = 1                  # Mutual impedance
theta = np.pi/2        # broadside or endfire
phi = np.pi/2           # parallel to the array axis

# Define element spacing range
num_points = 100
d_values = np.linspace(0.5*lam, 3*lam, num_points)  # Range of element spacings
#d_values = np.array([lam/2])
# Number of elements in the array
N = 4

# empty array of directivity values
directivity_values = []
directivity_values_cm = []

################################################## MAIN ##########################################################
for d in d_values:
    ## Define the receiver positions ##
    rx = np.array([n * d for n in range(N)])    # Positions of elements
    #print("rx \n", rx)                          # debug receiver positions

    ## compute the vector of phase shifted electric fields ##
    E0 = 1j * Im * eta / (2 * np.pi * r)                                          # Amplitude of the E-Field of a dipole on page 84
    E_el = E0 * (np.cos(k*l*np.cos(theta)/2)-np.cos(k*l/2)) / np.sin(theta) # E-field of a dipole
    Ep = np.zeros(N, dtype=complex)                                    # Initialize electric field
    # Ep loop for broadside
    for n in range(N):
        Ep[n] = E_el*np.exp(1j*k*d*np.sin(phi)*np.cos(theta)*rx[n])         # Compute electric field
    #print("Ep \n", Ep)                                                     # debug e-field

    ## compute the overlap matrix ##
    A = np.zeros((N, N))                                          # initialize the overlap matrix
    points = 50
    for i in range(N):
        for j in range(N):
            # numerical implementation of the quadrature rule according to equations 4.69 and 4.70
            I = 0                                                 # initialize the integral
            del_phi = 2*np.pi/points                                   # integration weights
            del_theta = del_phi                                   # integration weights
            for m in range(points):
                phi_m = (m - 1/2) * del_phi                       # midpoint for each integration step
                for n in range(int(points/2)):
                    theta_n = (n - 1/2) * del_theta               # midpoint for each integration step

                    # integrand
                    Em = E0 * (np.cos(k*l*np.cos(theta_n)/2)-np.cos(k*l/2)) / np.sin(theta_n) * np.exp(1j*k*d*np.sin(theta_n)*np.cos(phi_m)*rx[i])
                    En = E0 * (np.cos(k*l*np.cos(theta_n)/2)-np.cos(k*l/2)) / np.sin(theta_n) * np.exp(1j*k*d*np.sin(theta_n)*np.cos(phi_m)*rx[j])

                    I += Em * np.conj(En) * np.sin(theta_n) * del_phi * del_theta   # integral approximation
                    #print("I\n", I)                               # debug integral                     
            #print("I \n", I)                                      # debug integral                                 

            A[i, j] = I                                           # assign the value of the integral to the overlap matrix
    
    ## compute the mutual impedance matrix ##
    A = A / (2*eta)
    scalar = 2/np.abs(Im)**2    # Equation 4.108
    Za = scalar * A             # mutual impedance matrix
    #print(Za)
          
    ## Compute weights ##
    A_inv = np.linalg.pinv(Za)                  # invert overlap       
    w = np.dot(A_inv, Ep)                       # Compute weights equation 4.86
    #print("weights \n", w)                     # debug weights
    w_cm = Ep                                   # Conjugate matched weights

    ## compute signal response ##
    B = Ep*Ep.conj().T / (2*eta)    # Compute signal response equation 4.66
    
    ## Compute Directivity using equation 4.63 ##
    w_herm = w.conj().T                         # Compute hermitian of weights
    #print("hermitian of weights \n", w_herm)   # debug hermitian of weights
    scalar = 4 * np.pi * (r**2) / Pel           # Compute directivity scalar
    wB = np.dot(w_herm, B)
    numerator = np.dot(wB, w)
    wA = np.dot(w_herm, Za)
    denominator = np.dot(wA, w)
    D = 6*scalar * numerator / denominator        # Compute directivity equation 4.63
    #print("Directivity \n", D)                 # debug directivity

    ## Compute Conjugate Matched Directivity ##
    w_herm = w_cm.conj().T
    #print("hermitian of weights \n", w_herm)
    wB = np.dot(w_herm, B)
    numerator = np.dot(wB, w_cm)
    wA = np.dot(w_herm, Za)
    denominator = np.dot(wA, w_cm)
    D_cm = 6*scalar * numerator / denominator         # Compute directivity equation 4.63
    #print("Directivity \n", D_cm)                  # debug directivity

    # append zeros
    directivity_values.append(D[0])
    directivity_values_cm.append(D_cm[0])
    #print(k*d)

# equation 4.90 used as a target directivity
Del = 1.65              # directivity of a half-wave dipole
d_theoretical = N * Del # theoretical directivity of the array


####################################################### PLOTS ###########################################################
plt.figure(figsize=(8, 5))
plt.plot(d_values / lam, directivity_values, label="Not Conjugate Matched", linewidth=1)
plt.plot(d_values / lam, directivity_values_cm, label="Conjugate Matched", linewidth=1)
plt.axhline(y=d_theoretical, color='r', linestyle='--', label="Theoretical Directivity at d=λ/2")  # Reference line
plt.xlabel("Element Spacing (d/λ)")
plt.ylabel("Directivity")
plt.title("Directivity vs Element Spacing for 4-Element Isotropic ULA")
plt.grid(True)
plt.legend()
plt.show() 