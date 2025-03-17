#FOLLOWING THE STEPS OUTLINE IN SECTION 4.14.1 APPLYING THE LOSSLESS RESONANT MINIMUM SCATTERING APPROXIMATION
#EACH ELEMENT IS A HALF-WAVE DIPOLE

########### IMPORTS #############
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi



########################################### VARIABLES ############################################
# Define antenna parameters
f = 1e9                 # Frequency in Hz
c = 3e8                 # Speed of light in m/s
lam = c / f             # Wavelength in m
k = 2 * np.pi / lam     # Wave number
eta = 377               # Intrinsic impedance of free space
Prad = 1                # Radiated power
r = 1                   # Observation distance
l = lam / 2             # Half-wave dipole length
Im = 1                  # Mutual impedance
theta = np.pi/2          # broadside
#theta = 0               # endfire

# Define element spacing range
num_points = 100
d_values = np.linspace(0, 3*lam, num_points)  # Range of element spacings

# Number of elements in the array
N = 4

# empty array of directivity values
directivity_values = []
directivity_values_cm = []




################################################## MAIN ##########################################################
for d in d_values:
    ## Define the receiver positions ##
    rx = np.array([n * d for n in range(N)])    # Positions of elements

    ## compute Ep ##
    E0 = np.sqrt(2*eta*Prad/(4*np.pi*r**2))
    E_el = E0 * np.exp(1j*k*r)  # E-field of an isotropic radiator
    Ep = np.zeros((N, 1), dtype=complex)                        # Initialize electric field
    # Ep loop for broadside
    for n in range(N):
        Ep[n] = E_el*np.exp(1j*k*d*np.cos(theta)*rx[n])         # Compute phase shifted electric field vector
    #print("Ep \n", Ep)                                         # debug e-field

    ## compute the overlap matrix ##
    A = np.zeros((N, N))                # initialize the overlap matrix
    for m in range(N):
        for n in range(N):
            r_mn = rx[m] - rx[n]        # distance between elements
            A[m, n] = np.sinc(k*r_mn)   # assign the value of the integral to the overlap matrix
    #print("Overlap Matrix \n", A)      # debug overlap matrix           
    
    ## compute mutual impedance matrix ##
    # scalar = 2/np.abs(Im)**2    # Equation 4.108
    # Za = scalar * A             # mutual impedance matrix
    #print("Mutual Impedance Matrix \n", Za) # debug mutual impedance matrix
    
    ## Compute weights ##
    A_inv = np.linalg.pinv(A)                   # invert overlap
    w = np.dot(A_inv, Ep)                       # Compute weights equation 4.86
    #print("weights \n", w)                     # debug weights
    w_cm = Ep                                   # Conjugate matched weights

    ## compute signal response ##
    B = np.zeros((N, N), dtype=complex)     # Initialize signal response
    for m in range(N):
        for n in range(N):
            B[m, n] = Ep[m] * Ep[n].conj()  # Compute signal response equation 4.66
    B = B / (2 * eta)                       # Normalize signal response
    print("B \n", B)                        # debug signal response
    
    ## Compute Directivity using equation 4.63 ##
    w_herm = w.conj().T                         # Compute hermitian of weights
    #print("hermitian of weights \n", w_herm)   # debug hermitian of weights
    scalar = 4 * np.pi * (r**2) / Prad           # Compute directivity scalar
    wB = np.dot(w_herm, B)
    numerator = np.dot(wB, w)
    wA = np.dot(w_herm, A)
    denominator = np.dot(wA, w)
    D = scalar * numerator / denominator        # Compute directivity equation 4.63
    # D = D / (4*np.pi*r**2)                      # Normalize directivity
    #print("Directivity \n", D)                 # debug directivity

    ## Compute Conjugate Matched Directivity ##
    w_herm = w_cm.conj().T
    #print("hermitian of weights \n", w_herm)
    wB = np.dot(w_herm, B)
    numerator = np.dot(wB, w_cm)
    wA = np.dot(w_herm, A)
    denominator = np.dot(wA, w_cm)
    D_cm = scalar * numerator / denominator         # Compute directivity equation 4.63
    # D_cm = D_cm / (4*np.pi*r**2)                    # Normalize directivity
    #print("Directivity \n", D_cm)                  # debug directivity

    # append zeros
    directivity_values.append(D[0,0])
    directivity_values_cm.append(D_cm[0,0])

# equation 4.90 used as a target directivity
Del = 1              # directivity of a half-wave dipole
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