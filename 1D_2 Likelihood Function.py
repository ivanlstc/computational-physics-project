#Importing necessary modules
import numpy as np
import scipy.integrate
import scipy.special
import scipy.interpolate
import matplotlib.pyplot as plt
from Project_1 import logged_fitfunc1, NLL1

#Reading the datafile and extracting the decay times and errors
with open('lifetime-2018.txt', 'r') as f:
    lines = [np.float(element) for element in f.read().split()]
dtimes = lines[::2]
derrors = lines[1::2]

#Array of tau values to plot 1D NLL function
tau_arr0 = np.linspace(0.1, 2, 50)

#Plotting NLL function for tau ranging from 0.1 to 2; it can be seen from the
#graph that the minimum lies somewhere between 0.3 to 0.45, which will be the
#range used for the minimiser in the next section
plt.figure(figsize=(11,7))
plt.plot(tau_arr0, NLL1(tau_arr0), linewidth=3.0)
plt.xlim([-0.1, 2.1])
plt.xlabel('tau (ps)', fontsize=20)
plt.ylabel('Negative Log Likelihood (NLL)', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
#plt.savefig(r"C:\Users\Ivan\Desktop\Year 3\Computational Physics\Project\Graphs_1D\NLL_1D.pdf", fmt="pdf", transparent=True, rasterized=True, bbox_tight=True)

