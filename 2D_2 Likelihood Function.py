#Importing necessary modules
import numpy as np
import scipy.integrate
import scipy.special
import scipy.interpolate
import matplotlib.pyplot as plt
from Project_1 import fitfunc2, NLL2

#Reading the datafile and extracting the decay times and errors
with open('lifetime-2018.txt', 'r') as f:
    lines = [np.float(element) for element in f.read().split()]
dtimes = lines[::2]
derrors = lines[1::2]


#Range of tau and a values to plot 2D NLL function
tau_arr1 = np.linspace(0.1, 0.9, 10)
a_arr1 = np.linspace(0.001, 1, 10)

#Plotting the 2D NLL function for tau ranging from 0.1 to 0.9, and a ranging
#from 0.001 to 1. It can be seen from the graph that the minimum lies in the
#range where tau is between about 0.40 and 0.42, and a is between 0.97 and 0.99

print('Generating 2D NLL function...')
#Note: 2D NLL plot is interpolated for a smoother plot, not 100% indicative of minimum
#but it gives a good enough estimate for the naked eye
data = NLL2(tau_arr1, a_arr1)
X, Y = np.meshgrid(tau_arr1, a_arr1)
f = scipy.interpolate.interp2d(X, Y, data, kind='cubic') 
xnew = np.linspace(0.1, 0.9, 100)
ynew = np.linspace(0.001, 1, 100)
data1 = f(xnew,ynew)
Xn, Yn = np.meshgrid(xnew, ynew)
plt.figure(figsize=(15,8))
plt.pcolormesh(Xn, Yn, data1, cmap='RdBu', rasterized=True)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=12) 
plt.xlabel('tau (ps)', fontsize=20)
plt.ylabel('a', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Negative Log Likelihood (NLL)', fontsize=20)
#plt.savefig(r"C:\Users\Ivan\Desktop\Year 3\Computational Physics\Project\Graphs_2D\NLL_2D.pdf", fmt="pdf", transparent=True, rasterized=True, bbox_tight=True)
print('Done')