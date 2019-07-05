#Importing necessary modules
import numpy as np
import scipy.integrate
import scipy.special
import scipy.interpolate
import matplotlib.pyplot as plt
from Project_1 import fitfunc2, test_pdf2

#Reading the datafile and extracting the decay times and errors
with open('lifetime-2018.txt', 'r') as f:
    lines = [np.float(element) for element in f.read().split()]
dtimes = lines[::2]
derrors = lines[1::2]

#Array of time values used to plot the theoretical decay function for
#various a values in the 2D case
time_arr = np.linspace(-1, 2, 10000)

#Plotting the decay function for different values of a, and integrating the
#function each time to ensure that the integral goes to 1 (like in the case
#without background) as the function is a pdf and should go to 1. Again, the
#integral doesn't go to EXACTLY 1 but quite close to 1. Refer to test_pdf2
plt.figure(figsize=(10,7))
a_graph = np.linspace(0, 1, 5)
for i in range(len(a_graph)):
    plt.plot(time_arr, fitfunc2(t=time_arr, tau=0.4, sigma=0.05, a=a_graph[i]), linewidth=2.0, label='a = {0:.3f}, I = {1:.12f}'.format(a_graph[i], test_pdf2(0.4, 0.05, a_graph[i])[0]))
    print('For tau = 0.4, sigma = 0.05, and a = {0}, integral = {1}'.format(a_graph[i], (test_pdf2(0.4, 0.05, a_graph[i])[0])))
plt.xlabel('Decay Time (ps)', fontsize=20)
plt.ylabel('Decay function', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(fontsize=13)
#plt.savefig(r"C:\Users\Ivan\Desktop\Year 3\Computational Physics\Project\Graphs_2D\Vary_A.pdf", fmt="pdf", transparent=True, rasterized=True, bbox_tight=True)
