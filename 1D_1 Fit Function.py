#Importing necessary modules
import numpy as np
import scipy.integrate
import scipy.special
import scipy.interpolate
import matplotlib.pyplot as plt
from Project_1 import fitfunc1, test_pdf1

#Reading the datafile and extracting the decay times and errors
with open('lifetime-2018.txt', 'r') as f:
    lines = [np.float(element) for element in f.read().split()]
dtimes = lines[::2]
derrors = lines[1::2]

#Array of time values used to plot the theoretical decay function for
#various tau and sigma values
time_arr = np.linspace(-1, 2, 10000)

#Plotting the histogram of decay time measurements and a function which roughly
#fits the distribution; as it is a rough fit, it is simply used to show the
#approximate shape of the expected distribution curve w/o background
fig1, ax1 = plt.subplots(figsize=(10,7))
n1, bins1, patches1 = plt.hist(dtimes, bins=60)
ax1.plot(time_arr, 1000*fitfunc1(t=time_arr, tau=0.4, sigma=0.15), linewidth=3.0, label='Expected decay distribution without background', rasterized=True)
ax1.set_xlabel('Decay Time (ps)', fontsize=20)
ax1.set_ylabel('Frequency', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
plt.legend(loc=1, fontsize=13)
plt.savefig(r"C:\Users\Ivan\Desktop\Year 3\Computational Physics\Project\Graphs_1D\Histogram_1.pdf", fmt="pdf", transparent=True, rasterized=True, bbox_tight=True)

#Plotting the histogram of measured decay time errors
fig2, ax2 = plt.subplots(figsize=(10,7))
n2, bins2, patches2 = plt.hist(derrors, bins=45)
ax2.set_xlabel('Decay Time Error (ps)', fontsize=20)
ax2.set_ylabel('Frequency', fontsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)
plt.savefig(r"C:\Users\Ivan\Desktop\Year 3\Computational Physics\Project\Graphs_1D\Histogram_2.pdf", fmt="pdf", transparent=True, rasterized=True, bbox_tight=True)


#Plotting the decay function for different values of tau and sigma, and
#integrating the function for different parameters. Integral is approximately 1,
#showing that it is independent of tau and sigma as it is a PDF and it should
#integrate to 1 regardless. The value is not exactly 1 as the integral is
#performed over a limited domain; refer to test_pdf1
plt.figure(figsize=(10,7))
tau_graph = np.linspace(0.1, 2, 5)
for i in range(len(tau_graph)):
    plt.plot(time_arr, fitfunc1(t=time_arr, tau=tau_graph[i], sigma=0.05), linewidth=2.0, label='tau = {0:.3f}, I = {1:.12f}'.format(tau_graph[i], test_pdf1(tau_graph[i], 0.05)[0]))
    print('For tau = {0}, sigma = 0.05, integral = {1}'.format(tau_graph[i], (test_pdf1(tau_graph[i], 0.05)[0])))
plt.xlabel('Decay Time (ps)', fontsize=20)
plt.ylabel('Decay function', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(fontsize=13)
#plt.savefig(r"C:\Users\Ivan\Desktop\Year 3\Computational Physics\Project\Graphs_1D\Vary_Tau.pdf", fmt="pdf", transparent=True, rasterized=True, bbox_tight=True)

plt.figure(figsize=(10,7))
sigma_graph = np.linspace(0.02, 0.503, 5)
for i in range(len(sigma_graph)):
    plt.plot(time_arr, fitfunc1(t=time_arr, tau=0.4, sigma=sigma_graph[i]), linewidth=2.0, label='sigma = {0:.3f}, I = {1:.12f}'.format(sigma_graph[i], test_pdf1(0.4, sigma_graph[i])[0]))
    print('For tau = 0.4, sigma = {0}, integral = {1}'.format(tau_graph[i], (test_pdf1(0.4, sigma_graph[i])[0])))
plt.xlabel('Decay Time (ps)', fontsize=20)
plt.ylabel('Decay function', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend(fontsize=13)
#plt.savefig(r"C:\Users\Ivan\Desktop\Year 3\Computational Physics\Project\Graphs_1D\Vary_Sigma.pdf", fmt="pdf", transparent=True, rasterized=True, bbox_tight=True)
