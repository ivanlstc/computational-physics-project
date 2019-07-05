#Importing necessary modules
import numpy as np
import scipy.integrate
import scipy.special
import scipy.interpolate
import matplotlib.pyplot as plt
from Project_1 import fitfunc2, NLL2, find_grad, test_func2d, find_min_test2d, find_min_2d

#Reading the datafile and extracting the decay times and errors
with open('lifetime-2018.txt', 'r') as f:
    lines = [np.float(element) for element in f.read().split()]
dtimes = lines[::2]
derrors = lines[1::2]


#Range of tau and a values to plot zoomed-in 2D NLL function, used for
#the 2D minimisation task
tau_arr2 = np.linspace(0.405, 0.415, 20) #Making the values more precise for faster computation
a_arr2 = np.linspace(0.975, 0.9875, 20)

#Parameters used in find_min_test2d function to test the 2D minimiser
pos_init_test = [-1.5, 0.5]
accuracy_test = 0.0000000001
alpha_test = 0.01
h_test = 0.0000001

#Parameters used in find_min_2d function to minimse the 2D NLL function
pos_init = [0.41, 0.98]
accuracy = 0.0000001
alpha = 0.00001
h = 0.000001


#Testing the 2D minimiser on f = x**2 + 2y**2 + xy + 3x; refer to Chapter 7
#lecture notes. Minimum is when (x,y) = (-12/7, 3/7), which is approximately
#what the 2D minimiser outputs
test = find_min_test2d(pos_init_test, accuracy_test, alpha_test, h_test)
test_xmin = test[0][-1]
test_ymin = test[1][-1]
test_steps = test[2]
print('Minimum of f(x,y) = x**2 + 2y**2 + xy + 3x found at (x,y) = ({0}, {1}) after {2} steps'.format(test_xmin, test_ymin, test_steps))
print('For comparision, (-12/7, 3/7) = ({0}, {1})'.format(-12/7, 3/7))


#Plotting the 2D NLL function for tau ranging from 0.405 to 0.415, and a ranging
#from 0.9750 to 0.9875, which is the range of values the 2D minimiser used to
#find the tau and a values which yield the minimum NLL

print('Generating 2D NLL function...')
data = NLL2(tau_arr2, a_arr2)
X, Y = np.meshgrid(tau_arr2, a_arr2)
f = scipy.interpolate.interp2d(X, Y, data, kind='cubic')
tau_new = np.linspace(tau_arr2[0], tau_arr2[-1], len(tau_arr2)*10)
a_new = np.linspace(a_arr2[0], a_arr2[-1], len(a_arr2)*10)
data1 = f(tau_new, a_new)
Xn, Yn = np.meshgrid(tau_new, a_new)
plt.figure(figsize=(15,8))
plt.pcolormesh(Xn, Yn, data1, cmap='RdBu', rasterized=True)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=12) 


#Plotting the path taken by the minimiser
print('Plotting minimiser path...')
tau_steps, a_steps, nsteps = find_min_2d(pos_init, accuracy, alpha, h)

plt.scatter(tau_steps[0], a_steps[0], color='magenta', s=25, label='initial position')
plt.scatter(tau_steps[1:], a_steps[1:], color='lime', s=25, zorder=2, label='path taken by minimiser')
plt.legend(fontsize=13)
plt.xlabel('tau (ps)', fontsize=20)
plt.ylabel('a', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Negative Log Likelihood (NLL)', fontsize=20)
#plt.savefig(r"C:\Users\Ivan\Desktop\Year 3\Computational Physics\Project\Graphs_2D\Minimiser.pdf", fmt="pdf", transparent=True, rasterized=True, bbox_tight=True)
print('Done')


tau_min = tau_steps[-1]
a_min = a_steps[-1]

print('Minimum of NLL(tau, a) found at tau = {0} ps, a = {1}. Found with {2} steps'.format(tau_min, a_min, nsteps))

