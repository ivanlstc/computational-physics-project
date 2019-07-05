#Importing necessary modules
import numpy as np
import scipy.integrate
import scipy.special
import scipy.interpolate
import matplotlib.pyplot as plt
from Project_1 import logged_fitfunc1, NLL1, find_tau_min_test, find_tau_min

#Reading the datafile and extracting the decay times and errors
with open('lifetime-2018.txt', 'r') as f:
    lines = [np.float(element) for element in f.read().split()]
dtimes = lines[::2]
derrors = lines[1::2]

#Range of tau values used to perform minimisation task
t_searchtest = [-1, 0.5, 1] #t-values used to test the minimser with cosh(t) function
t_search = [0.3, 0.38, 0.45] #tau values used to perform minimisation of NLL function;
                             #also used to find errors associated with tau_min

#Output of 1D minimiser test function and actual 1D minimiser
tau_min_test_output = find_tau_min_test(t_searchtest)
tau_min_output = find_tau_min(t_search)

#Minimum value of cosh(t) and NLL(tau)
cosh_min = tau_min_test_output[0][0] 
tau_min = tau_min_output[0][0]

#Number of steps taken for each 1D minimiser
cosh_steps = tau_min_test_output[1]
tau_steps = tau_min_output[1]

#Minimum of cosh(t) is t=0, as expected. Min of NLL(tau) printed
print('Minimum of cosh(t) = {0}, found with {1} steps'.format(cosh_min, cosh_steps))
print('Minimum of NLL(tau) = {0} ps, found with {1} steps'.format(tau_min, tau_steps))
