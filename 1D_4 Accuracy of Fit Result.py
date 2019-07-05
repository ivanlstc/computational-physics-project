#Importing necessary modules
import numpy as np
import scipy.integrate
import scipy.special
import scipy.interpolate
import matplotlib.pyplot as plt
from Project_1 import logged_fitfunc1, NLL1, find_tau_min, find_tau_err

#Reading the datafile and extracting the decay times and errors
with open('lifetime-2018.txt', 'r') as f:
    lines = [np.float(element) for element in f.read().split()]
dtimes = lines[::2]
derrors = lines[1::2]

#Same range of tau values used in previous minimum search is used to find the
#error associated with the value; refer to find_tau_err()
t_search = [0.3, 0.38, 0.45] 

#Outputs tau_plus, tau_minus, and tau_err associated with the curvature/second derivative
print('Finding errors...')
tau_err_output = find_tau_err(t_search)

tau_min = tau_err_output[0]
tau_plus = tau_err_output[1]
tau_minus = tau_err_output[2]
tau_sec_deriv_err = tau_err_output[3]
print('Done')

print('Tau which minimises the NLL = {0} ps'.format(tau_min))
print('Errors associated with NLL changing by 0.5: tau_plus = {0} ps, tau_minus = {1} ps'.format(tau_plus, tau_minus))
print('Errors associated with NLL curvature: tau_sec_deriv_err = {0} ps'.format(tau_sec_deriv_err))