#Importing necessary modules
import numpy as np
import scipy.integrate
import scipy.special
import scipy.interpolate
import matplotlib.pyplot as plt
from Project_1 import fitfunc2, NLL2, find_grad, find_min_2d, find_min_err2d

#Reading the datafile and extracting the decay times and errors
with open('lifetime-2018.txt', 'r') as f:
    lines = [np.float(element) for element in f.read().split()]
dtimes = lines[::2]
derrors = lines[1::2]

#Parameters used in find_min_2d function to minimse the 2D NLL function;
#also used to find the associated errors in this case using the method of finding
#where the NLL changes by 0.5. Refer to the find_min_err2d function
pos_init = [0.41, 0.98]
accuracy = 0.0000001
alpha = 0.00001
h = 0.000001

#Outputs the tau and a values which yield the minimum NLL in 2D, and also their
#associated plus and minus errors
print('Finding errors...')
params_err_output = find_min_err2d(pos_init, accuracy, alpha, h)

tau_min = params_err_output[0]
tau_plus_err = params_err_output[1]
tau_minus_err = params_err_output[2]
a_min = params_err_output[3]
a_plus_err = params_err_output[4]
a_minus_err = params_err_output[5]
print('Done')

print('NLL minimised when tau = {0} ps, and a = {1}'.format(tau_min, a_min))
print('tau errors associated with NLL changing by 0.5: tau_plus = {0} ps, tau_minus = {1} ps'.format(tau_plus_err, tau_minus_err))
print('a errors associated with NLL changing by 0.5: a_plus = {0}, a_minus = {1}'.format(a_plus_err, a_minus_err))
