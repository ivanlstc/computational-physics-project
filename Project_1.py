#This section below provides a brief overview of the modules required and variables used.
#These steps are not required as they are already coded into the individual files,
#but are here for reference.

#Importing necessary modules
import numpy as np
import scipy.integrate
import scipy.special
import scipy.interpolate
import matplotlib.pyplot as plt

#Reading the datafile and extracting the decay times and errors
with open('lifetime-2018.txt', 'r') as f:
    lines = [np.float(element) for element in f.read().split()]
dtimes = lines[::2]
derrors = lines[1::2]

#1D_1, 2D_1 - array of time values used to plot the theoretical decay function for
#various tau and sigma values, and also a values in the 2D case
time_arr = np.linspace(-1, 2, 10000)

#1D_2 - array of tau values to plot 1D NLL function
tau_arr0 = np.linspace(0.1, 2, 50)

#1D_3, 1D_4 - range of tau values used to perform minimisation task
t_searchtest = [-1, 0.5, 1] #t-values used to test the minimser with cosh(t) function
t_search = [0.3, 0.38, 0.45] #tau values used to perform minimisation of NLL function;
                             #also used to find errors associated with tau_min

#2D_2 - range of tau and a values to plot 2D NLL function
tau_arr1 = np.linspace(0.1, 0.9, 10)
a_arr1 = np.linspace(0.001, 1, 10)

#2D_3 - range of tau and a values to plot zoomed-in 2D NLL function, used for
#the 2D minimisation task
tau_arr2 = np.linspace(0.405, 0.415, 20)
a_arr2 = np.linspace(0.975, 0.9875, 20)

#2D_3 - parameters used in find_min_test2d function to test the 2D minimiser
pos_init_test = [-1.5, 0.5]
accuracy_test = 0.0000000001
alpha_test = 0.01
h_test = 0.0000001

#2D_3 - parameters used in find_min_2d function to minimse the 2D NLL function
pos_init = [0.41, 0.98]
accuracy = 0.0000001
alpha = 0.00001
h = 0.000001

###############################################################################
###############################################################################
#This sections includes functions that are imported into the other files and
#used to perform the various tasks required for the project.
###############################################################################
###############################################################################

###############################################################################
###############################################################################

#1D Case, Question 3

###############################################################################
###############################################################################
def fitfunc1(t, tau, sigma):
    """
    Finds the expected distribution of the decay time of the D0 meson for a
    given average lifetime (tau) and decay time error (sigma) WITHOUT a background
    contribution. Refer to equation (3) given on the project 1 sheet.
    
    Returns the value of the expected decay time distribution for a given t,
    tau and sigma. Returns a list if t is given as a list.
    """
    exponential = np.exp((sigma**2/(2*tau**2)) - t/tau)/(2*tau)
    error_func = scipy.special.erfc((sigma/tau - t/sigma)/np.sqrt(2))
    fit = exponential*error_func
    return fit


def fitfunc2(t, tau, sigma, a):
    """
    Finds the expected distribution of the decay time of the D0 meson for a
    given average lifetime (tau) and decay time error (sigma) WITH a background
    contribution parameterised by 'a'. Refer to equation (8) given on the
    project 1 sheet.
    
    Returns the value of the expected decay time distribution for a given t,
    tau, sigma, and a. Returns a list if t is given as a list.
    """
    exponential = np.exp((sigma**2/(2*tau**2)) - t/tau)/(2*tau)
    error_func = scipy.special.erfc((sigma/tau - t/sigma)/np.sqrt(2))
    signal = a*exponential*error_func
    background = (1 - a)*np.exp(-t**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    fit = signal + background
    return fit


def test_pdf1(tau, sigma):
    """
    Integrates fitfunc1 by quadrature for a given tau and sigma value. As the function
    associated with fitfunc1 is a probability density function, its integral over
    all space should add up to 1. The integral is evaluated from -50 ps and +50 ps.
    This domain is deemed sufficient for the purposes of this task as fitfunc1
    exponentially decays at high values of t and also tails off at negative t,
    making the integrand insignificant at those values. The evaluation
    of the integral is not exactly 1 (as can be seen in the 1D_1 task), but is
    close enough to 1 to demonstrate that the function is valid as a PDF.
    
    Returns the value of the integral of fitfunc1 for a given tau and sigma value.
    """
    I1 = scipy.integrate.quad(fitfunc1, -50, 50, args=(tau, sigma))
    return I1


def test_pdf2(tau, sigma, a):
    """
    Integrates fitfunc2 by quadrature for given tau, sigma, and 'a' values.
    Similar to fitfunc1, the function associated with fitfunc2 is a probability
    density function and its integral over all space should add up to 1.
    The integral is evaluated from -50 ps and +50 ps.
    This domain is again deemed sufficient  for the purposes of this task
    (refer to test_pdf1).
    
    Returns the value of the integral of fitfunc2 for a given tau, sigma, and 'a' value.
    """
    I2 = scipy.integrate.quad(fitfunc2, -50, 50, args=(tau, sigma, a))
    return I2


def logged_fitfunc1(t, tau, sigma):
    """
    Returns the logarithm of fitfunc1 evaluated at t for a given tau and sigma.
    Returns a list of t is given as a list/array.
    """
    logged_exponential = (sigma**2/(2*tau**2) - t/tau) - np.log(2*tau)
    logged_error_func = np.log(scipy.special.erfc((sigma/tau - t/sigma)/np.sqrt(2)))
    logfit = logged_exponential + logged_error_func
    return logfit


def NLL1(tau):
    """
    Calculates the negative log likelihood (NLL) for a given value of tau, using
    the given t and sigma measurements. For this case, NLL only depends on tau,
    so a 1D minimisation task can be performed to find the tau value which
    minimises the NLL.
    
    Returns the NLL evaluated at a given value of tau, or an array of NLL values
    if a list of various tau values is the input.
    """
    if isinstance(tau, (list, np.ndarray)):
        NLL_1 = [] #NLL_1 is a list if tau is a list/array
        for tau_val in tau:
            NLL_sum = 0
            for i in range(len(dtimes)):
                NLL_sum -= logged_fitfunc1(t=dtimes[i], tau=tau_val, sigma=derrors[i])
            NLL_1.append(NLL_sum)
    else:
        NLL_1 = 0 #NLL_1 is a floating point if tau is just a number
        for i in range(len(dtimes)):
            NLL_1 -= logged_fitfunc1(t=dtimes[i], tau=tau, sigma=derrors[i])
            
    return NLL_1


def find_tau_min_test(t_vals):
    """
    Tests the 1D minimiser used in 1D_3. The minimiser is tested on f(t) = cosh(t),
    which has a minimium at t=0. Uses the parabolic method outlined in Chapter
    7 of the course to find the minimum. If f(t0), f(t1), f(t2) are known for
    a given t0, t1, t2, a t3 can be found such that t3 is the minimum of the parabolic
    function interpolated using t0, t1 and t2 (refer to equation 7.6). From
    [t0, t1, t2, t3], the t value which yields the largest f(t) value is discarded,
    and the process is repeated until the desired accuracy of the minimum is achieved.
    
    At each iteration, f(t) is compared between the new and old t values which
    yield the minimum at each step. If the difference is smaller than a relatively
    small number, the process stops as it is assumed that the minimum has been reached
    to a reasonable degree of accuracy.
    
    Returns the [t0, t1, t2] values used in the last iteration, and the number
    of steps taken to reach the minimum using the parabolic method. Since [t0, t1, t2]
    are sorted according to [f(t0), f(t1), f(t2)] at the end of each iteration,
    it is assumed that t0 is the value which yields the minimum f value.
    """
    steps = 0
    cosh_oldmin = 0
    cosh_newmin = np.cosh(t_vals[1])
    cosh_mindiff = np.abs(cosh_newmin)
    
    while cosh_mindiff >= 0.0000001: #condition for stopping the loop; desired accuracy achieved
        t_vals.sort()
        
        t0 = t_vals[0]
        t1 = t_vals[1]
        t2 = t_vals[2]
        y0 = np.cosh(t0)
        y1 = np.cosh(t1)
        y2 = np.cosh(t2)
        
        t3_num = y0*(t2**2 - t1**2) + y1*(t0**2 - t2**2) + y2*(t1**2 - t0**2)
        t3_denom = 2*(y0*(t2 - t1) + y1*(t0 - t2) + y2*(t1 - t0))
        t3 = t3_num/t3_denom #equation 7.6
        y3 = np.cosh(t3)
        
        cosh_vals = [y0, y1, y2, y3]
        t_vals = [t0, t1, t2, t3]
        
        #Values sorted in ascending order according to the value of the cosh function;
        #the t value which yields the smallest cosh(t) value will be the first element
        #in the list
        cosh_vals, t_vals = (list(x) for x in zip(*sorted(zip(cosh_vals, t_vals))))
        cosh_vals = cosh_vals[:-1]
        t_vals = t_vals[:-1]
        
        #Compares the value of the old and new minima to see when the desired
        #accuracy is achieved
        cosh_oldmin = cosh_newmin
        cosh_newmin = cosh_vals[0]
        cosh_mindiff = np.abs(cosh_newmin - cosh_oldmin)
        steps += 1

    return t_vals, steps


def find_tau_min(t_vals):
    """
    Uses the same 1D minimiser previously tested. Instead of f(t) = cosh(t),
    the function being minimised is now f(tau) = NLL(tau). Uses the same parabolic
    method outlined in Chapter 7 of the course to find the minimum. Refer
    to the docstring in find_tau_min_test for an explanation of the method.
    
    Returns the [t0, t1, t2] values used in the last iteration, and the number
    of steps taken to reach the minimum using the parabolic method.
    """
    steps = 0
    NLL_oldmin = 0
    NLL_newmin = NLL1(t_vals[1])
    NLL_mindiff = np.abs(NLL_newmin)
    
    while NLL_mindiff >= 0.0000001:
        t_vals.sort()
        
        t0 = t_vals[0]
        t1 = t_vals[1]
        t2 = t_vals[2]
        y0 = NLL1(t0)
        y1 = NLL1(t1)
        y2 = NLL1(t2)
        
        t3_num = y0*(t2**2 - t1**2) + y1*(t0**2 - t2**2) + y2*(t1**2 - t0**2)
        t3_denom = 2*(y0*(t2 - t1) + y1*(t0 - t2) + y2*(t1 - t0))
        t3 = t3_num/t3_denom #equation 7.6
        y3 = NLL1(t3)
        
        NLL_vals = [y0, y1, y2, y3]
        t_vals = [t0, t1, t2, t3]
        
        #Values sorted in ascending order according to the value of the NLL(tau);
        #the tau value which yields the smallest NLL value will be the first element
        #in the list
        NLL_vals, t_vals = (list(x) for x in zip(*sorted(zip(NLL_vals, t_vals))))
        NLL_vals = NLL_vals[:-1]
        t_vals = t_vals[:-1]
        
        #Compares the value of the old and new minima to see when the desired
        #accuracy is achieved
        NLL_oldmin = NLL_newmin
        NLL_newmin = NLL_vals[0]        
        NLL_mindiff = np.abs(NLL_newmin - NLL_oldmin)
        steps += 1

    return t_vals, steps


def find_tau_err(t_vals):
    """
    Finds the errors associated with the tau value found using the minimiser in
    find_tau_min. Uses two methods of finding the errors:
        1. finds the tau values which yield a change of +0.5 in the NLL; two values
        are found, tau_plus and tau_minus, and their difference with tau_min is
        calculated
        2. finds the tau error associated with the curvature of the function at the
        minimum, which is found by evaluating the second derivative of the function
        near the minimum using the [t0, t1, t2] values taken from the last iteration
        of the minimiser. Refer to the second derivative of equation 7.5 in the notes.
        The error is equal to 1/sqrt(second derivative at the minimum). Can also
        refer to Lecture8 from Statistics of Measurement course.
    
    Returns the tau value which minimises the 1D NLL, the +/- tau error associated
    with the NLL changing by 0.5 (absolute value, not factor), and the tau error
    associated with the curvature of the function near the minimum.
    """
    tau_min_vals, nsteps = find_tau_min(t_vals)

    #Method 1: finding tau_plus and tau_minus
    tau_min = tau_min_vals[0]
    NLL1_min = NLL1(tau_min)
    #Range of tau values used to 'scan' for tau_plus/minus by evaluating the NLL
    #at each point and see if it differs by 0.5 relative to NLL at the minimum
    tau_err_search = np.linspace(0.40, 0.41, 500)
    NLL1_err_search = NLL1(tau_err_search)
    tau_plus_search = []
    tau_minus_search = []
    for i in range(len(NLL1_err_search)):
        NLL1_diff = NLL1_err_search[i] - NLL1_min #finding difference between min NLL and NLL at a given tau
        if np.abs(NLL1_diff - 0.5) <= 0.05: #can sacrifice accuracy here, no need for that many sig figs
            if tau_err_search[i] > tau_min: #tau_plus
                tau_plus_search.append(tau_err_search[i])
            if tau_err_search[i] < tau_min: #tau_minus
                tau_minus_search.append(tau_err_search[i])
    
    #Method is limited by step size anyway; approximate values of tau_plus/minus
    #is sufficient, we don't need that many significant figures for the error
    tau_plus = tau_plus_search[0]
    tau_minus = tau_minus_search[0]
    plus_err = tau_plus - tau_min
    minus_err = tau_min - tau_minus

    #Method 2: finding tau error associated with second derivative
    sec_deriv_search = tau_min_vals
    t0 = sec_deriv_search[0]
    t1 = sec_deriv_search[1]
    t2 = sec_deriv_search[2]
    
    y0 = NLL1(t0)
    y1 = NLL1(t1)
    y2 = NLL1(t2)
    
    #Refer to Lecture8 of StoM course, and second derivative of equation 7.5 in
    #Computational Physics notes
    d = (t1 - t0)*(t2 - t0)*(t2 - t1)
    sec_deriv = (2/d)*(y0*(t2 - t1) + y1*(t0 - t2) + y2*(t1 - t0))
    sec_deriv_err = np.sqrt(1/sec_deriv)
    
    return tau_min, plus_err, minus_err, sec_deriv_err

###############################################################################
###############################################################################

#2D Case, Question 4

###############################################################################
###############################################################################
    
def NLL2(tau, a):
    """
    Calculates the negative log likelihood (NLL) for a given value of tau, and 'a',
    using the given t and sigma measurements. For this case, NLL depends on both
    tau and 'a', so the minimisation task will have to be performed in 2D
    to find the tau and 'a' values which minimise the NLL.
    
    Returns the NLL evaluated at a given value of tau and 'a', or an array of
    NLL values if either tau or 'a' is given as a list/array. If both tau and
    a are given as a list/array, the function returns a 2D array of the NLL
    values.
    """
    #tau and 'a' are both lists; returns a matrix, used for 2D NLL plots
    if isinstance(tau, (list, np.ndarray)) and isinstance(a, (list, np.ndarray)):
        ncols = len(tau)
        nrows = len(a)
        NLL_2 = np.zeros((ncols, nrows)) 
        for i in range(nrows):
            for j in range(ncols):
                NLL_sum = 0 #calculating the NLL value for each element using the data
                for k in range(len(dtimes)):
                    NLL_sum -= np.log(fitfunc2(t=dtimes[k], tau=tau[j], sigma=derrors[k], a=a[i]))
                NLL_2[i][j] = NLL_sum #assigning the NLL value to the appropriate element in matrix
    
    #either tau OR 'a' is a list, not both; returns an array, used for finding errors
    #as tau and a are varied independently to find their +/- errors in 2D
    elif isinstance(tau, (list, np.ndarray)) and not isinstance(a, (list, np.ndarray)):
        n = len(tau) #varying tau, a is constant
        NLL_2 = np.zeros(n) 
        for i in range(n):
            NLL_sum = 0 
            for k in range(len(dtimes)):
                NLL_sum -= np.log(fitfunc2(t=dtimes[k], tau=tau[i], sigma=derrors[k], a=a))
            NLL_2[i] = NLL_sum   
    elif not isinstance(tau, (list, np.ndarray)) and isinstance(a, (list, np.ndarray)):
        n = len(a) #varying a, tau is constant
        NLL_2 = np.zeros(n)
        for j in range(n):
            NLL_sum = 0
            for k in range(len(dtimes)):
                NLL_sum -= np.log(fitfunc2(t=dtimes[k], tau=tau, sigma=derrors[k], a=a[j]))
            NLL_2[j] = NLL_sum
        
    #neither tau nor 'a' is a list; returns a floating point value for the NLL,
    #essentially evaluating the NLL at a point
    else:
        NLL_2 = 0
        for i in range(len(dtimes)):
            NLL_2 -= np.log(fitfunc2(t=dtimes[i], tau=tau, sigma=derrors[i], a=a))
            
    return NLL_2


def find_grad(f, x, y, h):
    """
    Finds the gradient of a function at a point using the central difference
    scheme (CDS, Chapter 8 of CompPhysics notes)
    
    Returns the gradient of f evaluated at (x, y), for a given small change in
    function, h.
    """
    grad_x = (f(x + h, y) - f(x - h, y))/(2*h)
    grad_y = (f(x, y + h) - f(x, y - h))/(2*h)
    grad = np.array([grad_x, grad_y])
    return grad


def test_func2d(x, y):
    """
    Test function used for the 2D minimiser.
    
    Returns the value of the function f = x**2 + 2*y**2 + x*y + 3*x for a given (x,y).
    """
    f = x**2 + 2*y**2 + x*y + 3*x
    return f


def find_min_test2d(pos_init, accuracy, alpha, h):
    """
    Tests the 2D minimiser used in 2D_3. The minimiser is tested on the function
    given in test_func2d, which is f = x**2 + 2*y**2 + x*y + 3*x and has a minimum
    as (x,y) = (-12/7, 3/7). Uses the gradient method outlined in Chapter
    7 of the course to find the minimum. The minimiser starts at a given 
    initial position. The gradient of the function is evaluated at a point
    (x,y), and takes a step in the opposite direction to move towards
    the minimum. Iterating this will eventually find the minimum if alpha
    (related to the step size) is small enough, and h is not too large.
    The process is repeated until the desired accuracy of the minimum is achieved.
    
    Returns arrays of the (x,y) coordinates taken for each step, and the number
    of steps taken to reach the minimum using the gradient method.
    """
    x_steps = []
    y_steps = []
    pos_old = np.array([0.0, 0.0])
    pos_new = np.array(pos_init)
    x_diff = np.abs(pos_new[0] - pos_old[0])
    y_diff = np.abs(pos_new[1] - pos_old[1])
    
    steps = 0
    #goes through loop until desired accuracy for both values achieved
    while (x_diff >= accuracy or y_diff >= accuracy): 
        x = pos_new[0]
        y = pos_new[1]
        x_steps.append(pos_new[0])
        y_steps.append(pos_new[1])
        
        pos_old = np.copy(pos_new)
        pos_new -= alpha*find_grad(test_func2d, x, y, h) #evaluating grad of test function
        x_diff = np.abs(pos_new[0] - pos_old[0])
        y_diff = np.abs(pos_new[1] - pos_old[1])
        
        steps += 1

    return x_steps, y_steps, steps




def find_min_2d(pos_init, accuracy, alpha, h):
    """
    Uses the 2D minimiser previously tested. The minimiser is now used on the
    2D NLL function, which depends on both tau and 'a'.
    Again uses the gradient method to find the tau and 'a' values which yield
    the minimum NLL (refer to find_min_test2d). The minimiser starts at a [tau, a]
    coordinate [0.41, 0.98], as it is visibly close to the minimum, as seen in
    2D_2. 
    
    Returns arrays of the (tau, a) coordinates taken for each step, and the number
    of steps taken to reach the minimum using the gradient method.
    """
    tau_steps = []
    a_steps = []
    pos_old = np.array([0.0, 0.0])
    pos_new = np.array(pos_init)
    tau_diff = np.abs(pos_new[0] - pos_old[0])
    a_diff = np.abs(pos_new[1] - pos_old[1])
    
    steps = 0
    #goes through loop until desired accuracy for both values achieved; limiting number of steps just in case
    while (tau_diff >= accuracy or a_diff >= accuracy) and steps <= 1000:
        tau = pos_new[0]
        a = pos_new[1]
        tau_steps.append(pos_new[0])
        a_steps.append(pos_new[1])
        
        pos_old = np.copy(pos_new)
        pos_new -= alpha*find_grad(NLL2, tau, a, h) #evaluating grad of NLL function instead of the test function now
        tau_diff = np.abs(pos_new[0] - pos_old[0])
        a_diff = np.abs(pos_new[1] - pos_old[1])
        
        steps += 1

    return tau_steps, a_steps, steps



def find_min_err2d(pos_init, accuracy, alpha, h):
    """
    Finds the errors associated with the tau and 'a' values found using the
    2D minimiser in find_min_2d. Only uses the method of finding the tau and 
    'a' values which yield a change of +0.5 in the NLL. This is done by
    independently varying tau and 'a' (while keeping the other one constant)
    and evaluating NLL at each point to see when it changes by 0.5 Four values
    in total are found: tau_plus and tau_minus, and a_plus and a_minus.
    Their difference with tau_min and a_min respectively are calculated to get
    their associated error values.

    Returns the tau and 'a' values which minimise the 2D NLL, and the +/- tau
    and 'a' errors associated with the NLL changing by 0.5 (as an absolute value).
    """
    #Calling the find_min_2d function to find the tau and 'a' values associated
    #with the NLL minimum
    tau_steps, a_steps, nsteps = find_min_2d(pos_init, accuracy, alpha, h)

    tau_min = tau_steps[-1]
    a_min = a_steps[-1]
    NLL2_min = NLL2(tau_min, a_min)
    
    #Range of tau and 'a' values used to 'scan' for tau_plus/minus and a_plus/minus
    #by evaluating the NLL at each point and see if it differs by 0.5 relative
    #to NLL at the minimum
    tau_err_search = np.linspace(0.40, 0.415, 500)
    a_err_search = np.linspace(0.975, 0.995, 500)
    
    #The tau and 'a' values are varied independently;
    #arrays created of the 2D Nll values for varying tau and varying a
    NLL2_tau_err_search = NLL2(tau_err_search, a_min) #a fixed, changing tau
    NLL2_a_err_search = NLL2(tau_min, a_err_search) #tau fixed, changing a

    tau_plus_search = []
    tau_minus_search = []
    a_plus_search = []
    a_minus_search = []
    
    #First, keeping a_min fixed and varying the value of tau to find tau_plus/minus
    for i, NLL2_tau_val in enumerate(NLL2_tau_err_search):
        NLL2_tau_diff = NLL2_tau_val - NLL2_min
        if np.abs(NLL2_tau_diff - 0.5) <= 0.005: #can sacrifice accuracy here, no need for that many sig figs
            if tau_err_search[i] > tau_min:
                tau_plus_search.append(tau_err_search[i])
            if tau_err_search[i] < tau_min:    
                tau_minus_search.append(tau_err_search[i])
    
    #Then, keeping tau_min fixed and varying the value of 'a' to find a_plus/minus
    for j, NLL2_a_val in enumerate(NLL2_a_err_search):
        NLL2_a_diff = NLL2_a_val - NLL2_min
        if np.abs(NLL2_a_diff - 0.5) <= 0.005: #can sacrifice accuracy here, no need for that many sig figs
            if a_err_search[j] > a_min:
                a_plus_search.append(a_err_search[j])
            if a_err_search[j] < a_min:    
                a_minus_search.append(a_err_search[j])
    
    #Like the 1D case of finding the error, the method is limited by step size
    #but we don't that many sig figs for the error anyway
    tau_plus = tau_plus_search[0]
    tau_minus = tau_minus_search[0]
    tau_plus_err = tau_plus - tau_min
    tau_minus_err = tau_min - tau_minus
    
    a_plus = a_plus_search[0]
    a_minus = a_minus_search[0]
    a_plus_err = a_plus - a_min
    a_minus_err = a_min - a_minus

    return tau_min, tau_plus_err, tau_minus_err, a_min, a_plus_err, a_minus_err
    