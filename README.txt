Folder: Li-Ivan-CP2018-Project1-Code.zip

Name: Ivan Shiu Cheung Li
CID: 01200003

Included files:
- Project_1.py
- lifetime-2018.txt
- 1D_1 Fit Function.py
- 1D_2 Likelihood Function.py
- 1D_3 Minimise.py
- 1D_4 Accuracy of Fit Result.py
- 2D_1 Fit Function.py
- 2D_2 Likelihood Function.py
- 2D_3 Minimise.py
- 2D_4 Accuracy of Fit Result.py
- README.txt


Commentary:
Python is my preferred coding language as it provides easy access to good libraries (e.g. numpy, scipy, matplotlib, etc.), and the data structures are easy to understand and manipulate. I initially wrote all of the code on one file, but I have separated the code into multiple files so the output is much cleaner for each task. I hope that my use of comments and docstrings will allow you to easily guide you through my code and understand my thought-process when I wrote/tweaked the code for each task. Similar to the assignment, not many references were used in the report since most of the technical detail was based on the lecture notes/slides/project sheet.


Brief explanation about each file and what output to expect/how long it takes:

- Project_1.py
The 'master' file including all the necessary functions that are to be imported and used in each of the other files; outputs nothing

- lifetime-2018.txt
The data file containing the (t_i, sigma_i) measurements

- 1D_1 Fit Function.py (~1-2 seconds)
Plots histograms of the data
Plots Eqn. 3 on the project sheet for various tau and sigma values
Evaluates the integral of the plots to ensure that they go to 1

- 1D_2 Likelihood Function.py (~4 seconds)
Plots the negative log likelihood (NLL) function from Eqn. 6 in 1D

- 1D_3 Minimise.py (~3-4 seconds)
Tests the 1D minimiser using cosh(t), should output t_min=0 as expected
Performs minimisation task on 1D NLL function using parabolic method
Outputs tau_min and the number of steps taken

- 1D_4 Accuracy of Fit Result.py (~30-35 seconds)
Finds the associated errors with tau_min by finding when the NLL changes by 0.5, and by analysing 2nd derivative
Outputs tau_min, tau_plus/tau_minus, and error associated with curvature

- 2D_1 Fit Function.py (~1 second)
Plots Eqn. 8 for various 'a' values
Evaluates the integral of the plots to ensure that they go to 1

- 2D_2 Likelihood Function.py (~10 seconds)
Plots the 2D NLL function

- 2D_3 Minimise.py (~1:05 minutes)
Tests the 2D minimiser using (f = x**2 + 2*y**2 + x*y + 3*x), minimum should be at (x=-12/7, y=3/7)
Performs minimisation task on 2D NLL function using gradient method
Plots a zoomed in version of the 2D NLL near the minimum, and path taken by the 2D minimiser

- 2D_4 Accuracy of Fit Result.py (~2:15 minutes)
Finds the associated errors with tau_min and a_min by finding when the NLL changes by 0.5
Outputs tau_min, tau_plus/tau_minus, a_min, a_plus/a_minus

-------------------------------

Ignore this part, testing git commit function from linux terminal.
