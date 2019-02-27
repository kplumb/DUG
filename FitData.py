#!/Users/kemp/anaconda3/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# We define our residuals function
def Residuals(params,t,data,err):
   
    # need to input x and data 
    model = fit_function(t, *params)
    
    # return the error weighted residuals
    R = (data - model)/err
    return R


# define the function we want to fit
def fit_function(t,a,b):
    return a*t + b


########################
# Load the data
########################
filepath = './DataFile.txt' 

D = np.loadtxt(filepath)
Dx = D[:,0]
Dy = D[:,1]
Derr = D[:,2]


# find the index where  t > 5, we only want to fit data up to this value
idx = np.nonzero(Dx >= 4)[0][0]

# create new data that includes only the points for t < 5\
Dx_new = Dx[0:idx]
Dy_new = Dy[0:idx]
Derr_new = Derr[0:idx]

# initial parameter guess
p_guess = [-1,10]

# run the least squares minimization routine, we are fitting the logarithm of the data

results = least_squares(Residuals, p_guess, verbose = 2, args = (Dx_new, np.log(Dy_new), Derr_new/Dy_new))

print("\n**********************************")
print("Result of the fit: ")
print("a:", results.x[0], "b:", results.x[1])
print("**********************************")


fig = plt.figure(figsize = [5,4.5])
ax = fig.add_subplot(111)

ax.errorbar(Dx,np.log(Dy), Derr/Dy, marker = 'o',mfc = 'grey', mec = 'grey', color = 'lightgrey', ms = 5,ls = 'None', label = 'Data')

ax.plot(Dx,fit_function(Dx,*results.x), lw =2, label = 'fit, t < 4 s', color = 'orange', zorder = 10)
ax.set_xlabel('time (s)', fontsize = 12)
ax.set_ylabel('log(counts)', fontsize = 12)
ax.set_ylim([0,3])
ax.set_xlim([0,10])
ax.legend()

fig.savefig('MyPlot.pdf')
plt.show()
