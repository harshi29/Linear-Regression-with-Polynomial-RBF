# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y
  
import numpy as np                       # For all our math needs
n = 750                                  # Number of data points
X = np.random.uniform(-7.5, 7.5, n)  # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e                        # True labels with noise

import matplotlib.pyplot as plt          
plt.figure()

# Plot the data
plt.scatter(X, y, 12, marker='o')           

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')
plt.show()

# X float(n, ): univariate data
# d int: degree of polynomial  
def polynomial_transform(X, d):
    yes = np.vander(X,d,increasing=True)
    return yes

# Phi float(n, d): transformed data
# y   float(n,  ): labels
def train_model(Phi, y):
    trans = Phi.T
    multiply = trans @ Phi 
    inverse = np.linalg.inv(multiply)
    multiply2 = inverse @ trans
    w_final = multiply2 @ y 
    return w_final

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model
def evaluate_model(Phi, y, w):
    #print(phi.shape, y.shape, w.shape)
    summation = 0
    n = len(y)
    for i in range (0,n):
        w_transpose = w.T 
        multiply = w_transpose @ Phi[i] 
        difference = y[i] - multiply
        squared_difference = difference**2 
        summation = summation + squared_difference
    MSE = summation/n
    return MSE

w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  
    Phi_trn = polynomial_transform(X_trn, d)                 
    w[d] = train_model(Phi_trn, y_trn)                       
    
    Phi_val = polynomial_transform(X_val, d)                
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  
    
    Phi_tst = polynomial_transform(X_tst, d)          
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d]) 

plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])
plt.show()

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
  X_d = polynomial_transform(x_true, d)
  y_d = X_d @ w[d]
  plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])
plt.show()

#----------------------------------------------------------Question 2 ---------------------------------------------------------------------

# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel

def radial_basis_transform(X, B, gamma=0.1):
    n = len(X)
    k = len(B)
    rbk = np.zeros(shape =(n,k))
    for i in range(0,n):
        for j in range(0,k):
            rbk[i,j]= np.exp((-1 * gamma)*(X[i]-B[j])**2)
    return rbk 
  
# Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter

def train_ridge_model(Phi, y, lam):
    Phi_transpose = Phi.T
    t2 = np.eye(len(X_trn)) * lam
    t_inv = np.linalg.inv((Phi_transpose @ Phi) + t2 )
    trm = t_inv @ Phi_transpose
    trm_final = trm @ y
    return trm_final 

w = {} 
validationErr = {} 
testErr = {} 

lam=0.001
B = X_trn
while lam<=1000: 
    Phi_trn = radial_basis_transform(X_trn,X_trn) 
    w[lam] = train_ridge_model(Phi_trn, y_trn,lam) 
    
    Phi_val = radial_basis_transform(X_val,X_trn) 
    validationErr[lam] = evaluate_model(Phi_val, y_val, w[lam]) 
    
    Phi_tst = radial_basis_transform(X_tst, X_trn) 
    testErr[lam] = evaluate_model(Phi_tst, y_tst, w[lam]) 
    lam*=10
    
plt.figure()
plt.plot(list(validationErr.keys()), list(validationErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(testErr.keys()), list(testErr.values()), marker='s', linewidth=3, markersize=12)
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.xscale("log")
plt.show()

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

lam=0.001
while lam<=1000:
    X_lam = radial_basis_transform(x_true,X_trn)
    y_lam = X_lam @ w[lam]
    plt.plot(x_true, y_lam, marker='None', linewidth=2)
    lam*=10

plt.legend(['true'] + list(range(-3,4,1)))
plt.axis([-8, 8, -15, 15])
plt.show()

