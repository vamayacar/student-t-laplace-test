

# Victor Amaya @ Duke University.


import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.linalg import cholesky, cho_solve, sqrtm, block_diag
from scipy.optimize import minimize
#from sklearn.gaussian_process.kernels import RBF
#from scipy import stats
#from scipy.stats import loggamma


''' ######################################################################
            CODE FOR THE STUDENT-t DISTRIBUTION COMPARISON
###################################################################### '''


### Newton Method to estimate the mode (using natural gradiente)
##  as in paper https://arxiv.org/pdf/1712.07437.pdf


''' START of function definitions '''

def kernel_1D(X1, X2, theta = [1,1]):

    sqdist = (X1 ** 2) + (X2 ** 2) - 2 * np.dot(X1, X2)
    return theta[1] ** 2 * np.exp(-0.5 / theta[0] ** 2 * sqdist)

def kernel_RBF(X1, X2, theta = [1,1]):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).
        theta: Kernel parameters

    Returns:
        (m x n) matrix
    """

    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return theta[1] ** 2 * np.exp(-0.5 / theta[0] ** 2 * sqdist)


def posterior_mode(X, y, K, nu, max_iter=10 ** 3, tol=1e-9):
    """
    K:   prior covariance matrix K = diag(K_1,K_2).
         K_k is cov(f_k(x_i), f_k(x_j))_{i,j} ; k =1,2.

    logL_grad: is the gradient of the log-likelihood.

    W_fisher:  is the fisher matrix information E[W].
    """

    ''' here i will use same notation as in Algorithm 3.1 from book 
                Gaussian Processes for Machine Learning
                 https://gaussianprocess.org/gpml/chapters/RW.pdf
    '''

    n = y.shape[0] * 2
    f_h = np.ones(n) * 0.1

    for i in range(max_iter):
        logL_grad = Grad_logL(y, f_h, nu)
        W_fisher = W_FisherM(f_h, nu)  #takes the place of W in algo 3.1

        #K_inv = np.linalg.inv(K)
        #Q_inv = np.linalg.inv(K_inv + W_fisher)
        #f_h_new = Q_inv.dot(W_fisher.dot(f_h) + logL_grad)

        W_sqrt = sqrtm(W_fisher)
        L = np.linalg.cholesky( np.eye(n) + (np.matmul(W_sqrt,K)).dot(W_sqrt) )
        b = W_fisher.dot(f_h) + logL_grad

        aux_1 = np.linalg.solve(L, (np.matmul(W_sqrt,K)).dot(b) )
        a = b - np.linalg.solve(W_sqrt.dot( np.transpose(L) ),  aux_1 )
        f_h_new = K.dot(a)
        #print(success until ', i)
        f_h_diff = np.abs(f_h_new - f_h)
        f_h = f_h_new

        if not np.any(f_h_diff > tol):
            break

    return f_h



## need to compute W_fisher

# going to assume that f in an (2n)x1 vetor. the first n entries
# corresponds to f_1(x) and the rest corresponds to f_2(x).


# From appendix B.1 of paper:
def W_Matrix(y, f, nu):
    z = np.zeros_like(y) * 0.0
    N = np.size(f)
    W = np.eye(N) * 0.0

    for i in range(N // 2):
        z[i] = (y[i] - f[i]) / np.exp(f[i + (N // 2)])

    for i in range(N):
        for j in range(N):

            if (i == j):
                if (i < N // 2):
                    temp_val = 2 / (1 + z[i] ** 2 / nu) ** 2 - 1 / (1 + z[i] ** 2 / nu)
                    W[i, j] = 1 / (np.exp(f[i + N // 2])) ** 2 * (1 + 1 / nu) * temp_val
                else:
                    k = i - N // 2
                    W[i, j] = 2 * (1 / nu + 1) * z[k] ** 2 * 1 / (1 + z[k] ** 2 / nu) ** 2

            if ((i < N // 2) & (j == (i + N // 2))):
                W[i, j] = 2 / (np.exp(f[i + N // 2])) ** 2 * (1 + 1 / nu) * z[i] / (1 + z[i] ** 2 / nu) ** 2
                print('test1', W[i, j], (i, j))

            if ((i >= N // 2) & (j == (i - N // 2))):
                k = i - N // 2
                W[i, j] = 2 / (np.exp(f[i])) ** 2 * (1 + 1 / nu) * z[k] / (1 + z[k] ** 2 / nu) ** 2
                print('test2', W[i, j], (i, j))

    return W


### Gradient vector

# From appendix B.3 of paper:
def Grad_logL(y, f, nu):
    z = np.zeros_like(y) * 0.0
    N = np.size(f)

    v1 = v2 = np.zeros_like(y) * 0.0

    for i in range(N // 2):
        z[i] = (y[i] - f[i]) / np.exp(f[i + (N // 2)])
        v1[i] = (1 + 1 / nu) * z[i] / (np.exp(f[i + N // 2]) * (1 + z[i] ** 2 / nu))
        v2[i] = (z[i] ** 2 - 1) / (1 + z[i] ** 2 / nu)

    return np.append(v1, v2)


###     Expected Fisher matrix
# From appendix B.2 of paper:

def W_FisherM(f, nu):
    N = np.size(f)
    W = np.eye(N) * 0.0

    for i in range(N):
        if (i < N // 2):
            W[i, i] = (nu + 1) / (nu + 3) * np.exp(-2 * f[N // 2 + i])
        else:
            W[i, i] = (2 * nu) / (nu + 3)

    return W

# Log likelihood function
def logL(y, f, nu):
    n = y.size
    temp = 0

    for i in range(n):
        v1 = scipy.special.loggamma((nu + 1) / 2)
        v2 = scipy.special.loggamma(nu / 2) + f[n + i] + 0.5 * np.log(np.pi * nu)
        v3 = ((nu + 1) / 2) * np.log(1 + (1 / nu) * ((y[i] - f[i]) / np.exp(f[n + i])) ** 2)
        temp += (v1 - v2 - v3)
    return temp


# negative log likelihood marginal function approximation
def nll_fn(X, y):
    """
    Returns the negative log-likelihood function for data X, y.
    """

    y = y.ravel()

    def nll(theta):
        # K_a = K
        # K_a = kernel_1D(X,X,theta)   #= K_(X, theta)

        #### Commputation of matrix K
        K1 = kernel_RBF(X, X, theta[0:2])
        K2 = kernel_RBF(X, X, theta[2:4])
        K_a = block_diag(K1, K2)
        K_a_inv = np.linalg.inv(K_a)

        # posterior mode depends on theta (via K_a)
        f_hat = posterior_mode(X, y, K_a, theta[4]).ravel()
        W = W_FisherM(f_hat, theta[4])

        n = y.size
        temp1 = sqrtm(W)
        temp2 = np.eye(2 * n) + temp1 @ K_a @ temp1

        ll = - 0.5 * f_hat.T.dot(K_a_inv).dot(f_hat) \
             - 0.5 * np.linalg.slogdet(temp2)[1] \
             + logL(y, f_hat, theta[4])

        return -ll

    return nll


'''
prediction of new inquiry points
'''

''' Predictive mean function '''


#   tetha =  [l_1, s_1, l_2, s_2, nu]

def Pred_Student(kernel, X, y, f, theta, x_star):
    z = np.zeros_like(y) * 0.0;
    v1 = np.zeros_like(y) * 0.0
    n = X.shape[0]
    k = np.zeros(n) * 0.0

    # counter = 0
    for i in range(n):
        k[i] = kernel(x_star, X[i], theta[0:2])
        z[i] = (y[i] - f[i]) / np.exp(f[i + n])
        v1[i] = (1 + 1 / theta[4]) * z[i] / (np.exp(f[i + n]) * (1 + z[i] ** 2 / theta[4]))
        # counter += 1
        # print(v1[i], 'testing', counter)

    return k.dot(v1)


# TESTING!!!
#   X = np.random.uniform(1,2,25).reshape(-1,1)
#   y = np.sin( X )
#   f = np.append(y, y).ravel()
#   test = np.array([1.8]).reshape(-1,1)
#   theta = np.array([1,1,1,1,3])
#   testy = Pred_Student(kernel_1D,X,y,f,theta,test)
#   plt.scatter(X,y); plt.scatter(test,testy); plt.show()


def Pred_Student_vec(kernel, X, y, f, theta, X_star):
    m = X_star.shape[0]
    temp = np.zeros(m) * 0.0

    for j in range(m):
        temp[j] = Pred_Student(kernel, X, y, f, theta, X_star[j])

    return temp


''' END of function definitions '''






''' 
@@@@@@@@@@            MAIN  <<testing site>>
'''



''' Predictive covariance function '''

###### testing with actual data:
# usign the same example as in the begging to the code


x = np.random.uniform(-5, 5, 10).reshape(-1, 1)
y = 5 * np.sin(x) + 5

x_test = np.random.uniform(-5, 5, 5).reshape(-1, 1)
y_test = 5 * np.sin(x_test) + 5

plt.scatter(x, y, s=5)
plt.scatter(x_test, y_test, s=4)
plt.title("actual data -- Testing")
plt.xlabel("x-label")
plt.ylabel("y-label")
plt.show()

res = minimize(nll_fn(x, y), [1, 1, 1, 1, 5],
               bounds=((1e-3, None), (1e-3, None), (1e-3, None), (1e-3, None), (3, None)),
               method='L-BFGS-B')
# theta = np.array([1,1,1,1,5])
theta = res.x
print(theta)

#### Commputation of matrix K


K1 = kernel_RBF(x, x, theta[0:2])
K2 = kernel_RBF(x, x, theta[2:4])
K = block_diag(K1, K2)

# compute the mean for the approximation
f_hat = posterior_mode(X=x, y=y, K=K, nu = theta[4], max_iter=10)

'''   
the error here occurs when you increaase the number of iterations
'''




###
pred_mean = Pred_Student_vec(kernel_1D, x, y, f_hat, theta, x_test)

# plot test NG
plt.scatter(x, y, s=5)
plt.scatter(x_test, pred_mean, s=4)
# plt.scatter(x_test, f_hat[0:10], s=4)
plt.title("actual data -- Predicting")
plt.xlabel("x-label")
plt.ylabel("y-label")
plt.show()






