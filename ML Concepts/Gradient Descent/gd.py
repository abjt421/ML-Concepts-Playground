# This code is edited by ABHIJEET PARASHAR
# Gradient Descent

import numpy as np 

def g_d(x,y) :
    m_c = b_c = 0   # Current Intercept and Slope
    itr = 100000    # Iterations
    n = len(x)      # value of n
    l_r = 0.079     # Learning Rate

    for i in range(itr) :
        y_p = m_c * x + b_c
        cost = (1/n) * sum((y - y_p)**2)    # MSE
        md = -(2/n) * sum(x*(y - y_p))      # pde wrt m
        bd = -(2/n) * sum(y - y_p)          # pde wrt b
        m_c = m_c - l_r * md                # Learning rate
        b_c = b_c - l_r * bd                # Learning rate
        print("m{} , b{} , cost{}, itr{}".format(m_c , b_c ,cost, i))

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
g_d(x,y)