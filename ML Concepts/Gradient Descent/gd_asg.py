# This code is edited by ABHIJEET PARASHAR
# 1 Crore - iterations
# Don't Run the programme.
# Don't Run the programme.
# Don't Run the programme.
# Don't Run the programme.
# Don't Run the programme.

import numpy as np 
import pandas as pd 
import math

def g_d(x,y) :
    m_c = b_c = 0   # Current Intercept and Slope
    itr = 10000000    # Iterations
    n = len(x)      # value of n
    l_r = 0.0001     # Learning Rate
    m_c_prev = b_c_prev = float('inf')

    for i in range(itr) :
        y_p = m_c * x + b_c
        cost = (1/n) * sum((y - y_p)**2)    # MSE
        md = -(2/n) * sum(x*(y - y_p))      # pde wrt m
        bd = -(2/n) * sum(y - y_p)          # pde wrt b
        m_c = m_c - l_r * md                # Learning rate
        b_c = b_c - l_r * bd                # Learning rate
        print("m{} , b{} , cost{}, itr{}".format(m_c , b_c ,cost, i))
        if math.isclose(m_c, m_c_prev, rel_tol=1e-20, abs_tol=0.0) and math.isclose(b_c, b_c_prev, rel_tol=1e-20, abs_tol=0.0) :
            break
        m_c_prev = m_c
        b_c_prev = b_c
        


df = pd.read_csv(r"C:\Users\abhij\Desktop\Skills\ML\ML Concepts\Gradient Descent\test_scores.csv")
x = np.array(df['math'])
x = x.astype(float)
y = np.array(df['cs'])
y = y.astype(float)
g_d(x,y)

# m = 1.0177362378570707
# b = 1.9152193111471227