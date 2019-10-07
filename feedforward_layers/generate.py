import numpy as np
import random
import math

#return X, Y
def generate_data():
    X_list = []
    Y_list = []
    for i in range(100):
        a = random.uniform(0,1)-0.5
        b = random.uniform(0,1)-0.5
        c = a**2
        d = b**2
        e = a*b + c
        f = b**4
        g = math.exp(a)
        X_list.append([a,b,c,d,e,f,g])
        y_ = math.exp(a**4 * b**2) * pow(c,d) - math.cos(f*g)
        if y_ >= 0:
            y = y_
        else:
            y = y_ + 10  
        Y_list.append(y)
    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y 
        
    
