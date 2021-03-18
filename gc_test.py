
import numpy as np
from numpy.linalg import inv

X= np.random.rand(10,4)
def create_time_series(x,y,delay):

    # x = x.transpose()
    time_series_length = x.shape[0]
    time_series_range = np.arange(delay, time_series_length )
    X = np.zeros([time_series_range.shape[0],(x.shape[1]*delay)])
    for d in np.arange(1, delay +1):
        X [:,x.shape[1]*(d-1):x.shape[1]*d]= x[time_series_range-d,:]
            
    Y = y[time_series_range ]  
    
    return X , Y 


def my_gc(x,y,delay):
    return create_time_series(x,y,delay)

x1,y1= my_gc(X[0:10,0:4],X[0:10,0],3)

# def linear_granger_causality(X,delay):
    
#     if X.shape[0] < X.shape[1]:
#         X = X.transpose()
        
#     for c1 in np.arange(0, X.shape[1]):
#         for c2 in np.arange(c1+1, X.shape[1]):
#             x = X[:,[c1,c2]]
#             y = X[:,c1]
#             # [Input , Target] = linear_granger_causality.create_time_series(x,y,delay)
#             [Input , Target] = create_time_series(x,y,delay)
#             betta_coef = np.matmul(inv(np.matmul(Input.transpose(),
#                          Input)),np.matmul(Input.transpose(),Target))
            
            