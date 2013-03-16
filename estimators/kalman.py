import numpy as np
import time

class kalman(object):
    
    A = None
    B = None
    C = None
    D = None
    
    Q_k = None
    R_k = None
    
    P = None
    
    x_hat = None

    def __init__(self, A=np.matrix(np.eye(2)), B=np.matrix(np.eye(2)), C=np.matrix(np.eye(2)), D=np.matrix(np.zeros(2)), Q_k=np.matrix(np.eye(2)), R_k=np.matrix(np.eye(2)), x0=np.matrix([[0],[0]])):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
        self.Q_k = Q_k
        self.R_k = R_k
        self.P = self.Q_k
        
        self.x_hat = x0
    
    def update(self, y, u):

        x_est = self.x_hat
        P = self.P

        # (a priori) Predict the next state and covariance values
        x_est = self.A * x_est + self.B * u
        # covariance
        self.P = self.A * self.P * self.A.T + self.Q_k

        # Compute Kalman Gain
        self.K = self.P * self.C.T * np.linalg.inv(self.C * self.P * self.C.T + self.R_k)

        # (a posteriori) Update predictions
        self.x_adj = self.K * (y - (self.C * x_est + self.D * u)) # Calulate the measurement based adjustement
        self.x_hat = x_est + self.x_adj                 # Combine the predicted state with the measurement adjustment
        self.P = (np.matrix(np.eye(2)) - self.K * self.C) * self.P              # Update the coviance matrix
        
        return self.x_hat
