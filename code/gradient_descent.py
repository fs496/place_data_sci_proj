"""
Gradient descent for loss minimization for the problem L(b) = ||y - bx||^2 
where y and x are two vectors and b is a scalar.
"""
import numpy as np


def validate_inputs(x, y):
    assert x.shape == y.shape
    assert x.ndim == 1
    assert y.ndim == 1

def get_b(x, y):
    validate_inputs(x, y)
    return np.dot(x, y) / (np.linalg.norm(x)**2)


def get_loss(x, y, b):
    return np.linalg.norm(y - b * x) ** 2


def minimize_loss(x, y, b0, e, tol=1e-6, max_iters=1e6):
    validate_inputs(x, y)
    # Some validation of b0?
    
    L_current = get_loss(x, y, b0)
    diff_L = 1
    b_current = b0
    converged = False
    num_steps = 0
    for i in range(0, int(max_iters)):
        grad = -2 * np.dot(y - b_current * x, x)
        b_new = b_current - e * grad
        L_new = get_loss(x, y, b_new)
        
        diff_L = L_new - L_current
        diff_b = b_new - b_current

        L_current = L_new
        b_current = b_new
        
        if (np.abs(diff_L) < tol) & (diff_L <= 0):
            converged = True
            num_steps = i + 1
            break
    
    if num_steps == 0:
        num_steps = max_iters
    
    results = {
        'converged': converged,
        'b_min': b_current,
        'L_min': L_current,
        'num_steps': num_steps
    }
    return results 
        

if __name__ == '__main__':
    x = np.array([1, 1, 1])
    y = np.array([1, 1, 2])
    true_b = get_b(x, y)
    
    res = minimize_loss(x, y, b0=500, e=0.1)
