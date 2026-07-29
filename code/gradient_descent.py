"""
Gradient descent for loss minimization for the problem L(b) = ||y - bx||^2 
where y and x are two vectors and b is a scalar.
"""
import numpy as np


def validate_inputs(x: np.ndarray, y: np.ndarray):
    """
    Validate the type and dimensions of the vectors x and y
    as inputs in the loss function L(b) = ||y - bx||^2
    
    Parameters
    ----------
    x: np.ndarray
        Vector x in the loss function L(b)
    y: np.ndarray
        Vector y in the loss function L(b)
    """
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.shape == y.shape
    assert x.ndim == 1
    assert y.ndim == 1


def get_b(x: np.ndarray, y: np.ndarray) -> np.float64:
    """
    Calculated the true value of b that minimizes the loss
    L(b) = ||y - bx||^2

    Parameters
    ----------
    x: np.ndarray
        Vector x in the loss function L(b)
    y: np.ndarray
        Vector y in the loss function L(b)
    

    Returns
    -------
    np.float64
        The true value of b that minimizes the loss L(b)
    """
    validate_inputs(x, y)
    return np.dot(x, y) / (np.linalg.norm(x)**2)


def get_loss(x: np.ndarray, y: np.ndarray, b: np.float64) -> np.float64:
    """
    Calculate L(b) = ||y - bx||^2.
    
    Parameters
    ----------
    x: np.ndarray
        Vector x in the loss function L(b)
    y: np.ndarray
        Vector y in the loss function L(b)
    b: np.float64
        Scalar b in the loss function L(b)

    Returns
    -------
    np.float64
        Value of L(b)
    """
    return np.linalg.norm(y - b * x) ** 2


def minimize_loss(x: np.ndarray, y: np.ndarray, b0: float, e: float,
                  tol: float=1e-6, max_iters: int=1e6) -> dict:
    """
    Minimize the loss function L(b) = ||y - bx||^2 using gradient
    descent.

    Parameters
    ----------
    x: np.ndarray
        Vector x in the loss function L(b)
    y: np.ndarray
        Vector y in the loss function L(b)
    b0: float
        Initial guess for the value of b
    e: float
        Learning rate or step size in the gradient descent algorithm
    tol: float, optional
        Stopping criteria for gradient descent. If the absolute change in L(b)
        in the last step is less than this tolerance, the algorithm is declared
        to converge and ends. The default is 1e-6.
    max_iters: int, optional
        Maximum allowed iterations in gradient descent. The default is 1e6.

    Returns
    -------
    dict
        A dictionary containing the results of the gradient descent
        minimization.
    """
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
    y = np.array([1, 1, 2]) * 1000000
    b_true = get_b(x, y)
    res = minimize_loss(x, y, b0=500, e=0.1)
