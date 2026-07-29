"""
Gradient descent for loss minimization for the problem L(b) = ||y - bx||^2 
where y and x are two vectors and b is a scalar.
"""
import numpy as np
import warnings
from tqdm import tqdm
import pandas as pd
import itertools

SAVE_FOLDER = "(TOP FOLDER REMOVED FOR PRIVACY)"


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


def get_param_diff(param_current, param_new, rel=True):
    if rel:
        param_diff = (param_new - param_current) / (1 + np.abs(param_current))
    else:
        param_diff = param_new - param_current
    return param_diff


def minimize_loss(x: np.ndarray, y: np.ndarray, b0: float, e: float,
                  tol: float=1e-6, rel: bool=True, stop_on: str='loss',
                  max_iters: int=1e6) -> dict:
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
        Tolerance for the stopping criteria for gradient descent. The default
        is 1e-6.
    rel: bool, optional
        Whether the tolerance for the stopping criteria should be applied on
        the relative difference of the stopping parameter (see stop_on).
    stop_on: str, optional
        Stopping method used for gradient descent. The accepted options are:
            'loss': Stop when the absolute or relative difference (see rel)
                in the loss function is smaller than the tolerance in absolute
                value.
            'step': Stop when the absolute or relative difference (see rel)
                in the step size is smaller than the tolerance in absolute
                value.
            'grad': Stop when the absolute value of the gradient is less than
                the tolerance. rel is ignored when using this method.
    max_iters: int, optional
        Maximum allowed iterations in gradient descent. The default is 1e6.

    Returns
    -------
    dict
        A dictionary containing the results of the gradient descent
        minimization.
    """
    # Validation
    validate_inputs(x, y)
    assert e > 0
    assert tol > 0
    assert max_iters > 0
    # Some validation of b0?
    
    # Initialize key values
    b_current = b0
    L_current = get_loss(x, y, b0)
    converged = False
    num_steps = 0

    # Set these temporarily so that we can define some functions on them
    grad = 0
    b_new = 0
    L_new = 0
    
    # Set stopping criteria
    # Would be better to refactor this as a class and set these
    # as instance attributes - but do it this way to save time
    if stop_on == 'loss':
        stop_param_current = lambda: L_current
        stop_param_new = lambda: L_new
    elif stop_on == 'step':
        stop_param_current = lambda: b_current
        stop_param_new = lambda: b_new
    else:
        assert stop_on == 'grad', f'Unrecognized value of stop_on: {stop_on}'
    
    grad_path = [[np.nan, b_current, L_current]]
    for i in range(int(max_iters)):
        # Execute a step of gradient descent
        grad = -2 * np.dot(y - b_current * x, x)  # L'(b_current)
        b_new = b_current - e * grad
        L_new = get_loss(x, y, b_new)
        grad_path.append([grad, b_new, L_new])
        
        if np.isinf(grad):
            warnings.warn("Gradient is infinitely large")
        
        # Determine whether we have converged
        if stop_on in ['loss', 'step']:
            param_diff = get_param_diff(
                stop_param_current(), stop_param_new(), rel=rel
            )
            if (np.abs(param_diff)) < tol:
                converged = True
                num_steps = i + 1
                break
        else:
            if np.abs(grad) < tol:
                converged = True
                num_steps = i + 1
                break

        L_current = L_new
        b_current = b_new

    num_steps = max_iters if num_steps == 0 else num_steps

    results = {
        'converged': converged,
        'b_min': b_new,
        'L_min': L_new,
        'num_steps': num_steps,
        'grad_path': np.array(grad_path)
    }
    return results


if __name__ == '__main__':    
    # Generate random x's and y's from a normal distribution - chose
    # to use a normal distribution rather than uniform to avoid extreme
    # values that blow up the gradient
    vec_len = 2  # Testing vectors of length 2 for simplicity
    num_trials = 100
    rng = np.random.default_rng()
    trials = [
        {
            'id': i,
            'x': rng.normal(loc=0, scale=1, size=(vec_len,)),
            'y': rng.normal(loc=0, scale=1, size=(vec_len,)),
        } for i in range(num_trials)
    ]

    # Calculate true bs and generate random b0s
    for i in range(num_trials):
        trials[i]['b_true'] = get_b(trials[i]['x'], trials[i]['y'])
        trials[i]['b0'] = rng.normal(loc=0, scale=1)

    # Run gradient descent on the x, y, b0 values for a range of e values
    e_vals = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5, 1]
    save_data = []
    rel = True
    stop_on = 'loss'
    for e, trial in tqdm(itertools.product(e_vals, trials)):
        result = minimize_loss(
            x=trial['x'],
            y=trial['y'],
            b0=trial['b0'],
            e=e,
            rel=rel,
            stop_on=stop_on
        )
        data = {**result, **trial}
        data['e'] = e
        data['error_b'] = (
            np.abs((data['b_min'] - data['b_true']) / data['b_true'])
        )
        save_data.append(data)
    
    # Save this result as a pickled dataframe
    # Saved outside repository because it is too large to push to GitHub
    df = pd.DataFrame(save_data)
    df.to_pickle(
        f"{SAVE_FOLDER}/grad_desc_rel_{rel}_stop_on_{stop_on}.pkl"
    )

    # Plot performance against: e, ||x||, ||y||, ||x||/||y||,
    # angle between x and y, ratio of b0/b_true,
    # Combinations of e and the other input params
    # Repeat for different stopping criteria if possible - start with
    # stop_on = loss, rel=True
    # Repeat for different vector lengths?
