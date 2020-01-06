import numpy as np


#!begin
def gradient_descent(loss_grad, theta, lr=0.001, maxsteps=100,
                     mindiff=1e-7):
    """Gradient descent optimization method.

    Find an approximation of the vector theta that minimizes loss_fun.

    Parameters
    ----------
    loss_grad : callable
        function computing the gradient.
    theta : ndarray, shape (n,)
        initial value.
    lr : float, optional
        learning rate.
    maxsteps : int, optional
        maximum number of iterations.
    mindiff : float, optional
        stop when the change is below this value.

    Returns
    -------
     ndarray, shape (n,)
        the final estimate found by the algorithm.

    """
    for _ in range(maxsteps):
        diff = lr * loss_grad(theta)
        theta -= diff
        if np.linalg.norm(diff) < mindiff:
            break
    return theta
#!end


#!begin2
def gradient_descent_with_momentum(loss_grad, theta, lr=0.001,
                                   momentum=0.9, maxsteps=100):
    """Gradient descent with momentum.

    Find an approximation of the vector theta that minimizes an
    objective function.

    Parameters
    ----------
    loss_grad : callable
        function computing the gradient of the objective.
    theta : ndarray, shape (n,)
        initial value.
    lr : float, optional
        learning rate.
    momentum : float, optional
        momentum term.
    maxsteps : int, optional
        maximum number of iterations.

    Returns
    -------
    ndarray, shape (n,)
        the final estimate found by the algorithm.

    """
    v = theta.copy()
    for _ in range(maxsteps):
        delta = momentum * v + loss_grad(theta)
        theta -= lr * delta
    return theta
#!end2


def check_gradient(f, grad_f, x, eps=1e-5):
    y = f(x)
    g = grad_f(x)
    y1 = f(x + eps * g)
    d = y1 - y - eps * (g ** 2).sum()
    print(d)


def _test():
    """Test the method."""
    target = np.array((1, 2, 3))

    def f(x):
        return ((x - target) ** 2).sum()

    def df(x):
        return 2 * (x - target)

    xx = gradient_descent(df, np.zeros(3), 0.1)
    print(xx)
    xx = gradient_descent_with_momentum(df, np.zeros(3), 0.1)
    print(xx)


if __name__ == "__main__":
    _test()
