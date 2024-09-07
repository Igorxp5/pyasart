import numpy as np

def init_adam_optimizer(input_shape):
    m = np.zeros(input_shape)
    v = np.zeros(input_shape)
    return m, v


def step_adam_optimizer(m, v, grad, diff_step, epoch, learning_rate=0.01,
                        beta1=0.9, beta2=0.009, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1**(epoch + 1))
    v_hat = v / (1 - beta2**(epoch + 1))
    diff_step -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return diff_step, (m, v)
