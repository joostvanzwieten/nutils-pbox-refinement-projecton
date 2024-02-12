import numpy as np


def EqualSpaced(n_steps, m_steps):
    return np.linspace(10**(1-n_steps),1,m_steps)

def ExpSpaced(n_steps, m_steps):
    return np.logspace(1 - n_steps, 0, (m_steps) * (n_steps - 1) + 1)

def DecrStepSpacing(n_steps, m_steps):
    recipr = np.cumsum(1 / np.linspace(1, m_steps, m_steps) ** 2)
    recipr = np.insert(recipr, 0, 0, axis=0)

    alpha = (10 ** (n_steps - 1)) ** (1 / recipr[-1])
    return 0.1 ** (n_steps - 1) * alpha ** (recipr)