import numpy as np

def noise_model(T, mN, mpi):
    """
    Computes the noise to signal ratio for a given time series length T, inspired by the Lepage argument.
    
    Parameters
    ----------
    T : array-like
        Time series.
    mN : float
        Mass of the hypothetical nucleon.
    mpi : float
        Mass of the hypothetical pion.

    Returns
    -------
    array-like
        The noise to signal ratio.
    """
    signal2noise = np.exp(-(mN - 3/2 * mpi)*T)
    return 1/signal2noise


def correlation_function(T, energies, overlaps):
    """
    Computes the correlation function for a given time series T, energies, and overlaps.
    
    Parameters
    ----------
    T : array-like
        Time series.
    energies : array-like
        Energies of the states.
    overlaps : array-like
        Overlaps of the states.

    Returns
    -------
    array-like
        The correlation function.
    """
    if len(energies) != len(overlaps):
        raise ValueError("Energies and overlaps must have the same length.")
    
    Es = energies.reshape(-1, 1)
    As = overlaps.reshape(-1, 1)
    T = T.reshape(1, -1)

    C = np.sum(As * np.exp(-Es * T), axis=0)
    C = C.reshape(-1)
    return C


def generate_fake_data(T, Nc, energies, overlaps, noise_factor=0.01, seed=12, mpi=0.13):
    """
    Generates fake data for a given time series T, number of configurations Nc, energies, overlaps, and noise factor.
    
    Parameters
    ----------
    T : array-like
        Time series.
    Nc : int
        Number of configurations.
    energies : array-like
        Energies of the states.
    overlaps : array-like
        Overlaps of the states.
    noise_factor : float, optional
        Factor to scale the noise, by default 0.01
    seed : int, optional
        Random seed for reproducibility, by default 12
    mpi : float, optional
        Mass of the hypothetical pion, by default 0.13
    
    Returns
    -------
    array-like shape (Nc, len(T))
        The generated fake data.
    """

    rng = np.random.default_rng(seed)
    
    C = correlation_function(T, energies, overlaps)
    Cstd = noise_factor * noise_model(T, energies[0], mpi) * C
    
    data = np.zeros((Nc, len(T)))
    for i in range(Nc):
        data[i,:] = C + rng.normal(0, Cstd)

    return data



