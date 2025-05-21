# InverseLaplaceAI

This is a side project aimed at exploring whether a dense neural network can solve the inverse Laplace problem for a noisy correlation function, similar to those used in Lattice QCD.

### Setup

The goal of this project is to use a dense neural network with one hidden layer to predict the overlap and energy parameters of a correlation function of the form:

$$
C(t) = \sum_{n=1}^N A_n e^{-E_n t},
$$

where we assume $A_n > 0$ and ascending energies, i.e., $E_0 < E_1 < \dots < E_N$. This type of correlation function frequently appears in Lattice QCD when the same operators are used at the source and sink, resulting in positive overlap factors. 

There are many well-established methods for determining the spectrum, such as matrix-Prony and multi-state fits. However, these methods often struggle to identify higher states in the spectrum. The ambitious goal of this project is to use a neural network to predict the parameters of the correlation function. A more realistic objective is to create a playground for understanding neural network concepts in a way that is intuitive for someone familiar with Lattice QCD.

Since, in practice, we lack prior information about the spectrum, I have chosen an unsupervised approach. The neural network is tasked with finding parameters that reproduce the input correlation function.

In `example.py`, an example calculation demonstrates the approach using a predefined correlation function. The noise in the correlation function is artificially generated using a noise model inspired by the Lepage argument, with Gaussian-distributed deviations from the mean correlator.

### Results

The results for `example.py`, obtained after 100 training epochs, are as follows:


Parameter | Expected | Predicted
----------|----------|-----------------------
A0        | 0.5      | 0.4569557 ± 0.0000061
A1        | 0.3      | 0.3021244 ± 0.0000097 
A2        | 0.2      | 0.2408266 ± 0.0000102
E0        | 0.6      | 0.5940353 ± 0.0000028 
E1        | 0.9      | 0.8352580 ± 0.0000121
E2        | 1.2      | 1.1914995 ± 0.0000205



### Interpretation of the Results

The chosen model does not appear to accurately calculate the parameters. However, it seems to provide parameter estimates that are in the general vicinity of the true values. This could make it useful for generating educated guesses for the parameters, which can then be refined as priors in multi-state fits.
