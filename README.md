# Investigating Global Dynamics in C. Elegans

The goal of this project is to investigate the global brain dynamics proposed by Kato et. al in [Global Brain Dynamics Embed the Motor
Command Sequence of C. elegans](https://www.sciencedirect.com/science/article/pii/S0092867415011964?via%3Dihub). 
To investigate whether these population-level findings are a byproduct of primary features of the data, we use the novel frameworks presented in "Structure in neural population recordings: an expected
byproduct of simpler phenomena?" by Gamaleldin Elsayed and John Cunningham, linked [here](http://stat.columbia.edu/~cunningham/pdf/ElsayedNN2017.pdf).

The presented methods develop a "neural population control" which we can use to test whether the population structure is an expected byproduct of primary features across time, neurons, and conditions.

The code for the TME (tensor maximum entropy) method, which generates a maximum-entropy distribution with specified marginal means and covariances, can be found [here](https://github.com/gamaleldin/rand_tensor).

We are investigating fitting a linear dynamical system model (Markov Chain with continuous latent variables) to the neural data, and using a Kalman Filter to recover the underlying latent variables.


