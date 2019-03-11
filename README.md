FastKPM
=======

What it does
------------

This package implements [_gradient-based
probing_](https://arxiv.org/abs/1711.10570) for fast electronic structure
calculations. It is written in C++, and supports CUDA, multi-threading, and MPI
acceleration. Running on one or more GPUs is _highly_ recommended. A modern GPU
will typically accelerate this code by 100x.

Given a sparse Hamiltonian matrix `H` in a real-space, orthogonal basis, the
goal of FastKPM is to calculate elements of the density matrix `D` at a
computational cost that scales linearly with system size `N`. The density matrix
is defined as `D = f(H)` where `f` is the Fermi function. Although `D` might be
dense, typical applications require only a sparse set of matrix elements `D_ij`.

The [Kernel Polynomial Method](https://arxiv.org/abs/cond-mat/0504627) (KPM)
achieves linear scaling by employing two tricks: (1) Expanding the Fermi
function `f` up to some order `M` of Chebyshev polynomials, and (2) Using a
stochastic approximation to estimate `D`. This approximation is _unbiased_, and
can be controlled by a tuneable parameter `S`. The total computational cost to
estimate density matrix elements scales as `O(N M S)`.

Saad and collaborators introduced a [probing
technique](https://doi.org/10.1002/nla.779) to significantly reduce the
stochastic approximation error as a function of `S`. Their approach directly
leverages the fact that the density matrix elements `D_ij` decay with spatial
distance between orbitals `i` and `j` in a real-space basis. Our [gradient-based
probing](https://arxiv.org/abs/1711.10570) builds on prior work to achieve even
faster convergence.

The hardest system to simulate is a zero temperature metal, because the
electronic wavefunctions decay only polynomially. In this case, the
gradient-based probing error scales as `S^{-(d+2)/(2d)}`, where `d` is the
spatial dimension. This is much faster than in the original KPM method, for
which the stochastic error scales like `sqrt(S)`.

Building
--------

Building is handled with CMake.

CMake requires the following libraries: `Armadillo`, `Boost > 1.55`.

CMake will use the following libraries, if it can find them: `fftw`, `CUDA`, `TBB`, `MPI`.

To tell CMake where it can find a dependency, use the command `cmake -D CMAKE_PREFIX_PATH=...`

Usage
-----

CMake will compile a `fastkpm` library, which can then be linked into binaries. For usage examples, please see the [Kondo](https://github.com/kbarros/kondo) package.

Citing
------

If you find FastKPM useful, please cite this paper:

```
@article{doi:10.1063/1.5017741,
author = {Wang, Zhentao and Chern, Gia-Wei and Batista, Cristian D. and Barros, Kipton},
title = {Gradient-based stochastic estimation of the density matrix},
journal = {J. Chem. Phys.},
volume = {148},
pages = {094107},
year = {2018},
}
```
