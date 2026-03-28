# Enclosure of the spectrum of linearized differential operators around localized solutions using a Gershgorin approach



Table of contents:


* [Introduction](#introduction)
* [The Swift Hohenberg equation](#the-swift-hohenberg-equation)
* [The capillary-gravity Whitham equations](#the-capillary-gravity-Whitham-equations)
* [The Gray-Scott system of equations](#the-Gray-Scott-system-of-equations)
* [Utilisation and References](#utilisation-and-references)
* [License and Citation](#license-and-citation)
* [Contact](#contact)



# Introduction

This Julia code is a complement to the article 

#### [[1]](https://arxiv.org/abs/2505.03091) : "Stability analysis for localized solutions in PDEs and nonlocal equations on $\mathbb{R}^m$" M. Cadiot [ArXiv Link](https://arxiv.org/abs/2505.03091)

in which we derive a Gershgorin-type of approach in the case of localized solutions. Specifically, we construct explicit disks containing the eigenvalues of the liearized operator at a given localized solution.

This code provides the necessary rigorous computations of the bounds presented along the paper. The computations are performed using the package [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl). The mathematical objects (spaces, sequences, operators,...) are built using the package [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl). Moreover, it relies on previously performed constructive existence proofs in the following papers :

#### [[2]](https://arxiv.org/abs/2403.10450) Stationary non-radial localized patterns in the planar Swift-Hohenberg PDE: Constructive proofs of existence, M. Cadiot, J.-P. Lessard, J.-C. Nave. Journal of Diﬀerential Equations, 414:555–608, 2025 
#### [[3]](https://arxiv.org/abs/2403.18718) Constructive proofs of existence and stability of solitary waves in the Whitham and capillary–gravity Whitham equations, M. Cadiot  Nonlinearity, 38(3):035021, 2025.
#### [[4]](https://arxiv.org/abs/2404.08529) The 2D Gray–Scott system of equations: constructive proofs of existence of localized stationary patterns, M. Cadiot and D. Blanco. Nonlinearity, 38(4):045016, 2025.


In [[1]](https://arxiv.org/abs/2505.03091), we provide the stability analysis of some of the  solutions which existence was proven in the above papers. This code provides the rigorous computations required for our analysis, and complements the results of Section 5 of [[1]](https://arxiv.org/abs/2505.03091). For this matter, we present below each application and the related computational details

# The Swift-Hohenberg equation

The Swift-Hohenberg equation
$$(I_d+\Delta)^2u +  \mu u + \nu_1 u^2 + \nu_2 u^3 =0$$
is known to have localized solutions on $\mathbb{R}^2$ that vanish at infinity. These solutions are called localized patterns (see [[paper]](https://arxiv.org/abs/2302.12877) for an introduction to the subject). 

In Section 5.1 of [[1]](https://arxiv.org/abs/2505.03091), we study the stability of an hexagonal and a square localized solution, which existence were obtained in [2]. In particular, we rely on the approximate solutions provided at https://github.com/matthieucadiot/LocalizedPatternSH.jl, which were used for the constructive existence proofs. The files stability_D4.jl (square solution) and stability_D6.jl (hexagonal solution) contain the numerical details for such rigorous computations, following the estimates provided in Section 5.1 of the paper.


# The capillary-gravity Whitham equations

The capillary-gravity Whitham equtation is a nonlocal equation modelling surface water waves. The equation possesses travelling waves solutions, which can be studied throughout the following equation
$$\mathbb{M}_Tu - cu + u^2 = 0,$$
where $c$ is the is the velocity of the wave and $\mathbb{M}_T$ is a Fourier multiplier operator given by its symbol
$$\mathcal{F}(\mathbb{M}_Tu)(\xi)  = \sqrt{\frac{\tanh(2\pi\xi)(1+T(2\pi\xi)^2)}{2\pi\xi}}\hat{u}(\xi).$$
$T \geq 0$ is the Bond number accounting for the surface tension. 

In Section 5.2 of [[1]](https://arxiv.org/abs/2505.03091), we study the linear stability of a solitary wave solution, which existence was obtained in [3]. In particular, we rely on the approximate solution provided at https://github.com/matthieucadiot/WhithamSoliton.jl, which was used for the constructive existence proofs. The file stability.jl contains the numerical details for such rigorous computations, following the estimates provided in Section 5.2 of the paper.


# The Gray-Scott system of equations

The Gray-Scott system of equations
$$\partial_tu_1 = \lambda_1 \Delta u_1 -  u_1 + (u_2 + 1 - \lambda_1 u_1)u_1^2 =0,   $$
$$\partial_tu_2 = \Delta u_2 - \lambda_2 u_2 + (\lambda_2 \lambda_1 - 1)u_1 = 0$$
is known to have stationary localized solutions on $\mathbb{R}^2$ that vanish at infinity. These solutions are called localized patterns (see [[paper]](https://arxiv.org/abs/2302.12877) for an introduction to the subject). 

In Section 5.3 of [[1]](https://arxiv.org/abs/2505.03091), we study the stability of a radial localized pattern, which existence was obtained in [4]. In particular, we rely on the approximate solution provided at https://github.com/dominicblanco/LocalizedPatternsGS.jl, which was used for the constructive existence proofs. The file stability.jl contains the numerical details for such rigorous computations, following the estimates provided in Section 5.3 of the paper.
 
 # Utilisation and References

Each folder contains all the necessary approximate solutions (in the form of JLD2 files) for the stability analysis. The approximate solutions correspond to the ones previously used for the existence proofs (in the aforementioned papers), as mentioned above. Then, each folder contains a list of functions, specific to each example, as well as a main file for performing the required rigorous computations.

Some of the rigorous computations, in particular the matrix products, are performed in Matlab, thanks to the MATLAB.jl package on Julia. In particular, they rely on the usage of [Intlab](https://www.tuhh.de/ti3/intlab/), for interval arithmetic computations, which is called inside Julia thanks to the file matproducts.jl. The user has to enter its personal path of Intlab in the matproducts.jl file. 
 
 The code is build using the following packages :
 - [RadiiPolynomial](https://github.com/OlivierHnt/RadiiPolynomial.jl) 
 - [IntervalArithmetic](https://github.com/JuliaIntervals/IntervalArithmetic.jl)
 - [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)
 - [JLD2](https://github.com/JuliaIO/JLD2.jl)
 - [MATLAB](https://github.com/JuliaInterop/MATLAB.jl)
 
 
 # License and Citation
 
This code is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).
  
If you wish to use this code in your publication, research, teaching, or other activities, please cite it using the following BibTeX template:

```
@software{Gershgorin-localized.jl,
  author = {Matthieu Cadiot},
  title  = {Gershgorin-localized.jl},
  url    = {https://github.com/matthieucadiot/Gershgorin-localized.jl},
  note = {\url{ https://github.com/matthieucadiot/Gershgorin-localized.jl},
  year   = {2025}
}
```


# Contact

You can contact me at :

matthieu.cadiot@polytechnique.edu


