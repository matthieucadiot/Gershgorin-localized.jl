using MATLAB, MAT, JLD2, RadiiPolynomial, IntervalArithmetic, LinearAlgebra

include("list_functions.jl")
include("matproducts.jl")
####################################################################################################################################
######### This code provides the computer-assisted part of "Stability analysis for localized solutions in PDEs and nonlocal equations on $\R^m$" for the Swift-Hohenberg application. It relies on the code given at https://github.com/matthieucadiot/LocalizedPatternSH.jl and on the analysis established in the paper : "Stationary non-radial localized patterns in the planar Swift-Hohenberg PDE: constructive proofs of existence"
#################################################################################################################################### 

### We start with the "square" solution 
#### parameters of the solution under study
 N0 = 130 ;  d=interval(70.0) ; N = 80; 
 precision_param = 80 ;        setprecision(precision_param)
 μ = interval(0.28) ;  ν1 = interval(-1.6) ; ν2 = interval(1.0) 
μbig = interval(big(0.28)) ; v1big = interval(big(-1.6)) ; v2big =interval(big(1.0)) 
dbig = interval(big(70.0)) 

### definition of the space of the solution
fourier0 = CosFourier(N0, π/d)⊗CosFourier(N0, π/d)
fourier = CosFourier(N, π/d)⊗CosFourier(N, π/d)

### loading the solution U0 and the square U2 = U0*U0 already computed in multiprecision. This data is available at https://github.com/matthieucadiot/LocalizedPatternSH.jl
 U0 = load("Wc_028_70_130_projected.jld2","U")
 U2 = load("Wc_028_70_130_square.jld2","U2")

# We build V0 in big float precision
U0 = Sequence(fourier0,U0)
U2 = Sequence(CosFourier(2N0, π/d)⊗CosFourier(2N0, π/d),U2)
V0 = ExactReal(2)*v1big*U0 + ExactReal(3)*v2big*U2

# Conversion into usual Float64 precision
U0 = interval.(Float64.(inf.(U0),RoundDown),Float64.(sup.(U0),RoundUp) )
U2 = interval.(Float64.(inf.(U2),RoundDown),Float64.(sup.(U2),RoundUp) )

################ Choices of parameters for the proof 
δ0 = ExactReal(0.01) ; δ0big = interval(big(0.01))
r0 = interval(1.16e-5) #### comes from Theorem 4.1 in "Stationary non-radial localized patterns in the planar
##Swift-Hohenberg PDE": Constructive proofs of existence. Journal of Differential Equations,
##414:555–608, 2025.

########### Computation of the quantities which can be reused in all cases 

# C0 = interval(1.335)/(sqrt(μ-δ0))
# N1 = 150 ; V0_p = project(V0,CosFourier(N1, π/d)⊗CosFourier(N1, π/d))

# 𝒵u1, 𝒵u2 = computation_Zu(V0_p,N1,C0)
# 𝒵u1 = 𝒵u1 +   (1/(μ-δ0)*norm(V0_p-V0,1)) ; 𝒵u2 = 𝒵u2 +   (1/(μ-δ0)*norm(V0_p-V0,1))

V0 = interval.(Float64.(inf.(V0),RoundDown),Float64.(sup.(V0),RoundUp) )
##### We have already computed the values for 𝒵u1 and 𝒵u2 using the code commented above. 
𝒵u1 = interval(0.0019246)
𝒵u2 = interval(0.001925)


## computation of the 2 norm of 1/l to compute λmax
norml = sqrt( (ExactReal(2)*sqrt(μ) + (ExactReal(1)+ μ)*(ExactReal(2)*π - ExactReal(2)*atan(sqrt(μ))))/(ExactReal(8)*(1+μ)*μ^(interval(3/2))) + ExactReal(2)*(π^2/d)*(interval(3)^(interval(3/4))/(μ^(interval(7/4))) + interval(3)/(μ^(interval(5/2)))) )

λmax = norm(V0,1) + ExactReal(2)*abs(ν1)*norml*r0 + ExactReal(3)*abs(ν2)*norml*r0*(ExactReal(2)*norm(U0,1) + norml*r0) - μ

## we choose t slighlty smaller than λmax
t = - λmax - interval(0.00001)
# display("value of t")
# display(t)

##### We run the program below with t= - λmax - interval(0.00001) given as above. We obtain intersecting Gershgorin disks and cannot conclude about the localization of the first eigenvalues. However, we obtain that the biggest eigenvalue is smaller than 0.0798 and consequently we can choose t = -0.11 for instance. We re-run the program a second time with t = -0.11 this time, and obtain the enclosures given in the paper.

t = interval(-0.11)

################# Case cc : cosine-cosine expansion ###########

### computation of the weights allowing to switch from exponential to cosine series (useful for norm computations)
D1 = convert(Vector{Interval{Float64}},exp2cos(N))
D2 = interval.(ones((N+1)^2))./D1

SS = CosFourier(N, π/d)⊗CosFourier(N,π/d)
ϵ,S = computation_bounds(U0,V0,r0,μ,δ0,t,d,ν1,ν2,SS,D1,D2,𝒵u1,𝒵u2)


#### In this case we expect one positive eigenvalue, the rest of the spectrum negative

display("cosine-cosine part")
display("upper bound S[1]")
display(S[1] + ϵ*abs(t+S[1]))
display("lower bound S[1]")
display(S[1] - ϵ*abs(t+S[1]))

display("upper bound S[2]")
display(S[2] + ϵ*abs(t+S[2]))
display("lower bound S[2]")
display(S[2] - ϵ*abs(t+S[2]))

########### verification that the first  disk is disjoint from the rest of the disks. We compute an interval (-∞, Imax] containing the intersection of the rest of the disks  

Imax = interval(-10)
for n = 2:length(S)
    Imax = maximum([Imax S[n]+ϵ*abs(t+S[n])])
end 

if (sup(Imax)<inf(S[1]- ϵ*abs(t+S[1])))
    display("Case cosine-cosine : the first disk is disjoint from the rest of the disks")
    display("upper bound on the rest of the disks")
    display(sup(Imax))
else 
    display("Case cosine-cosine : the first disk is intersecting the rest of the disks")
end 

# ################# Case cs : cosine-sine expansion ###########

D1 = convert(Vector{Interval{Float64}},exp2cos_sin(N))
D2 = interval.(ones((N+1)*N))./D1
SS = CosFourier(N,π/d)⊗SinFourier(N,π/d)

ϵ,S = computation_bounds(U0,V0,r0,μ,δ0,t,d,ν1,ν2,SS,D1,D2,𝒵u1,𝒵u2)


#### In this case we expect one positive eigenvalue, one eigenvalue around zero and the rest of the spectrum negative

display("cosine-sine part")
display("upper bound S[1]")
display(S[1] + ϵ*abs(t+S[1]))
display("lower bound S[1]")
display(S[1] - ϵ*abs(t+S[1]))

display("upper bound S[2]")
display(S[2] + ϵ*abs(t+S[2]))
display("lower bound S[2]")
display(S[2] - ϵ*abs(t+S[2]))

display("upper bound S[3]")
display(S[3] + ϵ*abs(t+S[3]))
display("lower bound S[3]")
display(S[3] - ϵ*abs(t+S[3]))


########### verification that the first two disks are disjoint from the rest of the disks. We compute an interval (-∞, Imax] containing the intersection of the rest of the disks  

Imax = interval(-10)
for n = 3:length(S)
    Imax = maximum([Imax S[n]+ϵ*abs(t+S[n])])
end 

if (sup(Imax)<inf(S[1]- ϵ*abs(t+S[1])))&&(sup(Imax)<inf(S[1]- ϵ*abs(t+S[1])))
    display("Case cosine-sine : the first two disks are disjoint from the rest of the disks")
    display("upper bound on the rest of the disks")
    display(sup(Imax))
else 
    display("Case cosine-sine : the first disk is intersecting the rest of the disks")
end 


# # ################# Case sc : sine-cosine expansion ###########

# # # D1 = convert(Vector{Interval{Float64}},exp2sin_cos(N))
# # # D2 = ones((N+1)*N)./D1
# # # SS = SinFourier(N, π/d)⊗CosFourier(N,π/d)

# # # ϵ,S = computation_bounds(U0,V0,r0,μ,δ0,t,d,ν1,ν2,SS,D1,D2,𝒵u1,𝒵u2)

# # ##### In this case we expect one positive eigenvalue, one eigenvalue around zero and the rest of the spectrum negative

# # #display("sine-cosine part")
# # # display("upper bound S[1]")
# # # display(S[1] + ϵ*abs(t+S[1]))
# # # display("lower bound S[1]")
# # # display(S[1] - ϵ*abs(t+S[1]))

# # # display("upper bound S[2]")
# # # display(S[2] + ϵ*abs(t+S[2]))
# # # display("lower bound S[2]")
# # # display(S[2] - ϵ*abs(t+S[2]))

# # # display("upper bound S[3]")
# # # display(S[3] + ϵ*abs(t+S[3]))
# # # display("lower bound S[3]")
# # # display(S[3] - ϵ*abs(t+S[3]))


########### verification that the first two disks are disjoint from the rest of the disks. We compute an interval (-∞, Imax] containing the intersection of the rest of the disks  

Imax = interval(-10)
for n = 3:length(S)
    Imax = maximum([Imax S[n]+ϵ*abs(t+S[n])])
end 

if (sup(Imax)<inf(S[1]- ϵ*abs(t+S[1])))&&(sup(Imax)<inf(S[1]- ϵ*abs(t+S[1])))
    display("Case sine-cosine : the first two disks are disjoint from the rest of the disks")
    display("upper bound on the rest of the disks")
    display(sup(Imax))
else 
    display("Case sine-cosine : the first disk is intersecting the rest of the disks")
end 



# # ################# Case ss : sine-sine expansion ###########

D1 = ExactReal(2)*interval.(ones(N^2)) ; D2 = ExactReal(0.5)*interval.(ones(N^2))
SS = SinFourier(N,π/d)⊗SinFourier(N,π/d)

ϵ,S = computation_bounds(U0,V0,r0,μ,δ0,t,d,ν1,ν2,SS,D1,D2,𝒵u1,𝒵u2)

#### In this case we expect one eigenvalue around zero and the rest of the spectrum negative
display("sine-sine part")
display("upper bound S[1]")
display(S[1] + ϵ*abs(t+S[1]))
display("lower bound S[1]")
display(S[1] - ϵ*abs(t+S[1]))

display("upper bound S[2]")
display(S[2] + ϵ*abs(t+S[2]))
display("lower bound S[2]")
display(S[2] - ϵ*abs(t+S[2]))


########### verification that the first disk is disjoint from the rest of the disks. We compute an interval (-∞, Imax] containing the intersection of the rest of the disks  

Imax = interval(-10)
for n = 2:length(S)
    Imax = maximum([Imax S[n]+ϵ*abs(t+S[n])])
end 

if (sup(Imax)<inf(S[1]- ϵ*abs(t+S[1])))
    display("Case sine-sine : the first disk is disjoint from the rest of the disks")
    display("upper bound on the rest of the disks")
    display(sup(Imax))
else 
    display("Case sine-sine : the first disk is intersecting the rest of the disks")
end 




#### verification that the solution is not radially symmetric in order to apply Lemma 5.1 in Section 5. We use that \|u\|∞ ≤ norml/μ \|u\|_{\mathcal{H}}

θ = interval(π)/interval(6)
x = cos(θ) ; y = sin(θ)

diff_u = abs(U0(x,y) - U0(interval(1),interval(0))) - interval(2)*norml/μ*r0 

if inf(diff_u) > 0
    display("the solution is not radially symmetric")
else
    display("could not conclude about non-radially symmetric. Try changing the value of θ")
end


