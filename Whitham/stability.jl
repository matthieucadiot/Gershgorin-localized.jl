using  JLD2, RadiiPolynomial, IntervalArithmetic, LinearAlgebra 


include("list_of_functions.jl")
include("matproducts.jl")
####################################################################################################################################
######### This code provides the computer-assisted part of "Stability analysis for localized solutions in PDEs and nonlocal equations on $\R^m$" for the capillary-gravity Whitham equation. It relies on the code given at https://github.com/matthieucadiot/WhithamSoliton.jl and on the analysis established in the paper : "Constructive proofs of existence and stability of solitary waves in the Whitham and capillary-gravity Whitham equations"
#################################################################################################################################### 

################### Loading the candidate ##############################

U0 = load("W_05_08_40.jld2","U0") ## solution in Theorem 4.8 \tilde{u}_3
N = 800 ; N0 = 800
dn= 40 ; db = interval(big(dn)); d = interval(dn)
c = interval(0.8)
Tn = 0.5; T= abs(interval(Tn)) ; Tb = abs(interval(big(Tn))) 
### candidate values for a, σ0 and σ1. These values were obtained studying the graph of the function |mT-c| 
####values for T=0.5 and c=0.8


################ Choices of parameters for the proof 

δ0 = interval(0.16) ; δ0b = interval(big(0.16))
r0 = interval(8.7e-9) #### comes from Theorem 4.8 in Constructive proofs of existence and stability of
##solitary waves in the Whitham and capillary–gravity Whitham equations. Nonlinearity, 2025


### value of a and σ0 such that |l(z)-δ0| ≥ σ0 for all z such that |Im(z)| ≤ a. The value are validated using the aforementioned paper
a = interval(0.57);  ab = interval(big(a)) ; a = interval(a)
σ0 = interval(0.01)

# 𝒵u1, 𝒵u2 = computation_Zu(ExactReal(2)*U0,a,d,ab,db,δ0b)
#### precomputed quantities which we provide (comes from the code commented above)
𝒵u1 = interval(0.00017225)
𝒵u2 = interval(0.00017225)

U0 = interval.(Float64.(inf.(U0),RoundDown),Float64.(sup.(U0),RoundUp) )

# ###########################        PROOF OF STABILITY       ##################################################################

λmax = -σ0 + interval(2)*norm(U0,1) + r0/(4*sqrt(T)*σ0)
λmax = mid(λmax)

t = -λmax - 0.0001


############# cosine-cosine part ###################

D1 = convert(Vector{Interval{Float64}},exp2cos(N))
D2 = interval.(ones(N+1))./D1
SS = CosFourier(N, π/d)


ϵ,S = computation_bounds(U0,c,T,r0,δ0,σ0,t,d,SS,D1,D2,𝒵u1,𝒵u2)

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

display("upper bound S[3]")
display(S[3] + ϵ*abs(t+S[3]))
display("lower bound S[3]")
display(S[3] - ϵ*abs(t+S[3]))


########### verification that the first disk is disjoint from the rest of the disks. We compute an interval (-∞, Imax] containing the intersection of the rest of the disks  

Imax = interval(-10)
for n = 2:length(S)
    Imax = maximum([Imax S[n]+ϵ*abs(t+S[n])])
end 

if (sup(Imax)<inf(S[1]- ϵ*abs(t+S[1])))
    display("Case cosine : the first disk is disjoint from the rest of the disks")
    display("upper bound on the rest of the disks")
    display(sup(Imax))
else 
    display("Case cosine : the first disk is intersecting the rest of the disks")
end 


############# sine-sine part ###################

D1 = convert(Vector{Interval{Float64}},exp2sin(N))
D2 = interval.(ones(N))./D1
SS = SinFourier(N,π/d)

ϵ,S = computation_bounds(U0,c,T,r0,δ0,σ0,t,d,SS,D1,D2,𝒵u1,𝒵u2)

#### In this case we expect one eigenvalue around zero, the rest of the spectrum negative

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
    display("Case sine : the first disk is disjoint from the rest of the disks")
    display("upper bound on the rest of the disks")
    display(sup(Imax))
else 
    display("Case sine : the first disk is intersecting the rest of the disks")
end 
