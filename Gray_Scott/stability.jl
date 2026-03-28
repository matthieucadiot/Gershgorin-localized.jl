using RadiiPolynomial, IntervalArithmetic, LinearAlgebra, JLD2

# Needed additional sequence structures for RadiiPolynomial (see Section 6)
# You can download this file from Github and put it in the same folder as this one. Then, it will include automatically.
include("D4Fourier.jl")
include("list_functions.jl")
include("matproducts.jl")


U01 = load("U01.jld2","U01") #Spike solution first component
U02 = load("U02.jld2","U02") #Spike solution second component
U01_D4 = load("U01_D4.jld2","U01") #Spike solution first component with the D4 structure
U02_D4 = load("U02_D4.jld2","U02") #Spike solution second component with the D4 structure
N0 = 50   # number of Fourier modes for the ring
N = 50    # number of Fourier modes for operators for the ring
setprecision(80)
d = 8 ; d = interval(d) ; dbig = interval(big(d))   # size of the domain for the spike away from λ₁λ₂ = 1
λ2 = 10 ; λ2 = interval(λ2) ; λ2big = interval(big(λ2)) # values of parameters for the spike away from λ₁λ₂ = 1
λ1 = 1/9 ; λ1 = interval(λ1) ; λ1big = interval(big(λ1))
r0 = interval(6e-6) # value for the proof of existence

# V1 = big(2)*U01*U02   + big(2)*U01 - big(3)*λ1big*U01*U01
# V2 = U01*U01
# V1_D4 = big(2)*U01_D4*U02_D4   + big(2)*U01_D4 - big(3)*λ1big*U01_D4*U01_D4
# V2_D4 = U01_D4*U01_D4
V1 = load("V1.jld2","V1") # loading the already computed functions V1
V2 = load("V2.jld2","V2") # loading the already computed functions V2
V1_D4 = load("V1_D4.jld2","V1_D4") # loading the already computed functions V1_D4
V2_D4 = load("V2_D4.jld2","V2_D4") # loading the already computed functions V2_D4


U01 = interval.(Float64.(inf.(U01),RoundDown),Float64.(sup.(U01),RoundUp) )
U02 = interval.(Float64.(inf.(U02),RoundDown),Float64.(sup.(U02),RoundUp) )
U01_D4 = interval.(Float64.(inf.(U01_D4),RoundDown),Float64.(sup.(U01_D4),RoundUp) )
U02_D4 = interval.(Float64.(inf.(U02_D4),RoundDown),Float64.(sup.(U02_D4),RoundUp) )

U01 = Sequence(CosFourier(N0, mid(π/d))⊗CosFourier(N0, mid(π/d)),coefficients(mid.(U01)))
U02 = Sequence(CosFourier(N0, mid(π/d))⊗CosFourier(N0, mid(π/d)),coefficients(mid.(U02)))
U01_D4 = Sequence(CosFourier(N0, mid(π/d))⊗CosFourier(N0, mid(π/d)),coefficients(mid.(U01_D4)))
U02_D4 = Sequence(CosFourier(N0, mid(π/d))⊗CosFourier(N0, mid(π/d)),coefficients(mid.(U02_D4)))

####################### Computation of quantities of interest ##########################

δ0 = interval(0.01) ; δ0big = interval(big(0.01))
κ =  sqrt((interval(1) + abs(interval(1)-λ1*λ2))^2 + interval(1)/interval(λ2)^2)  ### value for κ = sup_{\xi \in \R^2}\|l^{-1}(\xi)\|_2 

#### computation of α1 and α2
α1 = interval(2)*(interval(1) + norm(U02,1) +κ*r0) + interval(3)*λ1*(interval(2)*norm(U01,1) + κ*r0)
α2 = interval(2)*norm(U01,1)

λmax = interval(2.56)
##### the right hand side of the definition below is decreasing in λmax. We just need to find a value for which λmax_test is negative
λmax_test = - interval(1)-λmax + norm(V1,1) + abs(λ1*λ2-interval(1))/(λ2+λmax)*norm(V2,1) + maximum([α2 α1+abs(λ1*λ2-interval(1))/(λ2+λmax)*(norm(U01,1)+κ*r0)])*κ*r0

if sup(λmax_test) < 0
    display("λmax is an upper bound for the largest real eigenvalue")
else
    display("λmax is not big enough")
    return Nan
end 

t = -λmax - interval(0.0001)

##### computation of an upper bound for \|(L-δ0)^{-1}\|_2 
up_Linv = sqrt((interval(1)/interval(1-δ0) + abs(interval(1)-λ1*λ2))^2 + interval(1)/interval(λ2-δ0)^2)


##### computation of the bounds 𝒵u1 and 𝒵u2

#  𝒵u1, 𝒵u2 = computation_Zu(V1_D4,V2_D4,d,N0)

 𝒵u1 = interval(1.9001e-6)
 𝒵u2 = interval(3.1165e-7)

 V1 = interval.(Float64.(inf.(V1),RoundDown),Float64.(sup.(V1),RoundUp) )
 V2 = interval.(Float64.(inf.(V2),RoundDown),Float64.(sup.(V2),RoundUp) )

################# Case cc : cosine-cosine expansion ###########

### computation of the weights allowing to switch from exponential to cosine series (useful for norm computations)
D1 = [convert(Vector{Interval{Float64}},exp2cos(N)) ; convert(Vector{Interval{Float64}},exp2cos(N))]
D2 = interval.(ones(2*(N+1)^2))./D1
SS = CosFourier(N, π/d)⊗CosFourier(N,π/d)

ϵ,S =  computation_bounds(U01,V1,V2,r0,λ1,λ2,δ0,t,d,SS,D1,D2,𝒵u1,𝒵u2)

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


########### verification that the first disk is disjoint from the rest of the disks. We compute an interval (-∞, Imax] containing the intersection of the rest of the disks  

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

D1 = [convert(Vector{Interval{Float64}},exp2cos_sin(N)) ; convert(Vector{Interval{Float64}},exp2cos_sin(N))]
D2 = interval.(ones(2*(N+1)*N))./D1
SS = CosFourier(N, π/d)⊗SinFourier(N,π/d)

ϵ,S =  computation_bounds(U01,V1,V2,r0,λ1,λ2,δ0,t,d,SS,D1,D2,𝒵u1,𝒵u2)

#### In this case we expect one eigenvalue around zero and the rest of the spectrum negative

display("cosine-sine part")
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
    display("Case cosine-sine : the first disk is disjoint from the rest of the disks")
    display("upper bound on the rest of the disks")
    display(sup(Imax))
else 
    display("Case cosine-sine : the first disk is intersecting the rest of the disks")
end 



# # ################# Case ss : sine-sine expansion ###########

D1 = interval.(2*ones(2*N^2))
D2 = interval.(ones(2*N^2))./D1
SS = SinFourier(N, π/d)⊗SinFourier(N,π/d)

ϵ,S =  computation_bounds(U01,V1,V2,r0,λ1,λ2,δ0,t,d,SS,D1,D2,𝒵u1,𝒵u2)


#### In this case we expect the spectrum to be negative
display("sine-sine part")
display("upper bound S[1]")
display(S[1] + ϵ*abs(t+S[1]))
display("lower bound S[1]")
display(S[1] - ϵ*abs(t+S[1]))

display("upper bound S[2]")
display(S[2] + ϵ*abs(t+S[2]))
display("lower bound S[2]")
display(S[2] - ϵ*abs(t+S[2]))


###########  We compute an interval (-∞, Imax] containing the intersection of all the disks  

Imax = interval(-10)
for n = 1:length(S)
    Imax = maximum([Imax S[n]+ϵ*abs(t+S[n])])
end 

if sup(Imax)<0
    display("Case sine-sine : the spectrum is strictly negative")
    display("upper bound on the union of the disks")
    display(sup(Imax))
else 
    display("Case sine-cosine : possibly a positive eigenvalue ?")
end 






