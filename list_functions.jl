#################################### List of the needed functions : go directly to line 245 for the main code ################################################# 

# αₙ for the trace operator (see Section 3.2).
function αₙ(n)
    if n[1] == n[2] == 0
        return 1
    elseif n[1] == n[2] != 0
        return 4
    elseif (n[1] != 0) & (n[2] == 0)
        return 2
    else
        return 4
    end
end

# Computes the trace for a D₄Fourier sequence.
function _trace_D₄(N)
    M = dimension(D₄Fourier(N,1.0))
    S = zeros(N+1,M)
    for n₂ = 0:N
        for n₁ = 0:N
            m = (max(n₁,n₂),min(n₁,n₂))
            α = αₙ(m)
            S[n₁+1,m[1] + m[2]*N - div(((m[2]-2)^2 + 3*(m[2]-2)),2)] = α*(-1)^n₂
        end
    end
    return S
end

# Allows us to switch between D₄ and exponential Fourier series
function _exp2D₄!(D::Vector{Float64},s::D₄Fourier)
    k = 1
    ord = order(s)[1]
    for k₂ = 0:ord
        for k₁ = k₂:ord
            if k₁ == k₂ == 0
                D[k] = 1
                k += 1
            elseif k₁ == k₂ != 0
                D[k] = sqrt(4)
                k += 1
            elseif (k₁ != 0) & (k₂ == 0)
                D[k] = sqrt(4)
                k += 1
            else
                D[k] = sqrt(8)
                k +=1 
            end
        end
    end
    return D
end

# Allows us to switch between D₂ and exponential Fourier series
function exp2cos(N)

    d = 2*((ones((N+1)^2)))

    d[1] = 1;
    for n2=1:N
        d[n2+1] = sqrt(2);
    end

    for n1 = 1:N
        d[n1*(N+1)+1] = sqrt(2);
    end

    return d
end

# Computes convolution of D₄Fourier sequences up to order N
function _conv_small(u,v,N)
    #Computes u*v only up to order N
    order_u = order(space(u))[1]
    order_v = order(space(v))[1]
    C = Sequence(D₄Fourier(N,frequency(u)[1]), interval.(zeros(dimension(D₄Fourier(N,frequency(u)[1])))))
    for i₂ ∈ 0:N
        for i₁ ∈ i₂:N
            Cᵢ = interval(zero(Float64))
            @inbounds @simd for j₁ ∈ max(i₁-order_u, -order_v):min(i₁+order_u, order_v)
                @inbounds for j₂ ∈ max(i₂-order_u, -order_v):min(i₂+order_u, order_v)
                    tu = (max(abs(i₁-j₁),abs(i₂-j₂)),min(abs(i₁-j₁),abs(i₂-j₂)))
                    tv = (max(abs(j₁),abs(j₂)),min(abs(j₁),abs(j₂)))
                    Cᵢ += u[tu] * v[tv]
                end
            end
            C[(i₁,i₂)] = Cᵢ
        end
    end
    return C
end

# Performs convolution up to order N of a D₄ and D₂ Fourier series
function __conv_small(u,v,N)
    #Computes u*v up to order N
    #u is a sequence in D₄Fourier
    #v is a sequence in CosFourier ⊗ CosFourier (D₂ symmetric)
    order_u = order(space(u))[1]
    order_v = order(space(v))[1]
    C = Sequence(CosFourier(N,frequency(u)[1])⊗CosFourier(N,frequency(u)[1]), interval.(zeros((N+1)^2)))
    for i₁ ∈ 0:N
        for i₂ ∈ 0:N
            Cᵢ = interval(zero(Float64))
            @inbounds @simd for j₁ ∈ max(i₁-order_u, -order_v):min(i₁+order_u, order_v)
                @inbounds for j₂ ∈ max(i₂-order_u, -order_v):min(i₂+order_u, order_v)
                    tu = (max(abs(i₁-j₁),abs(i₂-j₂)),min(abs(i₁-j₁),abs(i₂-j₂)))
                    tv = (abs(j₁),abs(j₂))
                    Cᵢ += u[tu] * v[tv]
                end
            end
            C[(i₁,i₂)] = Cᵢ
        end
    end
    return C
end

# Computes convolution of D₄Fourier sequences up to order N
function _conv_smallbig(u,v,N)
    #Computes u*v only up to order N
    order_u = order(space(u))[1]
    order_v = order(space(v))[1]
    C = Sequence(D₄Fourier(N,frequency(u)[1]), interval.(big.(zeros(dimension(D₄Fourier(N,frequency(u)[1]))))))
    for i₂ ∈ 0:N
        for i₁ ∈ i₂:N
            Cᵢ = interval(zero(BigFloat))
            setprecision(80)
            @inbounds @simd for j₁ ∈ max(i₁-order_u, -order_v):min(i₁+order_u, order_v)
                @inbounds for j₂ ∈ max(i₂-order_u, -order_v):min(i₂+order_u, order_v)
                    tu = (max(abs(i₁-j₁),abs(i₂-j₂)),min(abs(i₁-j₁),abs(i₂-j₂)))
                    tv = (max(abs(j₁),abs(j₂)),min(abs(j₁),abs(j₂)))
                    setprecision(80)
                    Cᵢ += u[tu] * v[tv]
                end
            end
            C[(i₁,i₂)] = Cᵢ
        end
    end
    return C
end

# Performs the estimate of Lemma 4.1
function φ(A,B,C,D)
    O₁ = max(A,D) + max(B,C)
    O₂ = sqrt(A^2 + D^2 + B^2 + C^2)
    return min(O₁,O₂)
end


function sinc_int(x)
        N = 8
        f = interval(big(0))
        for n=0:N 
            f = f + ExactReal(-1^n)*x^(2*n+1)/interval(big(factorial(2*n+1)))
        end 
        ξ = interval(big(0.1))^(2*(N+1)+1)/((ExactReal(1)-interval(big(0.1)))*interval(big(factorial(2*(N+1)+1))))
        f = interval(inf(f)-sup(ξ),sup(f)+sup(ξ))
    return f 
end


# Computes the Fourier coefficients of 1_𝒟₀²
function _char_boundary_coeffs(N,f,d)
    char = Sequence(Fourier(N,f)⊗Fourier(N,f), interval.(complex.(big.(zeros((2N+1)^2)))))
    for n₂ = -N:N
        for n₁ = -N:N
            char[(n₁,n₂)] = interval(big(1))/(interval(big(4))*d^2) * exp(1im*n₁*interval(big(π))*(interval(big(1))/d * interval(big(1/2)) - interval(big(1))))*exp(1im*n₂*interval(big(π))*(interval(big(1))/d * interval(big(1/2)) - interval(big(1)))) * sinc_int(n₁/d*interval(big(1/2)))*sinc_int(n₂/d*interval(big(1/2)))
        end
    end
    rchar = Sequence(D₄Fourier(N,f), interval.(big.(zeros(dimension(D₄Fourier(N,f))))))
    for n₂ = 0:N
        for n₁ = n₂:N
            rchar[(n₁,n₂)] = real(char[(n₁,n₂)] + char[(n₂,-n₁)] + char[(-n₁,-n₂)] + char[(-n₂,n₁)])
        end
    end
    return rchar
end

# Computes the sequence a(d,⋅) for a in D₄Fourier.
function _sequence_on_boundary(a)
    N = order(a)[1]
    f = frequency(a)[1]
    anew = Sequence(CosFourier(N,f), interval.(big.(zeros(N+1))))
    for n₁ = 0:N
        for n₂ = -N:N
            anew[n₁] += a[(max(n₁,abs(n₂)),min(n₁,abs(n₂)))]*(-1)^n₂
        end
    end
    return anew
end

# Computes the Fourier coefficients of 1_𝒟₀
function _char_1D_boundary_coeffs(N,f,d)
    char = Sequence(Fourier(N,f), interval.(complex.(big.(zeros((2N+1))))))
    for n = -N:N
        char[n] = interval(big(1))/(interval(big(2))*d) * exp(1im*n*interval(big(π))*(interval(big(1))/d * interval(big(1/2)) - interval(big(1)))) * sinc_int(n/d*interval(big(1/2)))
    end
    rchar = Sequence(CosFourier(N,f), interval.(big.(zeros(N+1))))
    for n = 0:N
        rchar[n] = real(char[n] + char[-n])
    end
    return rchar
end

# Computes the function needed to take the convolution with ∂ₓ₁V₁ᴺ
# We denote by (Ṽⱼ)ₘ = |m̃₁| (Vⱼᴺ)ₘ 
function _Ṽⱼ_coeffs(Vⱼᴺ)
    N = order(Vⱼᴺ)[1]
    f = frequency(Vⱼᴺ)[1]
    Ṽⱼ = Sequence(CosFourier(N,f)⊗CosFourier(N,f), interval.(big.(zeros((N+1)^2))))
    for n₁ = 0:N
        for n₂ = 0:N
            Ṽⱼ[(n₁,n₂)] = abs(n₁)*f*Vⱼᴺ[(max(n₁,n₂),min(n₁,n₂))]
        end
    end
    return Ṽⱼ
end

# Checks the conditions of the Radii-Polynomial Theorem 3.1.
function CAP(𝒴₀,𝒵₁,𝒵₂,r₀)
    if 𝒵₁ + 𝒵₂*r₀ < 1
        if interval(1/2)*𝒵₂*r₀^2 - (interval(1)-𝒵₁)*r₀ + 𝒴₀ < 0
          display("The proof was successful for r₀ = ")
          display(sup(r₀))  
        else
          display("The condition 2𝒴₀*𝒵₂ < (1-𝒵₁)² is not satisfied")
        end
    else
        if 𝒵₁ > 1
            display("𝒵₁ is too big")
        else
          display("failure: linear term is positive")
        end
      end
end




function D4_2_Cos(U)
    N = order(space(U))[1]
    f = frequency(U)[1]
    V = Sequence(CosFourier(N,f)⊗CosFourier(N,f),interval.(big.(zeros((N+1)^2))))
    for n1 = 0:N
        for n2 = 0:n1-1 
            V[(n1,n2)] = U[(n1,n2)]
        end 
        for n2 = n1:N 
            V[(n1,n2)] = U[(n2,n1)]
        end 
    end
    return V 
end 



### We adapt the code given at https://github.com/dominicblanco/LocalizedPatternsGS.jl for the computation of 𝒵u 
### In particular we compute it in the worst case scenario, that is when μ = δ0 
function computation_Zu(V1,V2,d,N)
    # Computation of the 𝒵ᵤ₁ bound defined in Lemma 4.9.
print("Starting 𝒵ᵤ")
setprecision(80)
a₁big = sqrt((ExactReal(1)-δ0big)/λ1big)
setprecision(80)
a₂big = sqrt(λ2big-δ0big)
a₁ = sqrt((interval(1)-δ0)/λ1)
a₂ = sqrt(λ2-δ0)
# The constants C₀f₁₁ and C₀f₂₂ in Lemma 4.8
C₀f₁₁ = max(a₁^2*interval(interval(2)*exp(interval(5/4)))*(interval(2)/a₁)^(interval(1/4)),a₁^2*sqrt(interval(π)/(interval(2)*sqrt(a₁))))
C₀f₂₂ = max(interval(interval(2)*exp(interval(5/4)))*(interval(2)/a₂)^(1/4),sqrt(interval(π)/(interval(2)*sqrt(a₂))))
# Computing the fourier series of E₁ and E₂ defined in Lemma 4.9.
setprecision(80)
E₁big = Sequence(D₄Fourier(4N,π/d), interval.(big.(zeros(dimension(D₄Fourier(4N,π/d))))))
setprecision(80)
E₂big = Sequence(D₄Fourier(4N,π/d), interval.(big.(zeros(dimension(D₄Fourier(4N,π/d))))))
for n₂ = 0:4N
    for n₁ = n₂:4N
        setprecision(80)
        E₁big[(n₁,n₂)] = real(interval(big(1))/(interval(big(8))*dbig) * ((-interval(big(1)))^n₁*sinc(n₂)*(interval(big(1))/(interval(big(2))*a₁big-im*n₁*interval(big(π))/dbig) + interval(big(1))/(interval(big(2))*a₁big + im*n₁*interval(big(π))/dbig)) + (-interval(big(1)))^n₂*sinc(n₁)*(interval(big(1))/(interval(big(2))*a₁big-im*n₂*interval(big(π))/dbig) + interval(big(1))/(interval(big(2))*a₁big + im*n₂*interval(big(π))/dbig))))
        setprecision(80)
        E₂big[(n₁,n₂)] = real(interval(big(1))/(interval(big(8))*dbig) * ((-interval(big(1)))^n₁*sinc(n₂)*(interval(big(1))/(interval(big(2))*a₂big-im*n₁*interval(big(π))/dbig) + interval(big(1))/(interval(big(2))*a₂big + im*n₁*interval(big(π))/dbig)) + (-interval(big(1)))^n₂*sinc(n₁)*(interval(big(1))/(interval(big(2))*a₂big-im*n₂*interval(big(π))/dbig) + interval(big(1))/(interval(big(2))*a₂big + im*n₂*interval(big(π))/dbig))))
    end
end
E₁ = interval.(Float64.(inf.(E₁big),RoundDown),Float64.(sup.(E₁big),RoundUp) )
E₂ = interval.(Float64.(inf.(E₂big),RoundDown),Float64.(sup.(E₂big),RoundUp) )
# Computes a larger operator to convert from D₄ to exponential since inner products will be of size 2N.
P2 = interval.(vec(_exp2D₄!(zeros(dimension(D₄Fourier(2N,π/d))),D₄Fourier(2N,π/d))))

setprecision(80)
P2big = interval.(big.(vec(_exp2D₄!(zeros(dimension(D₄Fourier(2N,π/d))),D₄Fourier(2N,π/d)))))

# Computation of the 𝒵ᵤ₁₁ bound, the first quantity defined in Lemma 4.9.
print("Computing 𝒵ᵤ₁₁")
V₁ᴺ_interval = project(V1,D₄Fourier(2N,π/d))
V₂ᴺ_interval = project(V2,D₄Fourier(2N,π/d))
Ω₀ = ExactReal(4)*d^2
#For spike and ring, use lines 457 through 461
E₁V₁ = _conv_small(E₁,V₁ᴺ_interval, 2N)
_inner_prod_E₁V₁ = abs(coefficients(P2.*V₁ᴺ_interval)'*coefficients(P2.*E₁V₁))
@show _inner_prod_E₁V₁
𝒵ᵤ₁₁ = sqrt(interval(2))*C₀f₁₁*(1-exp(-4a₁*d)) * (interval(2π))^(interval(1/4))/a₁^(interval(3/4))*sqrt(Ω₀) *sqrt(_inner_prod_E₁V₁)  
@show 𝒵ᵤ₁₁


# Computation of the 𝒵ᵤ₁₂ bound, the second quantity defined in Lemma 4.9.
print("Computing 𝒵ᵤ₁₂")
# For spike and ring, use lines 477 through 481
E₂V₂ = _conv_small(E₂,V₂ᴺ_interval, 2N)
_inner_prod_E₂V₂ = abs(coefficients(P2.*V₂ᴺ_interval)'*coefficients(P2.*E₂V₂))
@show _inner_prod_E₂V₂
𝒵ᵤ₁₂ = sqrt(interval(2))*C₀f₂₂*(interval(1)-exp(-4a₂*d)) * (interval(2π))^(interval(1/4))/a₂^(interval(3/4))*sqrt(Ω₀)*sqrt(_inner_prod_E₂V₂)  
@show 𝒵ᵤ₁₂

𝒵ᵤ₁₃ = 𝒵ᵤ₁₂  

#Now, we compute the full 𝒵ᵤ₁ bound concluding the computation of Lemma 4.9.
𝒵ᵤ₁ = sqrt((𝒵ᵤ₁₁ + 𝒵ᵤ₁₃)^2 + 𝒵ᵤ₁₂^2)
@show 𝒵ᵤ₁
################################ 𝒵ᵤ₂ BOUND ######################################################
# Computation of the 𝒵ᵤ₂ bound defined in Lemma 4.10.
# We begin by computing all the necessary constants.
# We start with C₁₁f₁₁,C₁₂f₁₁,C₁₁f₂₂, and C₁₂f₂₂ defined in Lemma 4.10.
print("Computing 𝒵ᵤ₂")
C₁₁f₁₁ = a₁^3*sqrt(interval(π/2))*interval(1)/sqrt(a₁ + interval(1))*(interval(1) + interval(1)/a₁)
C₁₂f₁₁ = a₁^2*sqrt(interval(π/2))*(sqrt(interval(2))*a₁ + interval(1))

C₁₁f₂₂ = a₂*sqrt(interval(π/2))*1/sqrt(a₂ + interval(1))*(interval(1)+interval(1)/a₂)
C₁₂f₂₂ = sqrt(interval(π/2))*(sqrt(interval(2))*a₂ + interval(1))

# Next, we define the constants Cⱼ,𝒞₁ⱼ, and 𝒞₂ⱼ for j = 1,2 defined in Lemma 4.10.
C₁ = sqrt(d^2/(interval(16)*a₁^2*interval(π)^5) + interval(1)/a₁^4 + d/a₁^3)
C₂ = sqrt(d^2/(interval(16)*a₂^2*interval(π)^5) + interval(1)/a₂^4 + d/a₂^3)
𝒞₁₁ = interval(2)*sqrt(Ω₀)*exp(-a₁*d)*(C₁₁f₁₁*exp(-a₁) + C₁₂f₁₁)/a₁
𝒞₂₁ = interval(2)*sqrt(Ω₀)*C₁₁f₁₁*sqrt(log(interval(2))^2 + interval(2)*log(interval(2)) + interval(2))
𝒞₁₂ = interval(2)*sqrt(Ω₀)*exp(-a₂*d)*(C₁₁f₂₂*exp(-a₂) + C₁₂f₂₂)/a₂
𝒞₂₂ = interval(2)*sqrt(Ω₀)*C₁₁f₂₂*sqrt(log(interval(2))^2 + interval(2)*log(interval(2)) + interval(2))

# Now, we compute 1_𝒟₀² and 1_𝒟₀, the Fourier series representations of the
# characteristic functions on 𝒟₀² and 𝒟₀ respectively. We do these computations
# using the functions _char_boundary_coeffs and _char_1D_boundary_coeffs.
print("Computing coefficients of characteristic function")
setprecision(80)
char = _char_boundary_coeffs(4N,frequency(U01_D4)[1],dbig)
setprecision(80)
char1D = _char_1D_boundary_coeffs(4N,frequency(U01_D4)[1],dbig)
#Note that the function char is the characteristic function on all four corners.
# Indeed, since Vⱼᴺ is D₄-symmetric, we can compute the norm of Vⱼᴺ on all four corners
# and divide by 1/4 to obtain the result. For the norm involving ∂ₓ₁v₁ᴺ, we upper bound
# by the norm in the upper right corner by the norm on all four corners. This allows us
# to compute the convolution of a D₄ and D₂ sequence, which is less computationally expensive.
# Indeed, (∂ₓ₁v₁ᴺ)² is an even (D₂) function.

# Similarly, char1D is the characteristic function of 𝒟₀ ∪ (-d,-d+1). Since v₁ᴺ(d,⋅)
# is even, we can take the norm on this domain and multiply by 1/2.
P3 = interval.(exp2cos(2N))
P4 = interval.([1 ; sqrt(2)*ones(2N)])

setprecision(80)
V₁ᴺbig = project(V1,D₄Fourier(2N,π/d))
setprecision(80)
Ṽ₁big = _Ṽⱼ_coeffs(V₁ᴺbig)
setprecision(80)
V₁ᴺdbig = _sequence_on_boundary(V₁ᴺbig)

setprecision(80)
V₂ᴺbig = project(V2,D₄Fourier(2N,π/d))
setprecision(80)
Ṽ₂big = _Ṽⱼ_coeffs(V₂ᴺbig)
setprecision(80)
V₂ᴺdbig = _sequence_on_boundary(V₂ᴺbig)

char = interval.(Float64.(inf.(char),RoundDown),Float64.(sup.(char),RoundUp) ) 
char1D = interval.(Float64.(inf.(char1D),RoundDown),Float64.(sup.(char1D),RoundUp) ) 
Ṽ₁_interval = interval.(Float64.(inf.(Ṽ₁big),RoundDown),Float64.(sup.(Ṽ₁big),RoundUp) ) 
V₁ᴺd_interval = interval.(Float64.(inf.(V₁ᴺdbig),RoundDown),Float64.(sup.(V₁ᴺdbig),RoundUp) ) 
Ṽ₂_interval = interval.(Float64.(inf.(Ṽ₂big),RoundDown),Float64.(sup.(Ṽ₂big),RoundUp) ) 
V₂ᴺd_interval = interval.(Float64.(inf.(V₂ᴺdbig),RoundDown),Float64.(sup.(V₂ᴺdbig),RoundUp) ) 


# We now compute each 𝒵ᵤ₂ⱼ bound for  j = 1,2,3. Beginning with 𝒵ᵤ₂₁,
print("Computing 𝒵ᵤ₂₁")
charṼ₁ = __conv_small(char,Ṽ₁_interval,2N)
_boundary_inner_prod∂ₓ₁V₁ = abs(coefficients(P3.*charṼ₁)'*coefficients(P3.*Ṽ₁_interval))
@show _boundary_inner_prod∂ₓ₁V₁

charV₁ = _conv_small(char,V₁ᴺ_interval,2N)
_boundary_inner_prodV₁ = abs(coefficients(P2.*charV₁)'*coefficients(P2.*V₁ᴺ_interval))
@show _boundary_inner_prodV₁

char1DV₁d = project(char1D*V₁ᴺd_interval,space(V₁ᴺd_interval))
_boundary_inner_prodV₁d = abs(coefficients(P4.*char1DV₁d)'*coefficients(P4.*V₁ᴺd_interval))
@show _boundary_inner_prodV₁d

CV₁ᴺ = sqrt(interval(1/8) * sqrt(_boundary_inner_prod∂ₓ₁V₁)*sqrt(_boundary_inner_prodV₁) + interval(1)/(2d) * interval(1/4) * _boundary_inner_prodV₁d)

𝒵ᵤ₂₁ = interval(4)/sqrt(Ω₀) * C₁ * (𝒞₁₁ * sqrt(_inner_prod_E₁V₁) + 𝒞₂₁*CV₁ᴺ)
@show 𝒵ᵤ₂₁

# Next, we compute 𝒵ᵤ₂₂
print("Computing 𝒵ᵤ₂₂")
charṼ₂ = __conv_small(char,Ṽ₂_interval,2N)
_boundary_inner_prod∂ₓ₁V₂ = abs(coefficients(P3.*charṼ₂)'*coefficients(P3.*Ṽ₂_interval))
@show _boundary_inner_prod∂ₓ₁V₂

charV₂ = _conv_small(char,V₂ᴺ_interval,2N)
_boundary_inner_prodV₂ = abs(coefficients(P2.*charV₂)'*coefficients(P2.*V₂ᴺ_interval))
@show _boundary_inner_prodV₂

char1DV₂d = project(char1D*V₂ᴺd_interval,space(V₂ᴺd_interval))
_boundary_inner_prodV₂d = abs(coefficients(P4.*char1DV₂d)'*coefficients(P4.*V₂ᴺd_interval))
@show _boundary_inner_prodV₂d

CV₂ᴺ = sqrt(interval(1/8) * sqrt(_boundary_inner_prod∂ₓ₁V₂)*sqrt(_boundary_inner_prodV₂) + interval(1)/(2d) * interval(1/4) * _boundary_inner_prodV₂d)

𝒵ᵤ₂₂ = interval(4)/sqrt(Ω₀) * C₂ * (𝒞₁₂ * sqrt(_inner_prod_E₂V₂) + 𝒞₂₂*CV₂ᴺ)
@show 𝒵ᵤ₂₂

# Finally, we compute 𝒵ᵤ₂₃. Note that we require an additional inner product for its computation.
print("Computing 𝒵ᵤ₂₃")
E₁V₂ = _conv_small(E₁,V₂ᴺ_interval, 2N)
_inner_prod_E₁V₂ = abs(coefficients(P2.*V₂ᴺ_interval)'*coefficients(P2.*E₁V₂))
@show _inner_prod_E₁V₂

𝒵ᵤ₂₃ =  min(C₁,C₂) *(𝒵ᵤ₂₂/C₂ + interval(4)*(λ1+δ0)/sqrt(Ω₀)*(𝒞₁₁ * sqrt(_inner_prod_E₁V₂) + 𝒞₂₁*CV₂ᴺ))
@show 𝒵ᵤ₂₃

# Finally, we can compute 𝒵ᵤ₂
𝒵ᵤ₂ = sqrt((𝒵ᵤ₂₁ + 𝒵ᵤ₂₃)^2 + 𝒵ᵤ₂₂^2)
@show 𝒵ᵤ₂

    return 𝒵ᵤ₁, 𝒵ᵤ₂
end 



function exp2cos(N)

    d = ExactReal(2)*(interval.(ones((N+1)^2)))

    d[1] =ExactReal(1);
    for n2=1:N
        d[n2+1] = sqrt(ExactReal(2));
    end

    for n1 = 1:N
        d[n1*(N+1)+1] = sqrt(ExactReal(2));
    end

    return d
end


function exp2sin_cos(N)

    d = ExactReal(2)*(interval.(ones((N+1)*N)))
    for n2=1:N
        d[n2] = sqrt(ExactReal(2));
    end
    return d
end


function exp2cos_sin(N)

    d = ExactReal(2)*(interval.(ones((N+1)*N)))
    for n1 = 1:N
        d[(n1-1)*(N+1)+1] = sqrt(ExactReal(2));
    end
    return d
end






function computation_bounds(U01,V1,V2,r0,λ1,λ2,δ0,t,d,SS,D1,D2,𝒵u1,𝒵u2)

    # Construction of the operator L
    ∂1 = project(Derivative((2,0)), SS, SS,Interval{Float64})
    ∂2 = project(Derivative((0,2)), SS, SS,Interval{Float64})
    Δ = ∂1 + ∂2

    L11 = Δ*λ1 - I
    L21 = - ExactReal(0)*Δ*λ1 + (λ1*λ2-1)*I
    L22 = Δ - λ2*I

    DG11 = project(Multiplication(V1),SS,SS,Interval{Float64})
    DG12 = project(Multiplication(V2),SS,SS,Interval{Float64})

    DF = [coefficients(L11)+coefficients(DG11) coefficients(DG12);
          coefficients(L21) coefficients(L22)]
    display("DF created")

    # DF = L + DG ; L = Nothing ; DG = Nothing 
    D,P =  eigen(-mid.(D1).*mid.(DF).*mid.(D2)')

    #### We compute an approximate inverse Pinv for P. Computing the defect nP, we can use a Neumann series argument to compute the real inverse of P as P^{-1} = sum_k (I - Pinv*P)^k Pinv. We propagate the error given by nP in the bounds below. In practice, nP is very small and will not affect the quality of the bounds 
    P = interval.(mid.(D2).*P.*mid.(D1)') ; Pinv = interval.(inv(mid.(P))) ; nP = opnorm(LinearOperator(I - D1.*_matprod(Pinv,P).*D2'),2)
    norm_P = spec_norm(D1.*P.*D2') ; norm_Pinv = spec_norm(D1.*Pinv.*D2')  

    display("norm of Pinv")
    display(norm_Pinv)

    D = _matprod(_matprod(Pinv,DF),P)  ; DF = Nothing
    S = diag(D) ; R = D - Diagonal(S) 

    St = S .+ t ;  Stinv = ExactReal(1) ./ St 

    ####### Computation of bounds Z1i ################
    
    ####### diagonal part of DG(U0) is given by V0[(0,0)]
    display("values of the Z1i")

    ### computation of \|π_N (L-δ0)^{-1}\|_2. For this we compute the eigenvalues of l^{-1}(ξ) and evaluate the maximum for ξ \geq (N+1)π/d 
    a1 = ExactReal(1)/abs(V1[(0,0)] - interval(1) - λ1*(interval(N+1)*π/d)^2) 
    a2 = ExactReal(0)
    a3 = abs(ExactReal(1)-λ1*λ2)/(abs(V1[(0,0)] - interval(1) - λ1*(interval(N+1)*π/d)^2)*abs(λ2+(interval(N+1)*π/d)^2))
    a4 = ExactReal(1)/abs(λ2+(interval(N+1)*π/d)^2) 

    ### formula for the eigenvalues of a matrix [a1 a2;a3 a4]
    max_Linv = ExactReal(0.5)*(a1 + a4 + sqrt(interval(4)*a2*a3 + (a1-a4)^2)) 

    Z12 = max_Linv*sqrt(norm(V1-V1[(0,0)],1)^2 + norm(V2,1)^2)
    display("Z12")
    display(Z12)

    Z13 = spec_norm((D1.*Stinv).*R.*D2')*(ExactReal(1)+ nP*norm_Pinv/(ExactReal(1)-nP))
    display("Z13")
    display(Z13)

    N1 = 50 ; V1_p = project(V1,CosFourier(N1, π/d)⊗CosFourier(N1, π/d)) ; V12 = V1_p*V1_p
    V2_p = project(V2,CosFourier(N1, π/d)⊗CosFourier(N1, π/d)) ; V22 = V2_p*V2_p
    DG11 = project(Multiplication(V1_p),SS,SS,Interval{Float64})
    DG12 = project(Multiplication(V2_p),SS,SS,Interval{Float64})


    M1 = coefficients(project(Multiplication(V12),SS, SS, Interval{Float64})) - _matprod(coefficients(DG11),coefficients(DG11))
    M2 = coefficients(project(Multiplication(V22),SS, SS, Interval{Float64})) - _matprod(coefficients(DG12),coefficients(DG12))

    M1 = [M1 M2;
          ExactReal(0)*M1 ExactReal(0)*M2]

    Z14 = sqrt(spec_norm((D1.*Stinv).*_matprod(_matprod(Pinv,M1),Pinv').*((Stinv.*D2)')))*(ExactReal(1)+ nP*norm_Pinv/(ExactReal(1)-nP) + norm_Pinv*sqrt(norm(V1_p-V1,1)^2+norm(V2_p-V2,1)^2)*maximum(Stinv))
    display("Z14")
    display(Z14)

    Z11 = max_Linv*sqrt(spec_norm(D1.*_matprod(_matprod(P',M1),P).*D2'))
    display("Z11")
    display(Z11)


    ###### Computation of the bounds 𝒞1*r0 an 𝒞2*r0

    L = [coefficients(L11) interval(0)*coefficients(L11);
        coefficients(L21) coefficients(L22)]
    norm_SPL = spec_norm((D1.*Stinv).*_matprod(Pinv,L-δ0*I).*D2')*(ExactReal(1)+ nP*norm_Pinv/(ExactReal(1)-nP))

    display("value norm SPL")
    display(norm_SPL)

    𝒞1 = up_Linv*maximum([α2 α1+abs(λ1*λ2-interval(1))/(λ2+λmax)*(norm(U01,1)+κ*r0)])*κ*r0
    𝒞2 = norm_SPL*up_Linv*maximum([α2 α1+abs(λ1*λ2-interval(1))/(λ2+λmax)*(norm(U01,1)+κ*r0)])*κ*r0

    display("values of the 𝒞i")
    display(𝒞1)
    display(𝒞2)

    ########## Computation of the bound 𝒵u3

    𝒵u3 = norm_SPL*𝒵u2

    display("value of 𝒵u3")
    display(𝒵u3)

    ######### Computation of ϵ


    if sup(𝒞1)<1
        κ1 = (𝒵u1 + 𝒞1)/(ExactReal(1)-𝒞1)
        display("κ1")
        display(κ1)
        if sup(Z12 + 𝒵u2 + sqrt(ExactReal(1) + κ1^2)*𝒞1) < 1
            if sup(Z12 + 𝒵u2) < 1
                κ2 = (Z11 + (𝒵u2 + sqrt(ExactReal(1) + κ1^2)*𝒞1)*norm_P)/(ExactReal(1) - (Z12 + 𝒵u2 + sqrt(ExactReal(1) + κ1^2)*𝒞1))
                display("κ2")
                display(κ2)
                ϵq = Z13 + Z14*(Z11 + ExactReal(2)*𝒵u2*norm_P)/(ExactReal(1) - Z12 - ExactReal(2)*𝒵u2) + ExactReal(2)*𝒵u3*(norm_P + (Z11 + ExactReal(2)*𝒵u2*norm_P)/(ExactReal(1) - Z12 - ExactReal(2)*𝒵u2))
                ϵ = Z13 + Z14*κ2 + (𝒵u3 + 𝒞2*sqrt(ExactReal(1) + κ1^2))*(norm_P + κ2)
                return maximum([ϵ ϵq]),S
            else 
                display("third condition not respected")
                return Nan
            end
        else 
            display("second condition not respected")
            return Nan 
        end 
    else 
        display("first condition not respected")
        return Nan 
    end 
   
end

