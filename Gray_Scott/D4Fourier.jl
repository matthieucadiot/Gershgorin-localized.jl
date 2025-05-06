# Sequence structure for RadiiPolynomial.jl that enforces D₄-symmetry.
# In this file, we create an abstract type that works natively with RadiiPolynomial.jl
# to enforce D₄-symmetry on the Fourier coefficients. That is, 
# (n₁,n₂) = (-n₁,n₂) = (n₁,-n₂) = (n₂,n₁).
# This allows us to isolate solutions when solving problems with Fourier series.

#####################################################################################################################################################################


# Needed packages
using RadiiPolynomial, SparseArrays

# Note that the abstract type below is already defined in RadiiPolynomia.jl, but
# we must include it directly in order to define our own subtype of SpecialOperator.
"""
    SpecialOperator

Abstract type for all special operators.
"""
abstract type SpecialOperator end

add!(C::LinearOperator, S₁::SpecialOperator, S₂::SpecialOperator) = add!(C, project(S₁, domain(C), codomain(C), eltype(C)), S₂)
add!(C::LinearOperator, S::SpecialOperator, A::LinearOperator) = add!(C, project(S, domain(C), codomain(C), eltype(C)), A)
add!(C::LinearOperator, A::LinearOperator, S::SpecialOperator) = add!(C, A, project(S, domain(C), codomain(C), eltype(C)))

sub!(C::LinearOperator, S₁::SpecialOperator, S₂::SpecialOperator) = sub!(C, project(S₁, domain(C), codomain(C), eltype(C)), S₂)
sub!(C::LinearOperator, S::SpecialOperator, A::LinearOperator) = sub!(C, project(S, domain(C), codomain(C), eltype(C)), A)
sub!(C::LinearOperator, A::LinearOperator, S::SpecialOperator) = sub!(C, A, project(S, domain(C), codomain(C), eltype(C)))

radd!(A::LinearOperator, S::SpecialOperator) = radd!(A, project(S, domain(A), codomain(A), eltype(A)))
rsub!(A::LinearOperator, S::SpecialOperator) = rsub!(A, project(S, domain(A), codomain(A), eltype(A)))

ladd!(S::SpecialOperator, A::LinearOperator) = ladd!(project(S, domain(A), codomain(A), eltype(A)), A)
lsub!(S::SpecialOperator, A::LinearOperator) = lsub!(project(S, domain(A), codomain(A), eltype(A)), A)

function Base.:*(S::SpecialOperator, A::LinearOperator)
    codomain_A = codomain(A)
    image_S = image(S, codomain_A)
    return project(S, codomain_A, image_S, _coeftype(S, codomain_A, eltype(A))) * A
end

function RadiiPolynomial.mul!(C::LinearOperator, S₁::SpecialOperator, S₂::SpecialOperator, α::Number, β::Number)
    domain_C = domain(C)
    return mul!(C, S₁, project(S₂, domain_C, image(S₂, domain_C), eltype(C)), α, β)
end
RadiiPolynomial.mul!(C::LinearOperator, S::SpecialOperator, A::LinearOperator, α::Number, β::Number) =
    RadiiPolynomial.mul!(C, project(S, codomain(A), codomain(C), eltype(C)), A, α, β)
RadiiPolynomial.mul!(C::LinearOperator, A::LinearOperator, S::SpecialOperator, α::Number, β::Number) =
    RadiiPolynomial.mul!(C, A, project(S, domain(C), domain(A), eltype(C)), α, β)

RadiiPolynomial.mul!(c::Sequence, S::SpecialOperator, a::Sequence, α::Number, β::Number) =
    RadiiPolynomial.mul!(c, project(S, space(a), space(c), eltype(c)), a, α, β)

#

function Base.:+(A::LinearOperator, S::SpecialOperator)
    domain_A = domain(A)
    new_codomain = image(+, codomain(A), image(S, domain_A))
    return ladd!(A, project(S, domain_A, new_codomain, _coeftype(S, domain_A, eltype(A))))
end
function Base.:+(S::SpecialOperator, A::LinearOperator)
    domain_A = domain(A)
    new_codomain = image(+, image(S, domain_A), codomain(A))
    return radd!(project(S, domain_A, new_codomain, _coeftype(S, domain_A, eltype(A))), A)
end
function Base.:-(A::LinearOperator, S::SpecialOperator)
    domain_A = domain(A)
    new_codomain = image(-, codomain(A), image(S, domain_A))
    return lsub!(A, project(S, domain_A, new_codomain, _coeftype(S, domain_A, eltype(A))))
end
function Base.:-(S::SpecialOperator, A::LinearOperator)
    domain_A = domain(A)
    new_codomain = image(-, image(S, domain_A), codomain(A))
    return rsub!(project(S, domain_A, new_codomain, _coeftype(S, domain_A, eltype(A))), A)
end


# Laplacian operator
# In D₄-symmetric Fourier series, the action of partial differentiation need not be closed
# This means if we take partial derivatives of a D₄ sequence, the result may no longer
# have D₄-symmetry. As a result, we restrict the class of equations under which
# D₄-symmetry can be enforced to those with Laplacian differential operators as
# the Laplacian is closed under the action of any space group.

struct Laplacian{T<:Union{Int,Tuple{Vararg{Int}}}} <: SpecialOperator
    order :: T
    function Laplacian{T}(order::T) where {T<:Int}
        order < 0 && return throw(DomainError(order, "Laplacian is only defined for positive integers"))
        return new{T}(order)
    end
    function Laplacian{T}(order::T) where {T<:Tuple{Vararg{Int}}}
        any(n -> n < 0, order) && return throw(DomainError(order, "Laplacian is only defined for positive integers"))
        return new{T}(order)
    end
    Laplacian{Tuple{}}(::Tuple{}) = throw(ArgumentError("Laplacian is only defined for at least one Int"))
end

Laplacian(order::T) where {T<:Int} = Laplacian{T}(order)
Laplacian(order::T) where {T<:Tuple{Vararg{Int}}} = Laplacian{T}(order)
Laplacian(order::Int...) = Laplacian(order)

RadiiPolynomial.order(Δ::Laplacian) = Δ.order

Base.:*(Δ::Laplacian, a::Sequence) = _compute_laplacian(a, order(Δ))

(Δ::Laplacian)(a::Sequence) = *(Δ, a)

function _compute_laplacian(a::Sequence, α=1)
    Δ = Laplacian(α)
    space_a = space(a)
    new_space = RadiiPolynomial.image(Δ, space_a)
    CoefType = RadiiPolynomial._coeftype(Δ, space_a, eltype(a))
    c = Sequence(new_space, Vector{CoefType}(undef, dimension(new_space)))
    RadiiPolynomial._apply!(c, Δ, a)
    return c
end



# Here, we create the abstract type under which our D₄ sequence structure will live.
# Note that we put it under BaseSpace as D₄-symmetry is purely 2D and cannot be broken
# down as a tensor product. Hence, it is a 2D BaseSpace based on the classification
# of RadiiPolynomial.jl
abstract type DihedralTensorSpace <: BaseSpace end

RadiiPolynomial.desymmetrize(s::DihedralTensorSpace) = s.space

RadiiPolynomial.order(s::DihedralTensorSpace) = RadiiPolynomial.order(RadiiPolynomial.desymmetrize(s))
RadiiPolynomial.frequency(s::DihedralTensorSpace) = RadiiPolynomial.frequency(RadiiPolynomial.desymmetrize(s))

Base.issubset(s₁::DihedralTensorSpace, s₂::DihedralTensorSpace) = false
Base.issubset(s₁::DihedralTensorSpace, s₂::TensorSpace) = issubset(RadiiPolynomial.desymmetrize(s₁), s₂)
Base.union(s₁::DihedralTensorSpace, s₂::DihedralTensorSpace) = union(RadiiPolynomial.desymmetrize(s₁), RadiiPolynomial.desymmetrize(s₂))
Base.union(s₁::DihedralTensorSpace, s₂::TensorSpace) = union(RadiiPolynomial.desymmetrize(s₁), s₂)
Base.union(s₁::TensorSpace, s₂::DihedralTensorSpace) = union(s₁, RadiiPolynomial.desymmetrize(s₂))


# D₄Fourier, the D₄-symmetric sequence structure
struct D₄Fourier{T<:Real} <: DihedralTensorSpace
    space :: TensorSpace{Tuple{Fourier{T},Fourier{T}}}
    D₄Fourier{T}(space::TensorSpace{Tuple{Fourier{T},Fourier{T}}}) where {T<:Real} = new{T}(space)
end
D₄Fourier(space::TensorSpace{Tuple{Fourier{T},Fourier{T}}}) where {T<:Real} = D₄Fourier{T}(space)
D₄Fourier{T}(order::Int, frequency::T) where {T<:Real} = D₄Fourier(TensorSpace(Fourier{T}(order, frequency),Fourier{T}(order,frequency)))
D₄Fourier(order::Int, frequency::Real) = D₄Fourier(TensorSpace(Fourier(order, frequency),Fourier(order,frequency)))

Base.:(==)(s₁::D₄Fourier, s₂::D₄Fourier) = RadiiPolynomial.desymmetrize(s₁) == RadiiPolynomial.desymmetrize(s₂)
Base.issubset(s₁::D₄Fourier, s₂::D₄Fourier) = issubset(RadiiPolynomial.desymmetrize(s₁), RadiiPolynomial.desymmetrize(s₂))
Base.intersect(s₁::D₄Fourier, s₂::D₄Fourier) = D₄Fourier(intersect(RadiiPolynomial.desymmetrize(s₁), RadiiPolynomial.desymmetrize(s₂)))
Base.union(s₁::D₄Fourier, s₂::D₄Fourier) = D₄Fourier(union(RadiiPolynomial.desymmetrize(s₁), RadiiPolynomial.desymmetrize(s₂)))

# Creates the indices for D₄Fourier without applying the algorithm to compute the 
# reduced set. We do this by using a Julia generator. Note that each time indices 
# is called, a vector is allocated. Hence, it is in our best interest to avoid
# calling this function when possible. As a result, throughout the code, we often
# take the steps necessary to remove calls of indices present in the other 
# structures of RadiiPolynomial.jl (see Fourier, CosFourier for examples).

RadiiPolynomial.indices(s::D₄Fourier) = collect((t for t in indices(s.space) if 0 ≤ t[2] ≤ t[1]))
RadiiPolynomial._findindex_constant(s::D₄Fourier) = (0,0)

function _isomorphism_position_unit_range(i::Tuple{UnitRange{Int64},UnitRange{Int64}},s::D₄Fourier,ord)
    K = []
    for k₂ = i[2][1]:i[2][end]
        for k₁ = k₂:i[1][end]
            push!(K,k₁ + k₂*ord - div(((k₂-2)^2 + 3*(k₂-2)),2))
        end
    end
    return vec(K)
end

function _isomorphism_position_unit_range(i::Tuple{UnitRange{Int64},Int64},s::D₄Fourier,ord)
    K = []
    α = i[2]*ord - div(((i[2]-2)^2 + 3*(i[2]-2)),2)
    for k₁ = i[1][1]:i[1][end]
        push!(K,k₁ + α)
    end
    return vec(K)
end

function _isomorphism_position_unit_range(i::Tuple{Int64,UnitRange{Int64}},s::D₄Fourier,ord)
    K = []
    for k₂ = i[2][1]:i[2][end]
        push!(K,i[1] + k₂*ord - div(((k₂-2)^2 + 3*(k₂-2)),2))
    end
    return vec(K)
end

# Functions to find the position of an index in the sequence. The coefficients
# are stored as a vector; hence, we must understand what we mean by s[n] for n in
# the reduced set of indices for D₄-symmetry. The following functions provide the answer.
RadiiPolynomial._findposition(i::Tuple{Int64,Int64}, s::D₄Fourier) = i[1] + i[2]*RadiiPolynomial.order(s)[1] - div(((i[2]-2)^2 + 3*(i[2]-2)),2)
RadiiPolynomial._findposition(u::AbstractRange{Tuple{Int64,Int64}}, s::D₄Fourier) = u[1] + u[2]*RadiiPolynomial.order(s)[1] - div(((u[2]-2)^2 + 3*(u[2]-2)),2)
RadiiPolynomial._findposition(u::AbstractVector{Tuple{Int64,Int64}}, s::D₄Fourier) = map(i -> RadiiPolynomial._findposition(i, s), u)
RadiiPolynomial._findposition(c::Colon, ::D₄Fourier) = c
RadiiPolynomial._findposition(i::Tuple{UnitRange{Int64},UnitRange{Int64}}, s::D₄Fourier) = _isomorphism_position_unit_range(i,s,RadiiPolynomial.order(s)[1])

# This checks if an index is valid (i.e in the reduced set).
RadiiPolynomial.__checkbounds_indices(α::Tuple{Int64,Int64}, s::D₄Fourier) = (0 ≤ α[2] ≤ α[1] ≤ order(s)[1])

#Other Methods
RadiiPolynomial.__checkbounds_indices(α::Tuple{UnitRange{Int64},Int64}, s::D₄Fourier) = ((0 ≤ α[2] ≤ sort(α[1])[1]) & (sort(α[1])[end] ≤ order(s)[1]))
RadiiPolynomial._findposition(i::Tuple{UnitRange{Int64},Int64}, s::D₄Fourier) = _isomorphism_position_unit_range(i,s,RadiiPolynomial.order(s)[1])

RadiiPolynomial.__checkbounds_indices(α::Tuple{Int64,UnitRange{Int64}}, s::D₄Fourier) = 0 ≤ sort(α[2])[end] ≤ α[1] ≤ order(s)[1]
RadiiPolynomial._findposition(i::Tuple{Int64,UnitRange{Int64}}, s::D₄Fourier) = _isomorphism_position_unit_range(i,s,RadiiPolynomial.order(s)[1])

function RadiiPolynomial.__checkbounds_indices(α::Tuple{UnitRange{Int64},UnitRange{Int64}}, s::D₄Fourier) 
    α₁ = α[1]
    α₂ = α[2]
    for β₁ ∈ α₁
        for β₂ ∈ α₂
            val = RadiiPolynomial.__checkbounds_indices((β₁,β₂),s)
            if val != true
                return false
            end
        end
    end
    return true
end

Base.convert(::Type{T}, s::T) where {T<:D₄Fourier} = s
Base.convert(::Type{D₄Fourier{T}}, s::D₄Fourier) where {T<:Real} =
    D₄Fourier{T}(RadiiPolynomial.order(s), convert(T, RadiiPolynomial.frequency(s)))

Base.promote_rule(::Type{T}, ::Type{T}) where {T<:D₄Fourier} = T
Base.promote_rule(::Type{D₄Fourier{T}}, ::Type{D₄Fourier{S}}) where {T<:Real,S<:Real} =
    D₄Fourier{promote_type(T, S)}

RadiiPolynomial._iscompatible(s₁::D₄Fourier, s₂::D₄Fourier) = RadiiPolynomial._iscompatible(RadiiPolynomial.desymmetrize(s₁), RadiiPolynomial.desymmetrize(s₂))

RadiiPolynomial._prettystring(s::D₄Fourier) = "D₄Fourier(" * string(RadiiPolynomial.order(s)[1]) * ", " * string(RadiiPolynomial.frequency(s)[1]) * ")"

RadiiPolynomial.dimension(s::D₄Fourier) = div((RadiiPolynomial.order(s)[1]+1)*(RadiiPolynomial.order(s)[1] + 2),2)

#Basic operations
RadiiPolynomial.image(::typeof(+), s₁::D₄Fourier, s₂::D₄Fourier) = D₄Fourier(RadiiPolynomial.image(+, RadiiPolynomial.desymmetrize(s₁), RadiiPolynomial.desymmetrize(s₂)))
RadiiPolynomial.image(::typeof(*), s₁::D₄Fourier, s₂::D₄Fourier) = D₄Fourier(RadiiPolynomial.image(*, RadiiPolynomial.desymmetrize(s₁), RadiiPolynomial.desymmetrize(s₂)))
RadiiPolynomial.image(::typeof(add_bar), s₁::D₄Fourier, s₂::D₄Fourier) = D₄Fourier(image(add_bar, RadiiPolynomial.desymmetrize(s₁), RadiiPolynomial.desymmetrize(s₂)))
RadiiPolynomial.image(::typeof(mul_bar), s₁::D₄Fourier, s₂::D₄Fourier) = D₄Fourier(image(mul_bar, RadiiPolynomial.desymmetrize(s₁), RadiiPolynomial.desymmetrize(s₂)))

# Convolution functions. Unlike the other sequence structures in RadiiPolynomial.jl,
# here we have double loops due to the fact that D₄ is purely a 2D structure.
function RadiiPolynomial._add_mul!(C, A, B, α, space_c::D₄Fourier, space_a::D₄Fourier, space_b::D₄Fourier)
    @inbounds order_c = RadiiPolynomial.order(space_c)[1]
    @inbounds for i₂ ∈ 0:order_c
        @inbounds for i₁ ∈ i₂:order_c
        RadiiPolynomial.__convolution!(C, A, B, α, order_c, space_a, space_b, (i₁,i₂))
        end
    end
    return C
end

function RadiiPolynomial.__convolution!(C, A, B, α, order_c, space_a::D₄Fourier, space_b::D₄Fourier, i)
    @inbounds order_a = RadiiPolynomial.order(space_a)[1]
    @inbounds order_b = RadiiPolynomial.order(space_b)[1]
    Cᵢ = zero(promote_type(eltype(A), eltype(B)))
    @inbounds @simd for j₁ ∈ max(i[1]-order_a, -order_b):min(i[1]+order_a, order_b)
        @inbounds for j₂ ∈ max(i[2]-order_a, -order_b):min(i[2]+order_a, order_b)
            tA = (max(abs(i[1]-j₁),abs(i[2]-j₂)),min(abs(i[1]-j₁),abs(i[2]-j₂)))
            tB = (max(abs(j₁),abs(j₂)),min(abs(j₁),abs(j₂)))
            Cᵢ += A[tA[1]+tA[2]*order_a - div(((tA[2]-2)^2 + 3*(tA[2]-2)),2)] * B[tB[1]+tB[2]*order_b - div(((tB[2]-2)^2 + 3*(tB[2]-2)),2)]
        end
    end
    C[i[1] + i[2]*order_c - div(((i[2]-2)^2 + 3*(i[2]-2)),2)] += Cᵢ * α
    return C
end

function _convolution!(C::AbstractArray{T,N}, A, B, α, current_space_c::D₄Fourier, current_space_a::D₄Fourier, current_space_b::D₄Fourier, remaining_space_c, remaining_space_a, remaining_space_b, i) where {T,N}
    order_a = RadiiPolynoimal.order(current_space_a[N])[1]
    order_b = RadiiPolynomial.order(current_space_b[N])[1]
    order_c = RadiiPolynomial.order(current_space_c[N])[1]
    @inbounds Cᵢ = selectdim(C, N, i[1]+i[2]*order_c- div(((i[2]-2)^2+3*(i[2]-2)),2))
    @inbounds @simd for j₁ ∈ max(i[1]-order_a, -order_b):min(i[1]+order_a, order_b)
        @inbounds for j₂ ∈ max(i[2]-order_a, -order_b):min(i[2]+order_a, order_b)
            tA = (max(abs(i[1]-j₁),abs(i[2]-j₂)),min(abs(i[1]-j₁),abs(i[2]-j₂)))
            tB = (max(abs(j₁),abs(j₂)),min(abs(j₁),abs(j₂)))
            _add_mul!(Cᵢ,
            selectdim(A, N, tA[1]+tA[2]*order_a - div(((tA[2]-2)^2 + 3*(tA[2]-2)),2)),
            selectdim(B, N, tB[1]+tB[2]*order_b - div(((tB[2]-2)^2 + 3*(tB[2]-2)),2)),
            α, remaining_space_c[N], remaining_space_a[N], remaining_space_b[N])
        end
    end
    return C
end


RadiiPolynomial._convolution_indices(s₁::D₄Fourier, s₂::D₄Fourier, i::Tuple{Int64,Int64}) =
    TensorIndices((max(i[1]-order(s₁)[1], -order(s₂)[1]):min(i[1]+order(s₁)[1], order(s₂)[1]),max(i[2]-order(s₁)[2], -order(s₂)[2]):min(i[2]+order(s₁)[2], order(s₂)[2])))

RadiiPolynomial._extract_valid_index(::D₄Fourier, i::Tuple{Int64,Int64}, j::Tuple{Int64,Int64}) = (max(abs(i[1]-j[1]),abs(i[2]-j[2])),min(abs(i[1]-j[1]),abs(i[2]-j[2])))
RadiiPolynomial._extract_valid_index(::D₄Fourier, i::Tuple{Int64,Int64}) = (max(abs(i[1]),abs(i[2])),min(abs(i[1]),abs(i[2])))

#Derivative
# Throws an error if one attempts to use the usual derivative operations of RadiiPolynomial.jl
RadiiPolynomial.image(𝒟::Derivative, s::D₄Fourier) = throw(ArgumentError("D₄Fourier is not closed under differentiation. The Laplacian is available for specific PDEs."))

# Laplacian 
# We code the Laplacian in the style of RadiiPolynomial.jl. That is, we (essentially)
# compute Derivative(2,0)*s + Derivative(0,2)*s and combine them rather than a direct
# approach. This is to be consistent with the methods defined in RadiiPolynomial.jl
# and avoid compatibility errors.
RadiiPolynomial.image(Δ::Laplacian, s::D₄Fourier) = s

RadiiPolynomial._coeftype(::Laplacian, ::D₄Fourier{T}, ::Type{S}) where {T,S} = typeof(zero(T)*0*zero(S))

function RadiiPolynomial._apply!(c::Sequence{<:D₄Fourier}, Δ::Laplacian, a)
    n = RadiiPolynomial.order(Δ)
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        order_c = RadiiPolynomial.order(c)[1]
        ω = one(eltype(a))*frequency(a)[1]
        @inbounds c[(0,0)] = zero(eltype(c))
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j₂ ∈ 0:order_c
            @inbounds for j₁ ∈ j₂:order_c
                iⁿωⁿjⁿ_real = iⁿ_real*(ω*j₁)^n + iⁿ_real*(ω*j₂)^n
                c[(j₁,j₂)] = iⁿωⁿjⁿ_real * a[(j₁,j₂)]
             end
        end
    end
    return c
end

function RadiiPolynomial._apply!(C::AbstractArray{T}, Δ::Laplacian, space::D₄Fourier, A) where {T}
    n = order(Δ)
    if n == 0
        C .= A
    else
        ord = order(space)[1]
        ω = one(eltype(A))*frequency(space)[1]
        @inbounds selectdim(C,1,1) .= zero(T)
        iⁿ_real = ifelse(n%4 < 2, 1, -1)
        @inbounds for j₂ ∈ 0:ord
            @inbounds for j₁ ∈ j₂:ord
                iⁿωⁿjⁿ_real = iⁿ_real*(ω*j₁)^n + iⁿ_real*(ω*j₂)^n
                selectdim(C,1,j₁ + j₂*ord - div(((j₂-2)^2 + 3*(j₂-2)),2)) .= iⁿωⁿjⁿ_real .* selectdim(A,1,j₁ + j₂*ord - div(((j₂-2)^2 + 3*(j₂-2)),2))
             end
        end
    end
    return C
end

function RadiiPolynomial._apply(Δ::Laplacian, space::D₄Fourier, ::Val{D}, A::AbstractArray{T,N}) where {D,T,N}
    n = order(Δ)
    CoefType = RadiiPolynomial._coeftype(Δ, space, T)
    if n == 0
        return convert(Array{CoefType,N},A)
    else
        C = Array{CoefType,N}(undef,size(A))
        ord = order(space)[1]
        ω = one(eltype(A))*frequency(space)[1]
        @inbounds selectdim(C,D,1) .= zero(CoefType)
        iⁿ_real = ifelse(n%4 < 2, 1, -1) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j₂ ∈ 0:ord
            @inbounds for j₁ ∈ j₂:ord
                iⁿωⁿjⁿ_real = iⁿ_real*(ω*j₁)^n + iⁿ_real*(ω*j₂)^n
                selectdim(C,D,j₁ + j₂*ord - div(((j₂-2)^2 + 3*(j₂-2)),2)) .= iⁿωⁿjⁿ_real .* selectdim(A,D,j₁ + j₂*ord - div(((j₂-2)^2 + 3*(j₂-2)),2))
             end
        end
    end
    return C
end

function _nzind_domain1(Δ::Laplacian, domain::D₄Fourier, codomain::D₄Fourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = (min(order(domain)[1], order(codomain)[1]),min(order(domain)[2], order(codomain)[2]))
    return ((order(Δ) > 0):ord[1],0:ord[2])
end

function _nzind_codomain1(Δ::Laplacian, domain::D₄Fourier, codomain::D₄Fourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = (min(order(domain)[1], order(codomain)[1]),min(order(domain)[2], order(codomain)[2]))
    return ((order(Δ) > 0):ord[1],0:ord[2])
end

function _nzind_domain2(Δ::Laplacian, domain::D₄Fourier, codomain::D₄Fourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = (min(order(domain)[1], order(codomain)[1]),min(order(domain)[2], order(codomain)[2]))
    return (0:ord[1],(order(Δ) > 0):ord[2])
end

function _nzind_codomain2(Δ::Laplacian, domain::D₄Fourier, codomain::D₄Fourier)
    ω₁ = frequency(domain)
    ω₂ = frequency(codomain)
    ω₁ == ω₂ || return throw(ArgumentError("frequencies must be equal: s₁ has frequency $ω₁, s₂ has frequency $ω₂"))
    ord = (min(order(domain)[1], order(codomain)[1]),min(order(domain)[2], order(codomain)[2]))
    return (0:ord[1],(order(Δ) > 0):ord[2])
end

function _nzval1(Δ::Laplacian, domain::D₄Fourier, ::D₄Fourier, ::Type{T}, i, j) where {T}
    n = order(Δ)
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (RadiiPolynomial.frequency(domain)[1]*j[1])^n
        return convert(T, ifelse(n%4 < 2, ωⁿjⁿ, -ωⁿjⁿ))
    end
end

function _nzval2(Δ::Laplacian, domain::D₄Fourier, ::D₄Fourier, ::Type{T}, i, j) where {T}
    n = order(Δ)
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (RadiiPolynomial.frequency(domain)[2]*j[2])^n
        return convert(T, ifelse(n%4 < 2, ωⁿjⁿ, -ωⁿjⁿ))
    end
end

function _projectD!(C::LinearOperator{D₄Fourier{Float64},D₄Fourier{Float64}}, Δ::Laplacian)
    domain_C = RadiiPolynomial.domain(C)
    codomain_C = RadiiPolynomial.codomain(C)
    CoefType = RadiiPolynomial.eltype(C)
    @inbounds for (α₁, β₁) ∈ RadiiPolynomial.zip(_nzind_codomain1(Δ, domain_C, codomain_C)[1], _nzind_domain1(Δ, domain_C, codomain_C)[1])
        @inbounds for (α₂, β₂) ∈ RadiiPolynomial.zip(_nzind_codomain1(Δ, domain_C, codomain_C)[2], _nzind_domain1(Δ, domain_C, codomain_C)[2])
            if (α₁ ≥ α₂) & (β₁ ≥ β₂)
                C[(α₁,α₂),(β₁,β₂)] = _nzval1(Δ, domain_C, codomain_C, CoefType, (α₁,α₂), (β₁,β₂))
            end
        end
    end
    @inbounds for (α₁, β₁) ∈ RadiiPolynomial.zip(_nzind_codomain2(Δ, domain_C, codomain_C)[1], _nzind_domain2(Δ, domain_C, codomain_C)[1])
        @inbounds for (α₂, β₂) ∈ RadiiPolynomial.zip(_nzind_codomain2(Δ, domain_C, codomain_C)[2], _nzind_domain2(Δ, domain_C, codomain_C)[2])
            if (α₁ ≥ α₂) & (β₁ ≥ β₂)
                C[(α₁,α₂),(β₁,β₂)] += _nzval2(Δ, domain_C, codomain_C, CoefType, (α₁,α₂), (β₁,β₂))
            end
        end
    end
    return C
end

function RadiiPolynomial._apply!(c::Sequence{D₄Fourier{Interval{Float64}}}, Δ::Laplacian, a)
    n = RadiiPolynomial.order(Δ)
    if n == 0
        coefficients(c) .= coefficients(a)
    else
        order_c = RadiiPolynomial.order(c)[1]
        ω = one(eltype(a))*frequency(a)[1]
        @inbounds c[(0,0)] = zero(eltype(c))
        iⁿ_real = interval(ifelse(n%4 < 2, 1, -1)) # (n%4 == 0) | (n%4 == 1)
        @inbounds for j₂ ∈ 0:order_c
            @inbounds for j₁ ∈ j₂:order_c
                iⁿωⁿjⁿ_real = iⁿ_real*(ω*interval(j₁))^interval(n) + iⁿ_real*(ω*interval(j₂))^interval(n)
                c[(j₁,j₂)] = iⁿωⁿjⁿ_real * a[(j₁,j₂)]
             end
        end
    end
    return c
end

function _nzval1(Δ::Laplacian, domain::D₄Fourier{Interval{Float64}}, ::D₄Fourier{Interval{Float64}}, ::Type{T}, i, j) where {T}
    n = order(Δ)
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (RadiiPolynomial.frequency(domain)[1]*interval(j[1]))^interval(n)
        return convert(T, ifelse(n%4 < 2, ωⁿjⁿ, -ωⁿjⁿ))
    end
end

function _nzval2(Δ::Laplacian, domain::D₄Fourier{Interval{Float64}}, ::D₄Fourier{Interval{Float64}}, ::Type{T}, i, j) where {T}
    n = order(Δ)
    if n == 0
        return one(T)
    else
        ωⁿjⁿ = (RadiiPolynomial.frequency(domain)[2]*interval(j[2]))^interval(n)
        return convert(T, ifelse(n%4 < 2, ωⁿjⁿ, -ωⁿjⁿ))
    end
end

function _projectD!(C::LinearOperator{D₄Fourier{Interval{Float64}},D₄Fourier{Interval{Float64}}}, Δ::Laplacian)
    domain_C = RadiiPolynomial.domain(C)
    codomain_C = RadiiPolynomial.codomain(C)
    CoefType = RadiiPolynomial.eltype(C)
    @inbounds for (α₁, β₁) ∈ RadiiPolynomial.zip(_nzind_codomain1(Δ, domain_C, codomain_C)[1], _nzind_domain1(Δ, domain_C, codomain_C)[1])
        @inbounds for (α₂, β₂) ∈ RadiiPolynomial.zip(_nzind_codomain1(Δ, domain_C, codomain_C)[2], _nzind_domain1(Δ, domain_C, codomain_C)[2])
            if (α₁ ≥ α₂) & (β₁ ≥ β₂)
                C[(α₁,α₂),(β₁,β₂)] = _nzval1(Δ, domain_C, codomain_C, CoefType, (α₁,α₂), (β₁,β₂))
            end
        end
    end
    @inbounds for (α₁, β₁) ∈ RadiiPolynomial.zip(_nzind_codomain2(Δ, domain_C, codomain_C)[1], _nzind_domain2(Δ, domain_C, codomain_C)[1])
        @inbounds for (α₂, β₂) ∈ RadiiPolynomial.zip(_nzind_codomain2(Δ, domain_C, codomain_C)[2], _nzind_domain2(Δ, domain_C, codomain_C)[2])
            if (α₁ ≥ α₂) & (β₁ ≥ β₂)
                C[(α₁,α₂),(β₁,β₂)] += _nzval2(Δ, domain_C, codomain_C, CoefType, (α₁,α₂), (β₁,β₂))
            end
        end
    end
    return C
end

# Method to create a project of a given order of the Laplacian operator.
# Note that since we compute a full Laplacian rather than separate derivatives,
# the result is sparse.
function RadiiPolynomial.project(Δ::Laplacian, domain::D₄Fourier, codomain::D₄Fourier, ::Type{T}=RadiiPolynomial._coeftype(Δ, domain, Float64)) where {T}
    image_domain = RadiiPolynomial.image(Δ, domain)
    RadiiPolynomial._iscompatible(image_domain, codomain) || return throw(ArgumentError("spaces must be compatible: image of domain under $Δ is $image_domain, codomain is $codomain"))
    ind_domain = RadiiPolynomial._findposition(_nzind_domain1(Δ, domain, codomain),domain)
    ind_codomain = RadiiPolynomial._findposition(_nzind_codomain1(Δ, domain, codomain),codomain)
    C = RadiiPolynomial.LinearOperator(domain, codomain, SparseArrays.sparse(ind_codomain, ind_domain, zeros(T, length(ind_domain)), RadiiPolynomial.dimension(codomain), RadiiPolynomial.dimension(domain)))
    _projectD!(C, Δ)
    return C
end


# Multiplication
# Methods to create a multiplication operator for D₄Fourier. The process is
# the same as all current sequence structures in RadiiPolynomial.jl except that
# we again have double loops.
RadiiPolynomial._mult_domain_indices(s::D₄Fourier) = TensorIndices((-order(s)[1]:order(s)[1],-order(s)[2]:order(s)[2]))#RadiiPolynomial._mult_domain_indices(Chebyshev(order(s)[1]) ⊗ Chebyshev(order(s)[2]))
RadiiPolynomial._isvalid(s::D₄Fourier, i::Tuple{Int64,Int64}, j::Tuple{Int64,Int64}) =  RadiiPolynomial._isvalid(Chebyshev(order(s)[1]) ⊗ Chebyshev(order(s)[2]), i, j)

function RadiiPolynomial._project!(C::LinearOperator{<:D₄Fourier,<:D₄Fourier}, ℳ::Multiplication)
    ord_cod = RadiiPolynomial.order(RadiiPolynomial.codomain(C))[1]
    ord_d = RadiiPolynomial.order(RadiiPolynomial.domain(C))[1]
    𝕄 = RadiiPolynomial.sequence(ℳ)
    ord_M = RadiiPolynomial.order(RadiiPolynomial.space(𝕄))[1]
    @inbounds @simd for β₁ ∈ -ord_d:ord_d
            @inbounds @simd for β₂ ∈ -ord_d:ord_d
                        @inbounds for α₂ ∈ 0:ord_cod
                            @inbounds for α₁ ∈ α₂:ord_cod
                                if (abs(α₁-β₁) ≤ ord_M) & (abs(α₂-β₂) ≤ ord_M)
                                    C[(α₁,α₂),(max(abs(β₁),abs(β₂)),min(abs(β₁),abs(β₂)))] += 𝕄[(max(abs(α₁ - β₁),abs(α₂ - β₂)),min(abs(α₁ - β₁),abs(α₂ - β₂)))]
                                end
                            end
                        end
                    end
        end
    return C
end

#Norms
# When computing the 1 and 2 norms, we must multiply by the orbit sizes to recover
# the norm of a full Fourier series. Since we know the reduced set has a specific form,
# we are able to do separate, smaller loops to avoid an if condition to check the
# orbit size of a given index. This implementation was also used in the inf norm, 
# but would not be necessary there. It is purely for consistency. 

#ℓ¹
function RadiiPolynomial._apply(::Ell1{IdentityWeight}, space::D₄Fourier, A::AbstractVector)
    ord = order(space)[1]
    V = zero(eltype(A))
    V += abs(A[1])
    for k₁ = 1:ord
        V += (4abs(A[k₁ + k₁*ord - div(((k₁-2)^2+3*(k₁-2)),2)]) + 4abs(A[k₁ + 1]))
    end
    for k₂ = 1:ord
        for k₁ = (k₂+1):ord
            V += 8abs(A[k₁ + k₂*ord - div(((k₂-2)^2+3*(k₂-2)),2)])
        end
    end
    return V
end

function RadiiPolynomial._apply(::Ell1{IdentityWeight}, space::D₄Fourier{Interval{Float64}}, A::AbstractVector)
    ord = order(space)[1]
    V = zero(eltype(A))
    V += abs(A[1])
    for k₁ = 1:ord
        V += (interval(4)*abs(A[k₁ + k₁*ord - div(((k₁-2)^2+3*(k₁-2)),2)]) + interval(4)*abs(A[k₁ + 1]))
    end
    for k₂ = 1:ord
        for k₁ = (k₂+1):ord
            V += interval(8)*abs(A[k₁ + k₂*ord - div(((k₂-2)^2+3*(k₂-2)),2)])
        end
    end
    return V
end

function RadiiPolynomial._apply_dual(::Ell1{IdentityWeight}, space::D₄Fourier, A::AbstractVector)
    ord = order(space)[1]
    V = zero(eltype(A))
    V = max(V,abs(A[1]))
    for k₁ = 1:ord
        V = max(V,abs(A[k₁ + k₁*ord - div(((k₁-2)^2+3*(k₁-2)),2)]),abs(A[k₁ + 1]))
    end
    for k₂ = 1:ord
        for k₁ = (k₂+1):ord
            V = max(V,abs(A[k₁ + k₂*ord - div(((k₂-2)^2+3*(k₂-2)),2)]))
        end
    end
    return V
end

#ℓ²
function RadiiPolynomial._apply(::Ell2{IdentityWeight}, space::D₄Fourier, A::AbstractVector)
    ord = order(space)[1]
    V = zero(eltype(A))
    V += abs2(A[1])
    for k₁ = 1:ord
        V += (4abs2(A[k₁ + k₁*ord - div(((k₁-2)^2+3*(k₁-2)),2)]) + 4abs2(A[k₁ + 1]))
    end
    for k₂ = 1:ord
        for k₁ = (k₂+1):ord
            V += 8abs2(A[k₁ + k₂*ord - div(((k₂-2)^2+3*(k₂-2)),2)])
        end
    end
    return sqrt(V)
end

function RadiiPolynomial._apply(::Ell2{IdentityWeight}, space::D₄Fourier{Interval{Float64}}, A::AbstractVector)
    ord = order(space)[1]
    V = zero(eltype(A))
    V += abs2(A[1])
    for k₁ = 1:ord
        V += (interval(4)*abs2(A[k₁ + k₁*ord - div(((k₁-2)^2+3*(k₁-2)),2)]) + interval(4)*abs2(A[k₁ + 1]))
    end
    for k₂ = 1:ord
        for k₁ = (k₂+1):ord
            V += interval(8)*abs2(A[k₁ + k₂*ord - div(((k₂-2)^2+3*(k₂-2)),2)])
        end
    end
    return sqrt(V)
end

function RadiiPolynomial._apply_dual(::Ell2{IdentityWeight}, space::D₄Fourier, A::AbstractVector)
    ord = order(space)[1]
    V = zero(eltype(A))
    V += abs2(A[1])
    for k₁ = 1:ord
        V += (abs2(A[k₁ + k₁*ord - div(((k₁-2)^2+3*(k₁-2)),2)])/4 + abs2(A[k₁ + 1])/4)
    end
    for k₂ = 1:ord
        for k₁ = (k₂+1):ord
            V += abs2(A[k₁ + k₂*ord - div(((k₂-2)^2+3*(k₂-2)),2)])/8
        end
    end
    return sqrt(V)
end

function RadiiPolynomial._apply_dual(::Ell2{IdentityWeight}, space::D₄Fourier{Interval{Float64}}, A::AbstractVector)
    ord = order(space)[1]
    V = zero(eltype(A))
    V += abs2(A[1])
    for k₁ = 1:ord
        V += (abs2(A[k₁ + k₁*ord - div(((k₁-2)^2+3*(k₁-2)),2)])/interval(4) + abs2(A[k₁ + 1])/interval(4))
    end
    for k₂ = 1:ord
        for k₁ = (k₂+1):ord
            V += abs2(A[k₁ + k₂*ord - div(((k₂-2)^2+3*(k₂-2)),2)])/interval(8)
        end
    end
    return sqrt(V)
end


#ℓ∞
function RadiiPolynomial._apply(::EllInf{IdentityWeight}, space::D₄Fourier, A::AbstractVector)
    ord = order(space)[1]
    V = zero(eltype(A))
    V = max(V,abs(A[1]))
    for k₁ = 1:ord
        V = max(V,abs(A[k₁ + k₁*ord - div(((k₁-2)^2+3*(k₁-2)),2)]),abs(A[k₁ + 1]))
    end
    for k₂ = 1:ord
        for k₁ = (k₂+1):ord
            V = max(V,abs(A[k₁ + k₂*ord - div(((k₂-2)^2+3*(k₂-2)),2)]))
        end
    end
    return V
end

function RadiiPolynomial._apply_dual(::EllInf{IdentityWeight}, space::D₄Fourier, A::AbstractVector)
    ord = order(space)[1]
    V = zero(eltype(A))
    V += abs(A[1])
    for k₁ = 1:ord
        V += (4abs(A[k₁ + k₁*ord - div(((k₁-2)^2+3*(k₁-2)),2)]) + 4abs(A[k₁ + 1]))
    end
    for k₂ = 1:ord
        for k₁ = (k₂+1):ord
            V += 8abs(A[k₁ + k₂*ord - div(((k₂-2)^2+3*(k₂-2)),2)])
        end
    end
    return V
end

function RadiiPolynomial._apply_dual(::EllInf{IdentityWeight}, space::D₄Fourier{Interval{Float64}}, A::AbstractVector)
    ord = order(space)[1]
    V = zero(eltype(A))
    V += abs(A[1])
    for k₁ = 1:ord
        V += (interval(4)*abs(A[k₁ + k₁*ord - div(((k₁-2)^2+3*(k₁-2)),2)]) + interval(4)*abs(A[k₁ + 1]))
    end
    for k₂ = 1:ord
        for k₁ = (k₂+1):ord
            V += interval(8)*abs(A[k₁ + k₂*ord - div(((k₂-2)^2+3*(k₂-2)),2)])
        end
    end
    return V
end

# Evaluation
# Similar to norms, we again compute separate, smaller loops to avoid an if condition
# during evaluation of D₄Fourier sequences.
RadiiPolynomial._memo(::D₄Fourier, ::Type) = nothing

RadiiPolynomial.image(::Evaluation{Nothing}, s::D₄Fourier) = s
RadiiPolynomial.image(::Evaluation, s::D₄Fourier) = D₄Fourier(0, RadiiPolynomial.frequency(s)[1])

RadiiPolynomial._coeftype(::Evaluation{Nothing}, ::D₄Fourier, ::Type{T}) where {T} = T
RadiiPolynomial._coeftype(::Evaluation{T}, s::D₄Fourier, ::Type{S}) where {T,S} =
    RadiiPolynomial.promote_type(typeof(cos(RadiiPolynomial.frequency(s)[1]*zero(Float64))*cos(RadiiPolynomial.frequency(s)[2]*zero(Float64))), S)

function RadiiPolynomial._apply!(c, ::Evaluation{Nothing}, a::Sequence{<:D₄Fourier})
    coefficients(c) .= coefficients(a)
    return c
end

function RadiiPolynomial._apply!(c, ℰ::Evaluation, a::Sequence{<:D₄Fourier})
    x = RadiiPolynomial.value(ℰ)
    ord = RadiiPolynomial.order(a)[1]
    c .= 0
    if (ord > 0)
        if (iszero(x[1])) & (iszero(x[2]))
            c[(0,0)] += a[(0,0)]
            for k₁ = 1:ord
                c[(0,0)] += (a[(k₁,0)] + a[(k₁,k₁)])*4
            end
            for k₂ = 1:ord
                for k₁ = (k₂+1):ord
                    c[(0,0)] += a[(k₁,k₂)]*8
                end
            end
            c[(0,0)] = c[(0,0)]
        else
            @inbounds f = RadiiPolynomial.frequency(a)[1]
            @inbounds ωx1 = f*x[1]
            @inbounds ωx2 = f*x[2]
            c[(0,0)] += a[(0,0)]
            for k₁ = 1:ord
                ck₁ωx1 = cos(k₁*ωx1)
                ck₁ωx2 = cos(k₁*ωx2)
                c[(0,0)] += (4a[(k₁,k₁)]*ck₁ωx1*ck₁ωx2 + 2a[(k₁,0)]*(ck₁ωx1 + ck₁ωx2))
            end
            for k₂ = 1:ord
                for k₁ = (k₂+1):ord
                    ck₁ωx1 = cos(k₁*ωx1)
                    ck₁ωx2 = cos(k₁*ωx2)
                    ck₂ωx1 = cos(k₂*ωx1)
                    ck₂ωx2 = cos(k₂*ωx2)
                    c[(0,0)] += 4a[(k₁,k₂)]*(ck₁ωx1*ck₂ωx2 + ck₂ωx1*ck₁ωx2)
                end
            end
        end
    end
    return c
end



function RadiiPolynomial._apply!(c, ℰ::Evaluation, a::Sequence{D₄Fourier{Interval{Float64}}})
    x = RadiiPolynomial.value(ℰ)
    ord = RadiiPolynomial.order(a)[1]
    c .= interval(0)
    if (ord > 0)
        if (iszero(x[1])) & (iszero(x[2]))
            c[(0,0)] += a[(0,0)]
            for k₁ = 1:ord
                c[(0,0)] += (a[(k₁,0)] + a[(k₁,k₁)])*interval(4)
            end
            for k₂ = 1:ord
                for k₁ = (k₂+1):ord
                    c[(0,0)] += a[(k₁,k₂)]*interval(8)
                end
            end
            c[(0,0)] = c[(0,0)]
        else
            @inbounds f = RadiiPolynomial.frequency(a)[1]
            @inbounds ωx1 = f*interval(x[1])
            @inbounds ωx2 = f*interval(x[2])
            c[(0,0)] += a[(0,0)]
            for k₁ = 1:ord
                ck₁ωx1 = cos(interval(k₁)*ωx1)
                ck₁ωx2 = cos(interval(k₁)*ωx2)
                c[(0,0)] += (interval(4)*a[(k₁,k₁)]*ck₁ωx1*ck₁ωx2 + interval(2)*a[(k₁,0)]*(ck₁ωx1 + ck₁ωx2))
            end
            for k₂ = 1:ord
                for k₁ = (k₂+1):ord
                    ck₁ωx1 = cos(interval(k₁)*ωx1)
                    ck₁ωx2 = cos(interval(k₁)*ωx2)
                    ck₂ωx1 = cos(interval(k₂)*ωx1)
                    ck₂ωx2 = cos(interval(k₂)*ωx2)
                    c[(0,0)] += interval(4)*a[(k₁,k₂)]*(ck₁ωx1*ck₂ωx2 + ck₂ωx1*ck₁ωx2)
                end
            end
        end
    end
    return c
end
