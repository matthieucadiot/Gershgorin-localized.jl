function exp2cos(N)
  
    d = sqrt(2)*(ones((N+1)))
  
    d[1] =1;
  
    return d
  end
  
  
  function exp2sin(N)
  
    d = sqrt(2)*(ones((N)))
  
    return d
  end
  
  
  
  function two_norm_inverse_l(d,ν,c,T)
  
      S = interval(0)
      N = floor(mid(100*d/π))
      for n=1:N
        mT = sqrt(tanh(n*π/d)*(1+T*(n*π/d)^2)/(n*π/d))
        S = S + 1/((mT-c)^2*(1+ν*(n*π/d)^2)^2)
      end
      S = S + 1/((1-c)^2) + d^4/(3*π^4*ν^2*N^3*sqrt(tanh((N+1)*π/d)*(1+T*((N+1)*π/d)^2)/(N*π/d))^2)
      S = S/d 
      return sqrt(S)
  
    end
  
  
  
  function computation_Cλ(c,a,σ0,T)
  
    setprecision(100)
  
  ##### Construction of the needed constants
  # Construction of ξ0 
    Ca = (1 + abs(cos(2*a)))/(1-abs(cos(2*a)))
  
  if inf(T)>0
  ξ0 = maximum([interval(1) 1/sqrt(T) c^2*Ca*4/T])
  ξ0 = maximum([3*T/(2*tanh(ξ0)) 2*c^2/(T*tanh(ξ0))])
  else
  ξ0 = interval(1)
  end
  
  if inf(T)>0
  K1 = 2*ξ0/(π*σ0) + 2*sqrt(ξ0)*(1+abs(c))/(π*σ0*sqrt(T)) + 2*(1/(3*T) + abs(c)/(4*T^(3/2)) + 2*c^2/T )/(π*sqrt(tanh(ξ0)*T*ξ0)) + abs(c)/(T*π)*(2+3*log(ξ0)) + 1/sqrt(2π*T) 
  K2 = Ca*abs(ξ0+im*a)/( 2π*σ0^2*(1-T*a^2)^2)*( 1/a^2 + T + Ca/a + Ca*T*abs(ξ0+im*a) ) + 2*Ca^2/π*( 2*(1+T)/sqrt(T*ξ0) + (2+a)/2*exp(-2ξ0))
  C2 = maximum([K2 K1*exp(a)])  
  else
  K1 = (2+4*exp(-2*ξ0))/(minimum([1 abs(c)^3])*π*sqrt(ξ0)*σ0) + 1/(π*σ0*abs(c)) + 2/(π*c^2) + 1/(π*abs(c)^3)*(2+3*log(3*ξ0)) + 1/(c^2*sqrt(2*π))
  K2 = Ca^(1/4)/π*( 1/(2*σ0^2)*( 2 + a^(-3/2)) + 1/(4*σ0^2*(1-abs(cos(2a)))^2*sqrt(a)) )
  C2 = maximum([K2 K1*exp(a)]) 
  end
  
  return C2
  
  end
  
  



function computation_Zu(V0,a,d,ab,db,δ0) 

      #computation of the supremum of Zu for all λ ∈ (0,λmin)
  # We first compute the supremum of Cλ
  Cλ = computation_Cλ(c-δ0,a,σ0,T)

  # coefficients of the characteristic function
  char = Sequence(Fourier(2*N,π/d),interval.(big.(zeros(4*N+1))))
  for n = [-2N:-1;1:2N]  
    char[n] = real((-1)^n*(exp(im*interval(big(n))*π/db)-1)/(ExactReal(2)*im*interval(big(n))*π))
  end
  char[0] = ExactReal(1)/(interval(big(2))*db)
  
  # Coefficients of cosh(2ax)
  E = Sequence(CosFourier(2*N,π/d),interval.(big.(zeros(2*N+1))))
  for n = 0:2N  
    E[n] = ab/db*ExactReal((-1)^n)*(ExactReal(1)-exp(-ExactReal(4)*ab*db))/(ExactReal(4)*ab^2 + (ExactReal(n)*π/db)^2 )
  end
  
  C1d = exp(-ExactReal(2)*ab*db)*( ExactReal(2)*sqrt(π)/(sqrt(ExactReal(4)*ab*db)*(ExactReal(1)-exp(-ExactReal(2)*ab*db))) + ExactReal(4)/(interval(big(1))-exp(-interval(big(2))*ab*db)) )

  k = (-N:N)*π/db
  fourier_f = Fourier(N,π/d)
  V0f = [reverse(V0[1:N]); V0[0] ; V0[1:N]].*k ; V0f = Sequence(fourier_f,V0f)

  Elog =  abs( (coefficients(project(V0f*char,fourier_f))')*coefficients(V0f) )
  𝒵u0 = abs( (coefficients(project(V0*E,CosFourier(N,π/d)))')*coefficients(V0) )
  
  #Computation of the supremum of 𝒵λ for all λ ≤ 0
  𝒵u1 = ExactReal(8)*Cλ^2*(d*𝒵u0/a+ ExactReal(4)*log(interval(2))*Elog)
  𝒵u2 = ExactReal(8)*Cλ^2*(d*𝒵u0*(interval(1)/a+C1d) + ExactReal(4)*log(interval(2))*Elog )
  return sqrt(𝒵u1),sqrt(𝒵u2)
  
end






function computation_bounds(U0,c,T,r0,δ0,σ0,t,d,SS,D1,D2,𝒵u1, 𝒵u2)


  σ0 = ExactReal(1)-c
  # Construction of the operator L and its inverse. Actually, we only need to build their diagonal
  D² = project(Derivative(2), SS, SS,Interval{Float64})

  d2 = diag(coefficients(D²))  ;   dd = sqrt.(-d2) ; d0=dd;  d0[1] = interval(1);
  dL = tanh.(dd).*(ones(length(dd))+T*dd.^2)./d0 ;  dL[1] = interval(1) ; dL = sqrt.(dL)
  L = LinearOperator(SS,SS,Diagonal(dL)) - c*I

  DF = L +  ExactReal(2)*project(Multiplication(U0),SS, SS,Interval{Float64}) ;
  # display("DG created")

  D,P =  eigen(coefficients(mid.(D1).*mid.(DF).*mid.(D2)'))

  #### We compute an approximate inverse Pinv for P. Computing the defect nP, we can use a Neumann series argument to compute the real inverse of P as P^{-1} = sum_k (I - Pinv*P)^k Pinv. We propagate the error given by nP in the bounds below. In practice, nP is very small and will not affect the quality of the bounds 
  P = interval.(mid.(D2).*P.*mid.(D1)') ; Pinv = interval.(inv(mid.(P))) ; nP = opnorm(LinearOperator(I - D1.*_matprod(Pinv,P).*D2'),2)
  norm_P = spec_norm(D1.*P.*D2') ; norm_Pinv = spec_norm(D1.*Pinv.*D2')  

  # display("norm of P")
  # display(norm_P)

  D = -_matprod(_matprod(Pinv,coefficients(DF)),P)  ; DF = Nothing
  S = diag(D) ; R = D - Diagonal(S) 

  St = S .+ t ;  Stinv = ExactReal(1) ./ St 

  ####### Computation of bounds Z1i ################
  
  ####### diagonal part of DG(U0) is given by V0_full[(0,0)]
  # display("values of the Z1i")

  lN = sqrt(tanh(interval(N+1)*π/d)*(interval(1)+T*(interval(N+1)^2*π^2/d^2))/(interval(N+1)*π/d))

  Z12 = interval(1)/(abs(-interval(2)*U0[0] + lN -δ0))*interval(2)*norm(U0-U0[0],1)
  # display("Z12")
  # display(Z12)


  Z13 = spec_norm((D1.*Stinv).*R.*D2')*(ExactReal(1)+ nP*norm_Pinv/(ExactReal(1)-nP))
    #  display("Z13")
    #  display(Z13)

     N1 = 800 ; V0_p = project(U0,CosFourier(N1, π/d)) ; V02 = V0_p*V0_p
     DG = project(Multiplication(V0_p),SS, SS,Interval{Float64})
     M1 = coefficients(project(Multiplication(V02),SS, SS, Interval{Float64})) - _matprod(coefficients(DG),coefficients(DG))
     Z14 = ExactReal(4)*(sqrt(spec_norm((D1.*Stinv).*_matprod(_matprod(Pinv,M1),Pinv').*((Stinv.*D2)')))*(ExactReal(1)+ nP*norm_Pinv/(ExactReal(1)-nP) + norm_Pinv*norm(V0_p-U0,1)*maximum(Stinv)))
    #  display("Z14")
    #  display(Z14)


     Z11 = ExactReal(1)/(abs(lN -δ0))*sqrt(spec_norm(D1.*_matprod(_matprod(P',M1),P).*D2'))
    #  display("Z11")
    #  display(Z11)


  ###### Computation of the bounds 𝒞1*r0 an 𝒞2*r0

  norm_SPL = spec_norm((D1.*Stinv).*Pinv.*(diag(coefficients(L)-δ0*I).*D2)')*(ExactReal(1)+ nP*norm_Pinv/(ExactReal(1)-nP))
  norml = interval(1)/(interval(4)*sqrt(T)*σ0)

  𝒞1 = interval(2)/(σ0-δ0)*norml*r0 
  𝒞2 = norm_SPL*interval(2)/(σ0-δ0)*norml*r0

  # display("values of the 𝒞i")
  # display(𝒞1)
  # display(𝒞2)

  ########## Computation of the bound 𝒵u3

  𝒵u3 = norm_SPL*𝒵u2

  # display("value of 𝒵u3")
  # display(𝒵u3)

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