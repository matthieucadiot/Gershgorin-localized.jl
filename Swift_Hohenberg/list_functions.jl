
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


function computation_b(a,d)
    x = 2*a*d
    b01 = (x+ 1 + exp(-x))/(a^2) + exp(-x)*(4*d + exp(-x)/a) + (2*exp(1)+1)/(a*(1-exp(x/2)))*(2*d+ (1+exp(-x))/a ) + (2*exp(1) +1)^2/(a^2*(1-exp(-x/2))^2)
    b02 = 2/a*( (1+exp(-x))/a + 2*d + exp(-x)*(4*d + exp(-x)/a) + (2*exp(1)+1)/(a*(1-exp(-x/2))) )

    b0 = 4*b01 + b02

    b10 = 2*d + (1+exp(-x))/(2*a) + (2*d + (1+exp(-x))/(2*a))*1/(1-exp(-x)) + (4*exp(1) + 1 + exp(-x))/(2*a*(1-exp(-x/2))^2)
    b11 = (2*d + 1/a)*8/(a*(1-exp(-x)))

    b1 = 8*b10*(2*d + 1/(2*a)) + b11

    b2 = 2/a*( (1+exp(-x))/a  + 2*d + exp(-x)*(4*d + exp(-x)/a) + (2*exp(1) + exp(-x))/(a*(1-exp(-x/2))))

    return b0, b1, b2
end




############################# Functions for stability ##############################




function computation_Zu(V0,N0,C0)

    abig = ExactReal(0.5)*sqrt(-ExactReal(1)+sqrt(ExactReal(1)+μbig-δ0big)) 
    
    fourierE = CosFourier(2*N0, π/d)⊗CosFourier(2*N0, π/d)
    E1 = Sequence(fourierE, interval.(big.(zeros((2*N0+1)^2))))
    E2 = Sequence(fourierE, interval.(big.(zeros((2*N0+1)^2))))
    E12 = Sequence(fourierE, interval.(big.(zeros((2*N0+1)^2))))
    m1 =2;m2=2;m3=2;
    
    for n1 = 1:m1*N0
        for n2 = 1:m1*N0
            if (n1 <= m3*N0)&&(n2 <= m3*N0)
                E12[(n1,n2)] = real(ExactReal((-1)^n1)*( (ExactReal(1)-exp(-ExactReal(4)*abig*dbig))/(ExactReal(2)*abig-im*ExactReal(n1)*π/dbig) ))*real(ExactReal((-1)^n2)*( (ExactReal(1)-exp(-ExactReal(4)*abig*dbig))/(ExactReal(2)*abig-im*ExactReal(n2)*π/dbig) ))  
            end 
        end
        E1[(n1,0)] = real(ExactReal((-1)^n1)*( (ExactReal(1)-exp(-ExactReal(4)*abig*dbig))/(ExactReal(2)*abig-im*ExactReal(n1)*π/dbig) ))*(ExactReal(1)-exp(-ExactReal(4)*abig*dbig))
        
        if (n1 <= m3*N0)
        E12[(n1,0)] = real(ExactReal((-1)^n1)*( (ExactReal(1)-exp(-ExactReal(4)*abig*dbig))/(ExactReal(2)*abig-im*ExactReal(n1)*π/dbig) ))*(ExactReal(1)-exp(-ExactReal(4)*abig*dbig))/(ExactReal(2)*abig) 
        end
    end
    
    for n2 = 1:m1*N0
        E2[(0,n2)] = real(ExactReal((-1)^n2)*( (ExactReal(1)-exp(-ExactReal(4)*abig*dbig))/(ExactReal(2)*abig-im*ExactReal(n2)*π/dbig) ))*(ExactReal(1)-exp(-ExactReal(4)*abig*dbig))
        
        if (n2 <= m3*N0)
        E12[(0,n2)] = real(ExactReal((-1)^n2)*( (ExactReal(1)-exp(-ExactReal(4)*abig*dbig))/(ExactReal(2)*abig-im*ExactReal(n2)*π/dbig) ))*(ExactReal(1)-exp(-ExactReal(4)*abig*dbig))/(ExactReal(2)*abig) 
        end
    end
    
    E1[(0,0)] = ExactReal(1)/(ExactReal(2)*abig)*(ExactReal(1)-exp(-ExactReal(4)*abig*dbig)); E1 = E1/(ExactReal(2)*dbig) ;
    E2[(0,0)] = ExactReal(1)/(ExactReal(2)*abig)*(ExactReal(1)-exp(-ExactReal(4)*abig*dbig)); E2 = E2/(ExactReal(2)*dbig) ; 
    E12[(0,0)] = ExactReal(1)/(ExactReal(4)*abig^2)*(ExactReal(1)-exp(-ExactReal(4)*abig*dbig)) ; E12 = E12/(ExactReal(4)*dbig^2)
    
    D12 = convert(Vector{Interval{Float64}},exp2cos(N0))

    V = project((E1+E2)*V0,CosFourier(N0, π/d)⊗CosFourier(N0, π/d))
    display("computation of convolution done")
    nV = abs(coefficients(D12.*V0)'*coefficients(D12.*V)); 
    
    Zu1 = C0^2/a^2*ExactReal(4)*d^2*nV
    display("value Zu1")
    display(sqrt(Zu1))
    
    
    # # ########################### Zu2 bound #####################################################
    
    b0, b1, b2 = computation_b(a,d)

    ### normalization factor which can sometimes be useful. By default it is chosen as 1.
    b_normal = ExactReal(1) ;

    V = project((exp(-ExactReal(2)*a*d)*b0*E1 + b1*E12 + exp(-ExactReal(2)*a*d)*b2*E2)*(b_normal*V0),CosFourier(N0, π/d)⊗CosFourier(N0, π/d))
    display("computation of convolution done")
    nV2 = abs(coefficients(D12.*V0)'*coefficients(D12.*V)) ; D12 = Nothing
     
    Zu2 = Zu1+nV2*ExactReal(4)*d^2/b_normal
    display("value Zu2")
    display(sqrt(Zu2))

    return sqrt(Zu1), sqrt(Zu2)

end






function computation_bounds(U0,V0,r0,μ,δ0,t,d,ν1,ν2,SS,D1,D2,𝒵u1,𝒵u2)

     # Construction of the operator L
     ∂1 = project(Derivative((2,0)), SS, SS,Interval{Float64})
     ∂2 = project(Derivative((0,2)), SS, SS,Interval{Float64})
     Δ = copy(∂1) ;  radd!(Δ,∂2) ; 
 
     ∂1 = Nothing ; ∂2 = Nothing ; 
 
     L = LinearOperator(SS,SS,Diagonal((diag(coefficients(Δ+I))).^2)) ;  Δ = Nothing
     L = L + μ*I
 
     DG = project(Multiplication(V0),SS, SS,Interval{Float64})
     DF = L +  DG ;
     display("DG created")
 
     # DF = L + DG ; L = Nothing ; DG = Nothing 
     D,P =  eigen(coefficients(mid.(D1).*mid.(DF).*mid.(D2)'))

     #### We compute an approximate inverse Pinv for P. Computing the defect nP, we can use a Neumann series argument to compute the real inverse of P as P^{-1} = sum_k (I - Pinv*P)^k Pinv. We propagate the error given by nP in the bounds below. In practice, nP is very small and will not affect the quality of the bounds 
     P = interval.(mid.(D2).*P.*mid.(D1)') ; Pinv = interval.(inv(mid.(P))) ; nP = opnorm(LinearOperator(I - D1.*_matprod(Pinv,P).*D2'),2)
     norm_P = spec_norm(D1.*P.*D2') ; norm_Pinv = spec_norm(D1.*Pinv.*D2')  

     display("norm of P")
     display(norm_P)

     D = -_matprod(_matprod(Pinv,coefficients(DF)),P)  ; DF = Nothing
     S = diag(D) ; R = D - Diagonal(S) 
 
     St = S .+ t ;  Stinv = ExactReal(1) ./ St 
 
     ####### Computation of bounds Z1i ################
     
     ####### diagonal part of DG(U0) is given by V0[(0,0)]
     display("values of the Z1i")
 
     Z12 = ExactReal(1)/(abs(V0[(0,0)] + μ + (1-(π/d*(N+1))^2)^2 -δ0))*norm(V0-V0[(0,0)],1)
     display("Z12")
     display(Z12)

     Z13 = spec_norm((D1.*Stinv).*R.*D2')*(ExactReal(1)+ nP*norm_Pinv/(ExactReal(1)-nP))
     display("Z13")
     display(Z13)

     N1 = 100 ; V0_p = project(V0,CosFourier(N1, π/d)⊗CosFourier(N1, π/d)) ; V02 = V0_p*V0_p
     DG = project(Multiplication(V0_p),SS, SS,Interval{Float64})
     M1 = coefficients(project(Multiplication(V02),SS, SS, Interval{Float64})) - _matprod(coefficients(DG),coefficients(DG))
     Z14 = sqrt(spec_norm((D1.*Stinv).*_matprod(_matprod(Pinv,M1),Pinv').*((Stinv.*D2)')))*(ExactReal(1)+ nP*norm_Pinv/(ExactReal(1)-nP) + norm_Pinv*norm(V0_p-V0,1)*maximum(Stinv))
     display("Z14")
     display(Z14)

     Z11 = ExactReal(1)/(abs(μ + (ExactReal(1)-(π/d*ExactReal(N+1))^2)^2 -δ0))*sqrt(spec_norm(D1.*_matprod(_matprod(P',M1),P).*D2'))
     display("Z11")
     display(Z11)
 
 
     ###### Computation of the bounds 𝒞1*r0 an 𝒞2*r0
 
     norm_SPL = spec_norm((D1.*Stinv).*Pinv.*(diag(coefficients(L)-δ0*I).*D2)')*(ExactReal(1)+ nP*norm_Pinv/(ExactReal(1)-nP))
 
     display("value norm SPL")
     display(norm_SPL)
 
     𝒞1 = ExactReal(1)/(μ-δ0)*(ExactReal(2)*abs(ν1)*norml*r0 + ExactReal(3)*abs(ν2)*norml*r0*(ExactReal(2)*norm(U0,1) + norml*r0))
     𝒞2 = norm_SPL/(μ-δ0)*(ExactReal(2)*abs(ν1)*norml*r0 + ExactReal(3)*abs(ν2)*norml*r0*(ExactReal(2)*norm(U0,1) + norml*r0))
 
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

