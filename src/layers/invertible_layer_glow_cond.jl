# Affine coupling layer from Dinh et al. (2017)
# Includes 1x1 convolution from in Putzky and Welling (2019)
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: January 2020

export CouplingLayerGlowCond

using Flux
"""
    CL = CouplingLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false)

or

    CL = CouplingLayerGlow(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, ndims=2) (2D)

    CL = CouplingLayerGlow(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, ndims=3) (3D)
    
    CL = CouplingLayerGlow3D(n_in, n_hidden; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false) (3D)

 Create a Real NVP-style invertible coupling layer based on 1x1 convolutions and a residual block.

 *Input*:

 - `C::Conv1x1`: 1x1 convolution layer

 - `RB::ResidualBlock`: residual block layer consisting of 3 convolutional layers with ReLU activations.

 - `logdet`: bool to indicate whether to compte the logdet of the layer

 or

 - `n_in`, `n_hidden`: number of input and hidden channels

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third
    operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `ndims` : number of dimensions

 *Output*:

 - `CL`: Invertible Real NVP coupling layer.

 *Usage:*

 - Forward mode: `Y, logdet = CL.forward(X)`    (if constructed with `logdet=true`)

 - Inverse mode: `X = CL.inverse(Y)`

 - Backward mode: `ΔX, X = CL.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `CL` itself

 - Trainable parameters in residual block `CL.RB` and 1x1 convolution layer `CL.C`

 See also: [`Conv1x1`](@ref), [`ResidualBlock`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct CouplingLayerGlowCond <: NeuralNetLayer
    C::Conv1x1
    RB::Union{ResidualBlock, FluxBlock}
    logdet::Bool
    activation::ActivationFunction
end

@Flux.functor CouplingLayerGlow

# Constructor from 1x1 convolution and residual block
function CouplingLayerGlowCond(C::Conv1x1, RB::ResidualBlock; logdet=false, activation::ActivationFunction=SigmoidLayer())
    RB.fan == false && throw("Set ResidualBlock.fan == true")
    return CouplingLayerGlowCond(C, RB, logdet, activation)
end

# Constructor from 1x1 convolution and residual Flux block
CouplingLayerGlowCond(C::Conv1x1, RB::FluxBlock; logdet=false, activation::ActivationFunction=SigmoidLayer()) = CouplingLayerGlow(C, RB, logdet, activation)

# Constructor from input dimensions
function CouplingLayerGlowCond(n_in::Int64, n_hidden::Int64, c_in::Int64; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false, activation::ActivationFunction=SigmoidLayer(), ndims=2)

    # 1x1 Convolution and residual block for invertible layer
    if Int(n_in/2) != c_in
        throw(error())  # n_in has to be equal to c_in/2. This is neccessary so that the Residual Block returns n_in outputs when fan = false.
    end

    C = Conv1x1(n_in)
    RB = ResidualBlock(n_in, n_hidden; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, fan=false, ndims=ndims)  # take the full input dimension here
    return CouplingLayerGlowCond(C, RB, logdet, activation)
end

# CouplingLayerGlow3D(args...;kw...) = CouplingLayerGlow(args...; kw..., ndims=3)

# Forward pass: Input X, Output Y
function forward(X::AbstractArray{T, 4}, C::AbstractArray{T, 4}, L::CouplingLayerGlowCond) where T

    # Get dimensions
    k = Int(L.C.k/2)

    X_ = L.C.forward(X)
    X2, X1 = tensor_split(X_)



    Y2 = copy(X2)
    X2_C = cat(X2, C, dims = (3))  # concatenate along channel length
    #println(size(X2_C))
    logS_T = L.RB.forward(X2_C)
    #println(size(logS_T))
    Sm = L.activation.forward(logS_T[:,:,1:k,:])
    Tm = logS_T[:, :, k+1:end, :]
    Y1 = Sm.*X1 + Tm

    Y = tensor_cat(Y2, Y1)

    L.logdet == true ? (return Y, glow_logdet_forward(Sm)) : (return Y)
end

# Inverse pass: Input Y, Output X
function inverse(Y::AbstractArray{T, 4}, C::AbstractArray{T, 4}, L::CouplingLayerGlowCond; save=false) where T

    # Get dimensions
    k = Int(L.C.k/2)
    Y2, Y1 = tensor_split(Y)

    X2 = copy(Y2)
    X2_C = cat(X2, C, dims = (3))  # concatenate along channel length
    logS_T = L.RB.forward(X2_C)
    Sm = L.activation.forward(logS_T[:,:,1:k,:])
    Tm = logS_T[:, :, k+1:end, :]
    X1 = (Y1 - Tm) ./ (Sm .+ eps(T)) # add epsilon to avoid division by 0

    X_ = tensor_cat(X2, X1)
    X = L.C.inverse(X_)

    save == true ? (return X, X1, X2_C, Sm) : (return X)
end

# Backward pass: Input (ΔY, Y, C), Output (ΔX, X, ΔC)
function backward(ΔY::AbstractArray{T, 4}, Y::AbstractArray{T, 4}, C::AbstractArray{T, 4}, L::CouplingLayerGlowCond; set_grad::Bool=true) where T

    # Recompute forward state
    k = Int(L.C.k/2)
    X, X1, X2_C, S = inverse(Y, C, L; save=true)
    X2, C = tensor_split((X2_C))

    # Backpropagate residual
    #ΔY1, ΔY2 = tensor_split(ΔY)
    ΔY2, ΔY1 = tensor_split(ΔY)
    ΔT = copy(ΔY1)
    ΔS = ΔY1 .* X1
    if L.logdet
        set_grad ? (ΔS -= glow_logdet_backward(S)) : (ΔS_ = glow_logdet_backward(S))
    end
    ΔX1 = ΔY1 .* S
    if set_grad
        ΔX2_C = L.RB.backward(cat(L.activation.backward(ΔS, S), ΔT; dims=3), X2_C)
        ΔX2, ΔC  = tensor_split(ΔX2_C)
        ΔX2 += ΔY2
    else
        ΔX2, Δθrb = L.RB.backward(cat(L.activation.backward(ΔS, S), ΔT; dims=3), X2; set_grad=set_grad)
        _, ∇logdet = L.RB.backward(cat(L.activation.backward(ΔS_, S), 0f0.*ΔT; dims=3), X2; set_grad=set_grad)
        ΔX2 += ΔY2
    end
    #print(size(ΔX2))
    #ΔX_ = tensor_cat(ΔX1, ΔX2)
    ΔX_ = tensor_cat(ΔX2, ΔX1)
    if set_grad
        #ΔX = L.C.inverse((ΔX_, tensor_cat(X1, X2)))[1]
        ΔX = L.C.inverse((ΔX_, tensor_cat(X2, X1)))[1]
    else
        #ΔX, Δθc = L.C.inverse((ΔX_, tensor_cat(X1, X2)); set_grad=set_grad)[1:2]
        ΔX, Δθc = L.C.inverse((ΔX_, tensor_cat(X2, X1)); set_grad=set_grad)[1:2]
        Δθ = cat(Δθc, Δθrb; dims=1)
    end

    if set_grad
        return ΔX, X, ΔC
    else
        L.logdet ? (return ΔX, Δθ, X, cat(0*Δθ[1:3], ∇logdet; dims=1)) : (return ΔX, Δθ, X)
    end
end

# function backward(ΔY::AbstractArray{T,4}, Y::AbstractArray{T,4}, CL::CouplingLayerGlow) where T

# #     #ΔY1, ΔY2 = tensor_split(ΔY); 
# #     #Y1, Y2 = tensor_split(Y);
     
#      ΔY2, ΔY1 = tensor_split(ΔY); 
#      Y2, Y1 = tensor_split(Y);

#     ΔX2 = ΔY2; 
#     X2 = Y2
#     t = CL.RB.forward(X2)
#     Δt = ΔY1
    
#     logs, t = tensor_split(t)
#     s = CL.activation.forward(logs)
#     X1 = (Y1-t)./s
#     ΔX1 = ΔY1.*s
#     Δs = X1.*ΔY1
#     #CL.logdet && (Δs .-= dlogdet(CL, s))
#     CL.logdet && (Δs .-= glow_logdet_backward(s))
#     #Δlogs = CL.activation.backward(Δs,  nothing, logs)
#     Δlogs = CL.activation.backward(Δs,  s)
#     ΔX2 .+= CL.RB.backward(tensor_cat(Δlogs, Δt), X2)

# #     #ΔX_ = tensor_cat(ΔX1, ΔX2)
# #     #ΔX = CL.C.backward(ΔX_, tensor_cat(X1, X2))[1]
# #     #X = CL.C.inverse(tensor_cat(X1, X2))
#      ΔX_ = tensor_cat(ΔX2, ΔX1)
#      ΔX = CL.C.inverse((ΔX_, tensor_cat(X2, X1)))[1]
#      X = CL.C.inverse(tensor_cat(X2, X1))

#     return ΔX, X

# end


## Jacobian-related functions

function jacobian(ΔX::AbstractArray{T, 4}, Δθ::Array{Parameter, 1}, X, L::CouplingLayerGlowCond) where T

    # Get dimensions
    k = Int(L.C.k/2)

    ΔX_, X_ = L.C.jacobian(ΔX, Δθ[1:3], X)
    X1, X2 = tensor_split(X_)
    ΔX1, ΔX2 = tensor_split(ΔX_)

    Y2 = copy(X2)
    ΔY2 = copy(ΔX2)
    ΔlogS_T, logS_T = L.RB.jacobian(ΔX2, Δθ[4:end], X2)
    Sm = L.activation.forward(logS_T[:,:,1:k,:])
    ΔS = L.activation.backward(ΔlogS_T[:,:,1:k,:], nothing;x=logS_T[:,:,1:k,:])
    Tm = logS_T[:, :, k+1:end, :]
    ΔT = ΔlogS_T[:, :, k+1:end, :]
    Y1 = Sm.*X1 + Tm
    ΔY1 = ΔS.*X1 + Sm.*ΔX1 + ΔT
    Y = tensor_cat(Y1, Y2)
    ΔY = tensor_cat(ΔY1, ΔY2)

    # Gauss-Newton approximation of logdet terms
    JΔθ = L.RB.jacobian(cuzeros(ΔX2, size(ΔX2)), Δθ[4:end], X2)[1][:, :, 1:k, :]
    GNΔθ = cat(0f0*Δθ[1:3], -L.RB.adjointJacobian(tensor_cat(L.activation.backward(JΔθ, Sm), zeros(Float32, size(Sm))), X2)[2]; dims=1)

    L.logdet ? (return ΔY, Y, glow_logdet_forward(Sm), GNΔθ) : (return ΔY, Y)
end

function adjointJacobian(ΔY::AbstractArray{T, N}, Y::AbstractArray{T, N}, L::CouplingLayerGlowCond) where {T, N}
    return backward(ΔY, Y, L; set_grad=false)
end


## Other utils

# Clear gradients
function clear_grad!(L::CouplingLayerGlowCond)
    clear_grad!(L.C)
    clear_grad!(L.RB)
end

# Get parameters
function get_params(L::CouplingLayerGlowCond)
    p1 = get_params(L.C)
    p2 = get_params(L.RB)
    return cat(p1, p2; dims=1)
end

# Logdet (correct?)
glow_logdet_forward(S) = sum(log.(abs.(S))) / size(S, 4)
glow_logdet_backward(S) = 1f0./ S / size(S, 4)
