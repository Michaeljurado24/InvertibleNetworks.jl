# Invertible network based on Glow (Kingma and Dhariwal, 2018)
# Includes 1x1 convolution and residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020
using Flux
export NetworkGlowCond
"""
    G = NetworkGlow(n_in, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)

    G = NetworkGlow3D(n_in, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)

 Create an invertible network based on the Glow architecture. Each flow step in the inner loop 
 consists of an activation normalization layer, followed by an invertible coupling layer with
 1x1 convolutions and a residual block. The outer loop performs a squeezing operation prior 
 to the inner loop, and a splitting operation afterwards.

 *Input*: 

 - 'n_in': number of input channels

 - `n_hidden`: number of hidden units in residual blocks

 - `L`: number of scales (outer loop)

 - `K`: number of flow steps per scale (inner loop)

 - `split_scales`: if true, perform squeeze operation which halves spatial dimensions and duplicates channel dimensions
    then split output in half along channel dimension after each scale. Feed one half through the next layers,
    while saving the remaining channels for the output.

 - `k1`, `k2`: kernel size of convolutions in residual block. `k1` is the kernel of the first and third 
 operator, `k2` is the kernel size of the second operator.

 - `p1`, `p2`: padding for the first and third convolution (`p1`) and the second convolution (`p2`)

 - `s1`, `s2`: stride for the first and third convolution (`s1`) and the second convolution (`s2`)

 - `ndims` : number of dimensions

 - `squeeze_type` : squeeze type that happens at each multiscale level

 *Output*:
 
 - `G`: invertible Glow network.

 *Usage:*

 - Forward mode: `Y, logdet = G.forward(X)`

 - Backward mode: `ΔX, X = G.backward(ΔY, Y)`

 *Trainable parameters:*

 - None in `G` itself

 - Trainable parameters in activation normalizations `G.AN[i,j]` and coupling layers `G.C[i,j]`,
   where `i` and `j` range from `1` to `L` and `K` respectively.

 See also: [`ActNorm`](@ref), [`CouplingLayerGlow!`](@ref), [`get_params`](@ref), [`clear_grad!`](@ref)
"""
struct NetworkGlowCond <: InvertibleNetwork
    AN::AbstractArray{ActNorm, 2}
    CL::AbstractArray{CouplingLayerGlowCond, 2}
    CP::AbstractArray{FluxBlock, 2}
    Z_dims::Union{Array{Array, 1}, Nothing}
    L::Int64
    K::Int64
    conditioning_network::FluxBlock
    squeezer::Squeezer
    split_scales::Bool
end

@Flux.functor NetworkGlow

# Constructor
function NetworkGlowCond(n_in, n_hidden, L, K, conditioning_network; split_scales=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=2, squeezer::Squeezer=ShuffleLayer(), activation::ActivationFunction=SigmoidLayer())
    AN = Array{ActNorm}(undef, L, K)    # activation normalization
    CL = Array{CouplingLayerGlowCond}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block
    CP = Array{FluxBlock}(undef, L, K) # conditioning pyramid

    
    c_in = size(conditioning_network.forward(zeros(32, 32, 1, 1)))[3]
    if split_scales
        Z_dims = fill!(Array{Array}(undef, L-1), [1,1]) #fill in with dummy values so that |> gpu accepts it   # save dimensions for inverse/backward pass
        channel_factor = 4
    else
        Z_dims = nothing
        channel_factor = 1
    end

    for i=1:L
        n_in *= channel_factor # squeeze if split_scales is turned on
        c_in *= channel_factor
        for j=1:K
            AN[i, j] = ActNorm(n_in; logdet=true)
            CP[i, j] = FluxBlock(Chain(Flux.Conv((3, 3), c_in => Int64(n_in/2), pad= Flux.SamePad(),relu)))
            c_in = Int64(n_in/2)
            CL[i, j] = CouplingLayerGlowCond(n_in, n_hidden, Int(n_in/2); k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true, activation=activation, ndims=ndims)
        end
        (i < L && split_scales) && (n_in = Int64(n_in/2)) # split
        c_in = Int64(n_in /2)
    end

    return NetworkGlowCond(AN, CL, CP, Z_dims, L, K, conditioning_network, squeezer, split_scales)
end

NetworkGlow3D(arΔC; kw...) = NetworkGlow(arΔC...; kw..., ndims=3)

# Forward pass and compute logdet
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::NetworkGlowCond) where {T, N}
    L, K = size(G.AN)
    feature_pyramid = Array{Array{Float32}}(undef, L, K)
    cond_network_inputs = Array{Array{Float32}}(undef, L,K) # create a list to hold the innputs to the conditional network

    C = G.conditioning_network(C)
    G.split_scales && (Z_save = array_of_array(X, G.L-1))
    logdet = 0
    for i=1:G.L
        if G.split_scales
            X = G.squeezer.forward(X)
            C =  G.squeezer.forward(C)
        end         

        # (G.split_scales) && (X = G.squeezer.forward(X))    
        for j=1:G.K  
            X, logdet1 = G.AN[i, j].forward(X)
            cond_network_inputs[i, j] = C
            C = G.CP[i, j].forward(C)
            feature_pyramid[i,j] = C
            X, logdet2 = G.CL[i, j].forward(X,C)
            logdet += (logdet1 + logdet2)
        end

        if G.split_scales && i < G.L    # don't split after last iteration
            C = C[:, :, 1:Int64(size(C)[3]/2), :]
            X, Z = tensor_split(X)
            Z_save[i] = Z
            G.Z_dims[i] = collect(size(Z))
        end
    end
    G.split_scales && (X = cat_states(Z_save, X))
    return X, feature_pyramid, cond_network_inputs, logdet
end

# Inverse pass 
function inverse(X::AbstractArray{T, N}, feature_pyramid, G::NetworkGlowCond) where {T, N}
    G.split_scales && ((Z_save, X) = split_states(X, G.Z_dims))
    for i=G.L:-1:1
        if G.split_scales && i < G.L
            X = tensor_cat(X, Z_save[i])
        end
        for j=G.K:-1:1

            X = G.CL[i, j].inverse(X, feature_pyramid[i, j])
            X = G.AN[i, j].inverse(X)
        end

        (G.split_scales) && (X = G.squeezer.inverse(X))
    end
    return X
end

# Backward pass and compute gradients
function backward(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, C, feature_pyramid, cond_network_inputs,  G::NetworkGlowCond; set_grad::Bool=true) where {T, N}
    if ~set_grad
        throw(error())
    end

    L, K = size(G.AN)
    gradient_feature_pyramid = Array{Array{Float32}}(undef, L,K) # create a list to hold the gradients of the feature pyramid outputs coming from the residual blocks

    # Split data and gradients
    if G.split_scales
        ΔZ_save, ΔX = split_states(ΔX, G.Z_dims)
        Z_save, X = split_states(X, G.Z_dims)
    end

    blkidx = 10*G.L*G.K
    for i=G.L:-1:1
        if G.split_scales && i < G.L
            X  = tensor_cat(X, Z_save[i])
            ΔX = tensor_cat(ΔX, ΔZ_save[i])
        end
        for j=G.K:-1:1
            if set_grad
                C_ = feature_pyramid[i, j]
                ΔX, X, ΔC = G.CL[i, j].backward(ΔX, X, C_)
                gradient_feature_pyramid[i, j] = ΔC
                ΔX, X = G.AN[i, j].backward(ΔX, X)
            end
            blkidx -= 10
        end

        if G.split_scales 
            X = G.squeezer.inverse(X)
          ΔX = G.squeezer.inverse(ΔX)
        end
    end

    ΔC = zeros(Float32, size(cond_network_inputs[L, K]))
    # Backward pass of conditioning network
    for i=G.L:-1:1
        for j=G.K:-1:1  # wrong order
            if size(ΔC) != size(gradient_feature_pyramid[i, j])
                ΔC_size = size(ΔC)
                result = fill(Float32(0), (ΔC_size[1], ΔC_size[2], ΔC_size[3]*2, ΔC_size[4]))
                result[:, :, 1: ΔC_size[3], :] .= ΔC 
                ΔC  = G.squeezer.inverse(result)
                println(size(ΔC))
            end
            println(size(ΔC))
            ΔC = G.CP[i, j].backward(gradient_feature_pyramid[i, j] .+ ΔC, cond_network_inputs[i, j])  
            println("Does not reach here")
        end
    end
    #backward pass of feature extractor
    ΔC = G.squeezer.inverse(ΔC) 
    ΔC = G.conditioning_network.backward(ΔC, C)
    set_grad ? (return ΔX, X, ΔC) : (return ΔX, Δθ, X, ∇logdet)
end


## Jacobian-related utils

function jacobian(ΔX::AbstractArray{T, N}, Δθ::Array{Parameter, 1}, X, G::NetworkGlow) where {T, N}

    if G.split_scales 
        Z_save = array_of_array(ΔX, G.L-1)
        ΔZ_save = array_of_array(ΔX, G.L-1)
    end
    logdet = 0
    GNΔθ = Array{Parameter, 1}(undef, 10*G.L*G.K)
    blkidx = 0
    for i=1:G.L
        if G.split_scales 
            X = G.squeezer.forward(X) 
            ΔX = G.squeezer.forward(ΔX) 
        end
        
        for j=1:G.K
            Δθ_ij = Δθ[blkidx+1:blkidx+10]
            ΔX, X, logdet1, GNΔθ1 = G.AN[i, j].jacobian(ΔX, Δθ_ij[1:2], X)
            ΔX, X, logdet2, GNΔθ2 = G.CL[i, j].jacobian(ΔX, Δθ_ij[3:end], X)
            logdet += (logdet1 + logdet2)
            GNΔθ[blkidx+1:blkidx+10] = cat(GNΔθ1,GNΔθ2; dims=1)
            blkidx += 10
        end
        if G.split_scales && i < G.L    # don't split after last iteration
            X, Z = tensor_split(X)
            ΔX, ΔZ = tensor_split(ΔX)
            Z_save[i] = Z
            ΔZ_save[i] = ΔZ
            G.Z_dims[i] = collect(size(Z))
        end
    end
    if G.split_scales 
        X = cat_states(Z_save, X)
        ΔX = cat_states(ΔZ_save, ΔX)
    end
    
    return ΔX, X, logdet, GNΔθ
end

adjointJacobian(ΔX::AbstractArray{T, N}, X::AbstractArray{T, N}, G::NetworkGlow) where {T, N} = backward(ΔX, X, G; set_grad=false)


## Other utils

# Clear gradients
function clear_grad!(G::NetworkGlowCond)
    L, K = size(G.AN)
    clear_grad!(G.conditioning_network)
    for i=1:L
        for j=1:K
            clear_grad!(G.AN[i, j])
            clear_grad!(G.CL[i, j])
            clear_grad!(G.CP[i, j])
        end
    end
end

# Get parameters
function get_params(G::NetworkGlowCond)
    L, K = size(G.AN)
    p = Array{Parameter, 1}(undef, 0)
    for i=1:L
        for j=1:K
            p = cat(p, get_params(G.AN[i, j]); dims=1)
            p = cat(p, get_params(G.CL[i, j]); dims=1)
            p = cat(p, get_params(G.CP[i, j]); dims=1)
        end
    end
    return p
end
