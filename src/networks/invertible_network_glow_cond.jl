# Invertible network based on Glow (Kingma and Dhariwal, 2018)
# Includes 1x1 convolution and residual block
# Author: Philipp Witte, pwitte3@gatech.edu
# Date: February 2020
using Flux
export NetworkGlowCond

"""
Everything is the same as NetworkGlow except for the following params:
    Constructor Differences:
    Inputs:
        CP:: List of Layers that produce the feature pyramid. (Conditioning Network)
        condition_extractor_net: ANN model to create features to be fed into the conditioning network

Forward Method Differences
    Inputs:
        C: is the condition that must be fed into the condition extractor net
    Outputs:
        feature_pyramid: This is the output of each conditioning layer. These are needed for the inverse method 
        cond_network_inputs: This is the input to each conditioning layer. These are needed for the backwards method
        
Inverse Method Differences
    Inputs:
        feature_pyramid: We need to use the feature pyramid in the inverse method. (This can be generated from a forward pass)        
Backwards Method Differences
    Inputs:
        C: condition that is fed into the feature extractor model
        feature_pyramid: list of conditions (outputs of conditioning network) 
        cond_network_inputs: list of inputs to the conditioning network layers
    Outputs:
        ΔC: gradient of condition

Additional Comments:
Only set_grad = True is supported. Any logic involving set_grad = False is untested
and probably broken

The first thing that I would do on this project given more time is to reformat the 
glow conditional coupling block so that c_in does not have to be equal to n_in / 2
"""
struct NetworkGlowCond <: InvertibleNetwork
    AN::AbstractArray{ActNorm, 2}
    CL::AbstractArray{CouplingLayerGlowCond, 2}
    CP::AbstractArray{FluxBlock, 2}
    Z_dims::Union{Array{Array, 1}, Nothing}
    L::Int64
    K::Int64
    condition_extractor_net::FluxBlock
    squeezer::Squeezer
    split_scales::Bool
end

@Flux.functor NetworkGlow

# Constructor
function NetworkGlowCond(n_in, n_hidden, L, K, c_in, condition_extractor_net; split_scales=false, k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, ndims=2, squeezer::Squeezer=ShuffleLayer(), activation::ActivationFunction=SigmoidLayer())
    AN = Array{ActNorm}(undef, L, K)    # activation normalization
    CL = Array{CouplingLayerGlowCond}(undef, L, K)  # coupling layers w/ 1x1 convolution and residual block
    CP = Array{FluxBlock}(undef, L, K) # conditioning pyramid
    
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
    end
    return NetworkGlowCond(AN, CL, CP, Z_dims, L, K, condition_extractor_net, squeezer, split_scales)
end

NetworkGlow3D(arΔC; kw...) = NetworkGlow(arΔC...; kw..., ndims=3)

# Forward pass and compute logdet
function forward(X::AbstractArray{T, N}, C::AbstractArray{T, N}, G::NetworkGlowCond) where {T, N}
    L, K = size(G.AN)
    feature_pyramid = Array{Array{Float32}}(undef, L, K)
    cond_network_inputs = Array{Array{Float32}}(undef, L,K) # create a list to hold the innputs to the conditional network

    C = G.condition_extractor_net(C)
    G.split_scales && (Z_save = array_of_array(X, G.L-1))
    logdet = 0
    for i=1:G.L
        if G.split_scales
            X = G.squeezer.forward(X)
            C =  G.squeezer.forward(C)
        end         

        for j=1:G.K  
            X, logdet1 = G.AN[i, j].forward(X)
            cond_network_inputs[i, j] = C
            C = G.CP[i, j].forward(C)  # generate new condition
            feature_pyramid[i,j] = C
            X, logdet2 = G.CL[i, j].forward(X,C)
            logdet += (logdet1 + logdet2)
        end

        if G.split_scales && i < G.L    
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
    # This list holds the contribution gradients of the condition coming from residual blocks
    # It holds incomplete gradients (ΔC)
    gradient_feature_pyramid = Array{Array{Float32}}(undef, L,K) 

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

    """Now we run backward on conditioning layers"""
    # ΔC now stores the gradient of the condition derived from downstream of the feature pyramid
    ΔC = zeros(Float32, size(cond_network_inputs[L, K])) 
    for i=G.L:-1:1
        for j=G.K:-1:1 
            
            # This "if" statement accounts for the CCB limitation
            """Since we discard half of the condition to satisfy the CCB constraint
               that c_in == n_in/2 we have to account for this somehow in the backward pass.
               Essentially ΔC (coming from downstream of the conditioning network)
               is now missing channels that the gradients coming from the 
               CCB has. So we pad the channel dimension of ΔC
            """
            if size(ΔC) != size(gradient_feature_pyramid[i, j])
                ΔC  = G.squeezer.inverse(ΔC)
            end
            ΔC = G.CP[i, j].backward(gradient_feature_pyramid[i, j] .+ ΔC, cond_network_inputs[i, j])  
        end
    end

    # backward pass of feature extractor
    ΔC = G.squeezer.inverse(ΔC) 
    ΔC = G.condition_extractor_net.backward(ΔC, C)
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
    clear_grad!(G.condition_extractor_net)
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
