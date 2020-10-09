# Author: Gabrio Rizzuti, grizzuti3@gatech.edu
# Date: September 2020
# Copyright: Georgia Institute of Technology, 2020
#
# Jacobian utilities

export Jacobian

function Jacobian(Net::Union{InvertibleNetwork,NeuralNetLayer}, X::Array{Float32,N}; Y::Union{Nothing,Array{Float32,N}}=nothing, opt_output=false) where N

    # Domain and range data types
    DDT = Tuple{Array{Float32,N}, Array{Parameter,1}}
    RDT = Array{Float32,N}

    # Clean output option
    if !opt_output

        # Forward evaluation
        fop1(input::Tuple{Array{Float32,N}, Array{Parameter,1}}) = Net.jacobian(input[1], input[2], X)[1]
        fop1(Δθ::Array{Parameter,1}) = Net.jacobian(zeros(Float32, size(X)), Δθ, X)[1]

        # Adjoint evaluation
        if isnothing(Y) # recomputing Y=f(X) if not provided
            Y = Net.forward(X)
            isa(Y, Tuple) && (Y = Y[1])
        end
        fop1_adj(ΔY::Array{Float32,N}) = Net.adjointJacobian(ΔY, Y)[1:2]

        return InvertibleNetworkLinearOperator{RDT,DDT}(fop1, fop1_adj)

    else

        # Forward evaluation
        fop(input::Tuple{Array{Float32,N}, Array{Parameter,1}}) = Net.jacobian(input[1], input[2], X)
        fop(Δθ::Array{Parameter,1}) = Net.jacobian(zeros(Float32, size(X)), Δθ, X)
        fop(input::Tuple{Array{Float32,N}, Array{Parameter,1}, Array{Float32,N}, Array{Parameter,1}}) = Net.jacobian(input[1], input[2], X) # when input = full output from adjoint eval (logdet=true)
        fop(input::Tuple{Array{Float32,N}, Array{Parameter,1}, Array{Float32,N}}) = Net.jacobian(input[1], input[2], X) # when input = full output from adjoint eval (logdet=false)

        # Adjoint evaluation
        if isnothing(Y) # recomputing Y=f(X) if not provided
            Y = Net.forward(X)
            isa(Y, Tuple) && (Y = Y[1])
        end
        fop_adj(ΔY::Array{Float32,N}) = Net.adjointJacobian(ΔY, Y)
        fop_adj(input::Tuple{Array{Float32,N}, Array{Float32,N}, Float32, Array{Parameter,1}}) = Net.adjointJacobian(input[1], input[2]) # when input = full output from forward eval (logdet=true)
        fop_adj(input::Tuple{Array{Float32,N}, Array{Float32,N}}) = Net.adjointJacobian(input[1], input[2]) # when input = full output from forward eval (logdet=true)

        return InvertibleNetworkLinearOperator{RDT,DDT}(fop, fop_adj)

    end

end

# Special case for coupling layer basic
function Jacobian(Net::CouplingLayerBasic, X1::Array{Float32,N}, X2::Array{Float32,N}; Y1::Union{Nothing,Array{Float32,N}}=nothing, Y2::Union{Nothing,Array{Float32,N}}=nothing) where N

    # Domain and range data types
    DDT = Tuple{Array{Float32,N}, Array{Parameter,1}}
    RDT = Array{Float32,N}

    # Forward evaluation
    fop(ΔX1::Array{Float32,N}, ΔX2::Array{Float32,N}, Δθ::Array{Parameter,1}) = Net.jacobian(ΔX1, ΔX2, Δθ, X1, X2)
    fop(Δθ::Array{Parameter,1}) = Net.jacobian(zeros(Float32, size(X)), zeros(Float32, size(X)), Δθ, X1, X2)

    # Adjoint evaluation
    if isnothing(Y) # recomputing Y=f(X) if not provided
        Y1, Y2 = Net.forward(X1, X2)
        isa(Y2, Tuple) && (Y2 = Y2[1])
    end
    fop_adj(ΔY1::Array{Float32,N}, ΔY2::Array{Float32,N}, varargs...) = Net.adjointJacobian(ΔY1, ΔY2, Y1, Y2)

    return InvertibleNetworkLinearOperator{RDT,DDT}(fop, fop_adj)

end

# Special input case
function *(J::InvertibleNetworkLinearOperator{Array{Float32,N},Tuple{Array{Float32,N}, Array{Parameter,1}}}, Δθ::Array{Parameter,1}) where N
    return J.fop(Δθ)
end
