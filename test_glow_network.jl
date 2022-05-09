using Flux

using BSON: @load
using MLDatasets
using ImageView
using Revise
using Xsum
using InvertibleNetworks
using LinearAlgebra
include("inpainting_helpers.jl")

# load in test data
test_x, test_y = MNIST.testdata()
num_digit = 1
range_dig = 0:num_digit-1

inds = findall(x -> x in range_dig, test_y)
test_x = test_x[:,:,inds]

X_test   = Float32.(reshape(test_x, size(test_x)[1], size(test_x)[2], 1, size(test_x)[3]))
X_test = permutedims(X_test, [2, 1, 3, 4])
nx = 28
ny = 28

@load "best_cnn.bson" model

incomplete_data = deepcopy(X_test[:, :, :, 1:4])
random_rectangle_draw(incomplete_data)
out = model(incomplete_data)


feature_extractor_model = FluxBlock(model[1:3])
c_in = 32
L = 2
K = 6
n_hidden = 64
low = 0.5f0
batch_size = 1


x_batch = X_test[:, :, :, 1:batch_size]
c_batch = incomplete_data[:, :, :, 1:batch_size]
G = NetworkGlowCond(1, n_hidden, L, K, c_in, feature_extractor_model; split_scales=true, p2=0, k2=1, activation=SigmoidLayer(low=low,high=1.0f0))
Zx, feature_pyramid, cond_network_inputs, logdet = G.forward(x_batch, c_batch)

X_reverse = G.inverse(Zx, feature_pyramid)
println("Testing to make sure that inverse functions")
println(isapprox(X_reverse, x_batch, atol=1e-3))

loss = xsum(Zx .* Zx)/ batch_size
ΔY = 2* Zx/batch_size
println("Testing that backward function is correct as well")
Δ = 5e-4
ΔX, X, ΔC = G.backward(ΔY, Zx, c_batch, feature_pyramid, cond_network_inputs)
clear_grad!(G)
params = get_params(G)
for x =1:size(c_batch)[1]
    for y =1:size(c_batch)[2]
        for z = 1:size(c_batch)[3]
            c_batch_copy = deepcopy(c_batch)
            c_batch_copy[x,y,z,:] += ones(Float32, batch_size) .*Δ 
            Zx_new  = G.forward(x_batch, c_batch_copy)[1]
            L_2 = xsum(Zx_new .* Zx_new) / batch_size

            c_batch_copy = deepcopy(c_batch)
            c_batch_copy[x,y,z,:] -= ones(Float32, batch_size) .*Δ 
            Zx_new  = G.forward(x_batch, c_batch_copy)[1]
            L_3 = xsum(Zx_new .* Zx_new) / batch_size

            lin_deriv = (L_2 - L_3) / (2 * Δ)
            ref_value = xsum(ΔC[x,y,z,:])

            if abs(ref_value) > .3

                ref_value = xsum(ΔC[x,y,z,:])
                println("Lin Approximation for ΔC = ", lin_deriv)
                println("Backwards Calculated ΔC = ", ref_value)
                println("-----------")                
                if !isapprox(lin_deriv, ref_value, atol=1)
                    throw(error())
                end
            end
        end
    end
end