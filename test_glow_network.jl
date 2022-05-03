using Flux

using BSON: @load
using MLDatasets
using ImageView
using Revise
using Xsum
using InvertibleNetworks
using LinearAlgebra
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


# Define function to convert floats into ints
function int(x)
    return Int(trunc(x))
end

# draw random rectangles on batches of images
function random_rectangle_draw(x_batch)
    max_width_mask = trunc(nx * .25)
    
    for b=1:size(x_batch)[4]
        x_coor = int(rand() * nx + 1)
        y_coor = int(rand() * ny + 1)
        start_x = int(max(1, x_coor - max_width_mask))
        end_x = int(min(ny, x_coor + max_width_mask))

        start_y = int(max(1, y_coor - max_width_mask))
        end_y = int(min(ny, y_coor + max_width_mask))

        # ImageView.imshow(x_batch[:, :, : , b])
        x_batch[start_x: end_x, start_y: end_y, : , b] -= x_batch[start_x: end_x, start_y:end_y, : , b]
        # ImageView.imshow(x_batch[:, :, : , b])
    end
end

incomplete_data = deepcopy(X_test[:, :, :, 1:4])
random_rectangle_draw(incomplete_data)
out = model(incomplete_data)
# for i=1:4
#     ImageView.imshow(X_test[:, :, :, i])
#     ImageView.imshow(incomplete_data[:, :, :, i])
#     ImageView.imshow(out[:, :, :, i])
# end



feature_extractor_model = model[1:3]
#feature_extractor_model = Chain()

L = 2
K = 6
n_hidden = 64
low = 0.5f0
batch_size = 1


x_batch = X_test[:, :, :, 1:batch_size]
c_batch = incomplete_data[:, :, :, 1:batch_size]
G = NetworkGlowCond(1, n_hidden, L, K, feature_extractor_model; split_scales=true, p2=0, k2=1, activation=SigmoidLayer(low=low,high=1.0f0))
Zx, feature_pyramid, cond_network_inputs, logdet = G.forward(x_batch, c_batch)

X_reverse = G.inverse(Zx, feature_pyramid)
println("Testing to make sure that inverse functions")
println(isapprox(X_reverse, x_batch, atol=1e-3))

loss = xsum(Zx .* Zx)/ batch_size
ΔY = 2* Zx/batch_size
println("Testing that backward function is correct as well")
Δ = 1e-3
ΔX, X, ΔC = G.backward(ΔY, Zx, c_batch, feature_pyramid, cond_network_inputs)
print("done")
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
            print(size(lin_deriv))
            ref_value = xsum(ΔC[x,y,z,:])

            if abs(ref_value) > .1

                ref_value = xsum(ΔC[x,y,z,:])
                println(lin_deriv)
                println(ref_value)
                println("-----------")                
                if !isapprox(lin_deriv, ref_value, atol=1)
                    throw(error())
                end
            end
        end
    end
end