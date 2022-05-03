using Flux

using BSON: @load
using MLDatasets
using ImageView
using Revise
using InvertibleNetworks
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

L = 2
K = 6
n_hidden = 64
low = 0.5f0

G = NetworkGlowCond(1, n_hidden, L, K, feature_extractor_model; split_scales=true, p2=0, k2=1, activation=SigmoidLayer(low=low,high=1.0f0))
out, feature_pyramid, cond_network_inputs, logdet = G.forward(X_test[:, :, :, 1:4], incomplete_data[:, :, :, 1:4])
print("nothing")
# isapprox(X_test[:, :, :, 1:4], G.inverse(out, feature_pyramid), atol =.0001)

<<<<<<< HEAD
ΔX, X = G.backward(out, out, incomplete_data[:, :, :, 1:4], feature_pyramid, cond_network_inputs)
=======
ΔX, X = G.backward(out, out, incomplete_data[:, :, :, 1:4], feature_pyramid)

print("done")
>>>>>>> 5c48411cd9841916d67fc10b2afd879cf7ca6128
