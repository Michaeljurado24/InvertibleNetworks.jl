using MLDatasets
using InvertibleNetworks
using Flux
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM
using Xsum

# load in training data
train_x, train_y = MNIST.traindata()
# Number of digist from mnist to train on 
num_digit = 1
range_dig = 0:num_digit-1

# Use a subset of total training data
batch_size = 1
num_samples = batch_size*45


# Reshape data to 4D tensor with channel length = 1 in penultimate dimension
X_train = reshape(train_x, size(train_x)[1], size(train_x)[2], 1, size(train_x)[3])
X_train = Float32.(X_train[:,:,:,1:num_samples])
X_train = permutedims(X_train, [2, 1, 3, 4])

# load in test data
test_x, test_y = MNIST.testdata()
inds = findall(x -> x in range_dig, test_y)
test_y = test_y[inds]
test_x = test_x[:,:,inds]

X_test   = Float32.(reshape(test_x, size(test_x)[1], size(test_x)[2], 1, size(test_x)[3]))
X_test = permutedims(X_test, [2, 1, 3, 4])


x_batch = X_test[:, :, :, 1:batch_size]
c = deepcopy(x_batch) # set the condition to x_batch just for testing
x_batch = repeat(x_batch, inner = (1, 1, 2, 1))

k1 = 3; p1 = 1; s1 = 1
k2 = 1; p2 = 0; s2 = 1
# (1, n_hidden, L, K; split_scales=true, p2=0, k2=1, activation=SigmoidLayer(low=low,high=1.0f0)) |> gpu

n_in = 2
c_in = 1
n_hidden = 64
layer = CouplingLayerGlowCond(n_in, n_hidden, c_in, k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true)


# Zx_2, _ = layer2.forward(x_batch)  # apply forward pass
# L_2 = xsum(Zx_2 .* Zx_2)/batch_size # your loss is equal to the sum of the output of the forward pass for testing reasons
# ΔY_2 = 2 * Zx_2/batch_size # gradient of loss
# ΔX_2, X_2 = layer2.backward(ΔY_2, Zx_2)

# for x =1:size(x_batch)[1]
#     for y =1:size(x_batch)[2]
#         for z = 1:size(x_batch)[3]
#             x_copy = deepcopy(x_batch)
#             x_copy[x,y,z,:] += ones(Float32, batch_size) * 1e-4
#             Zx_new_2  = layer2.forward(x_copy)[1]
#             L_new_2 = xsum(Zx_new_2 .* Zx_new_2)/batch_size

#             x_copy = deepcopy(x_batch)
#             x_copy[x,y,z,:] -= ones(Float32, batch_size) * 1e-4
#             Zx_new_3  = layer2.forward(x_copy)[1]
#             L_new_3 = xsum(Zx_new_3 .* Zx_new_3)/batch_size

#             lin_deriv_2 = (L_new_2 - L_new_3) / (2e-4)
#             if abs(L_new_2 - L_2) > 1e-7

#                 ref_value = xsum(ΔX_2[x,y,z,:])
#                 println(lin_deriv_2)
#                 println(ref_value)
#                 println("-----------")                
#                 if !isapprox(lin_deriv_2, ref_value, atol=.5)
#                     throw(error())
#                 end
#             end
#         end
#     end
# end
