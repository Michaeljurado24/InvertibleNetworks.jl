using MLDatasets
using Flux

using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM
using Xsum
using InvertibleNetworks
include("inpainting_helpers.jl")

# load in training data
train_x, train_y = MNIST.traindata()
# Number of digist from mnist to train on 
num_digit = 1
range_dig = 0:num_digit-1

# Use a subset of total training data
batch_size = 10
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


k1 = 3; p1 = 1; s1 = 1
k2 = 1; p2 = 0; s2 = 1
# (1, n_hidden, L, K; split_scales=true, p2=0, k2=1, activation=SigmoidLayer(low=low,high=1.0f0)) |> gpu

n_in = 2
c_in = 1
n_hidden = 64
Δ = 5e-4

x_batch = X_test[:, :, :, 1:batch_size]
c_batch =  repeat(deepcopy(x_batch), inner = (1, 1, c_in, 1)) * 20    # make the condition just be a copy of the data
x_batch = repeat(deepcopy(x_batch), inner = (1, 1, n_in, 1))  # duplicate x_data along channel axis so that we have an even number of channels

layer2 = CouplingLayerGlowCond(n_in, n_hidden, c_in; k1=k1, k2=k2, p1=p1, p2=p2, s1=s1, s2=s2, logdet=true)

println("Testing reverse function")
Zx, _ = layer2.forward(x_batch, c_batch)  # apply forward pass
x_batch_reverse = layer2.inverse(Zx, c_batch)
println(isapprox(x_batch_reverse[:],x_batch[:], atol=1e-3))  # use large atol due to scaling input size by 20

print("Testing that the linear gradient approximations are close to the calculated gradient for the conditon")
Zx, _ = layer2.forward(x_batch, c_batch)  # apply forward pass
x_batch_reverse = layer2.inverse(Zx, c_batch)
L = xsum(Zx .* Zx)/batch_size # your loss is equal to the sum of squared of the outputs divided by batch size
ΔY = 2 * Zx/batch_size # gradient of loss
ΔX, X, ΔC  = layer2.backward(ΔY, Zx, c_batch)


print("Testing that the linear gradient approximations are close to the calculated gradient for the conditon")
for x =1:size(c_batch)[1]
    for y =1:size(c_batch)[2]
        for z = 1:size(c_batch)[3]
            c_batch_copy = deepcopy(c_batch)
            c_batch_copy[x,y,z,:] += ones(Float32, batch_size) .*Δ 
            Zx_new  = layer2.forward(x_batch, c_batch_copy)[1]
            L_2 = xsum(Zx_new .* Zx_new)/batch_size

            c_batch_copy = deepcopy(c_batch)
            c_batch_copy[x,y,z,:] -= ones(Float32, batch_size) .*Δ 
            Zx_new  = layer2.forward(x_batch, c_batch_copy)[1]
            L_3 = xsum(Zx_new .* Zx_new)/batch_size


            lin_deriv = (L_2 - L_3) / (2 * Δ)
            ref_value = xsum(ΔC[x,y,z,:])

            if abs(ref_value) > .2

                ref_value = xsum(ΔC[x,y,z,:])
                println("Lin Approximation for ΔC = ", lin_deriv)
                println("Backwards Calculated ΔC = ", ref_value)
                println("-----------")                
                if !isapprox(lin_deriv, ref_value, atol=.5)
                    throw(error())
                end
            end
        end
    end
end

println("L should decrease consistently")
for i=1:100
    global c_batch
    global x_batch
    Zx, _ = layer2.forward(x_batch, c_batch)  # apply forward pass
    L = xsum(Zx .* Zx)/batch_size # your loss is equal to the sum of the output of the forward pass for testing reasons
    println(L)
    ΔY = 2 * Zx/batch_size # gradient of loss
    ΔX, X, ΔC  = layer2.backward(ΔY, Zx, c_batch)
    c_batch = c_batch -  ΔC * Float32(.1)

end