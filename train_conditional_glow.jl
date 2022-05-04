using Flux

using BSON: @load
using MLDatasets
using Revise
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM
using Xsum
using InvertibleNetworks
using LinearAlgebra
using HDF5
using Random
using PyPlot
using DrWatson
include("inpainting_helpers.jl")



# Random seed
Random.seed!(20)
pretrained_model = false

if pretrained_model
    @load "best_cnn.bson" model # run python train_cnn_inpainting.jl before
else
    model = create_autoencoder_net()
end

# Plotting dir
plot_dir = "glow_cond_out"
mkdir(plot_dir)

# Training hyperparameters
nepochs    = 300
batch_size = 16
lr        = 5f-4
noiseLev   = 0.01f0 # Additive noise


# Number of digist from mnist to train on 
num_digit = 1
range_dig = 0:num_digit-1

# load in training data
train_x, train_y = MNIST.traindata()

# grab the digist to train on 
inds = findall(x -> x in range_dig, train_y)
train_x = train_x[:,:,inds]

# Use a subset of total training data
num_samples = batch_size*45

# Number of training examples seen during training
total_seen = nepochs*num_samples

# Reshape data to 4D tensor with channel length = 1 in penultimate dimension
X_train = reshape(train_x, size(train_x)[1], size(train_x)[2], 1, size(train_x)[3])
X_train = Float32.(X_train[:,:,:,1:num_samples])
X_train = permutedims(X_train, [2, 1, 3, 4])

nx, ny, nc, n_samples_train = size(X_train);
N = nx*ny 

# Split in training/testing
#use all as training set because there is a separate testing set
train_fraction = 1
ntrain = Int(floor((n_samples_train*train_fraction)))
nbatches = cld(ntrain, batch_size)


# load in test data
test_x, test_y = MNIST.testdata()
inds = findall(x -> x in range_dig, test_y)
test_x = test_x[:,:,inds]

X_test   = Float32.(reshape(test_x, size(test_x)[1], size(test_x)[2], 1, size(test_x)[3]))
X_test = permutedims(X_test, [2, 1, 3, 4])

X_test_latent  = X_test[:,:,:,:];
X_test_latent  .+= noiseLev*randn(Float32, size(X_test_latent));

y_test = deepcopy(X_test)
random_rectangle_draw(X_test) # draw a set number of random rectangles on the testing dataset
hypothesis_to_generate = 5

_, _, _, n_samples_test = size(X_train);

# Split in training/testing
#use all as training set because there is a separate testing set
test_fraction = 1
ntest = Int(floor((n_samples_test*test_fraction)))
nbatches_test = cld(ntest, batch_size)
idx_e_test = reshape(randperm(ntest), batch_size, nbatches_test)

# define architecture with params
L = 2
K = 6
n_hidden = 64
low = 0.5f0
feature_extractor_model = FluxBlock(model[1:3])
G = NetworkGlowCond(1, n_hidden, L, K, feature_extractor_model; split_scales=true, p2=0, k2=1, activation=SigmoidLayer(low=low,high=1.0f0))

# Params copy for Epoch-adaptive regularization weight
θ = get_params(G);

opt = ADAMW(lr)

train_loss_list = Array{Float64}(undef, nepochs)  # contains training loss
test_loss_list =  Array{Float64}(undef, nepochs)  # contains mse
best_test_loss = typemax(Int)

save_weights_every = 5
save_image_number = 10
for e=1:nepochs
    print("epoch: ", e)
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)
    total_train_loss = 0
    for b = 1:nbatches # batch loop
        println("batch: ", b)
        x_batch = X_train[:, :, :, idx_e[:,b]] # obtain training batch 
        y_batch = deepcopy(x_batch)  # Undamaged ground truth
        random_rectangle_draw(x_batch)  # damage training batch

        Zx, feature_pyramid, cond_network_inputs, logdet = G.forward(y_batch, x_batch) # run forward pass
       
        # observe loss
        loss = norm(Zx)^2 / (N*batch_size)
        total_train_loss += loss  
        G.backward((Zx / batch_size), (Zx), x_batch, feature_pyramid, cond_network_inputs) 

        for i =1:length(θ)
            update!(opt, θ[i].data, θ[i].grad)
        end
    end

    train_loss_list[e] = total_train_loss
    total_test_loss = 0
    println("starting testing")
    for b = 1:nbatches_test # batch loop
        x_batch = X_test[:, :, :, idx_e_test[:,b]]  # these are our damaged images
        x_batch_latent = X_test_latent[:, :, :, idx_e_test[:,b]]  # random noise
        y_batch = y_test[:, :, :, idx_e_test[:,b]] #
        _, feature_pyramid, _, _ = G.forward(y_batch, x_batch) # generate the feature pyramid

        y_batch_reverse = G.inverse(x_batch_latent[:], feature_pyramid)  # generate a hypothesis

        # every save_weights_every epochs store some sample imagery and same weights
        if b == 1 && mod(e, save_weights_every) == 0
            save_dir = plot_dir * "/" * string(e)
            mkdir(save_dir)
            for i=1:save_image_number  # plot 10 images
                save_img = save_dir * "/" * string(i) * "_ground_truth.png"
                PyPlot.imsave(save_img, y_batch[:, :, 1, i])
                save_img = save_dir *  "/" * string(i) * "_corrupted.png"
                PyPlot.imsave(save_img, x_batch[:, :, 1, i])
                save_img = save_dir *  "/" * string(i) * "_hypothesis.png"
                PyPlot.imsave(save_img, clamp.(y_batch_reverse[:, :, 1, i], 0, 1))
            end
            Params = get_params(G)
            save_dict = @strdict Params
            @tagsave(
                plot_dir * "/" * string(e) * "/weights.jld2",
                save_dict,
                safe = true
            )

        end
        test_loss = Flux.mse(y_batch_reverse, y_batch)  # observe mse loss 
        total_test_loss += test_loss
    end
    test_loss_list[e] = total_test_loss

    h5open(plot_dir  * "/" * "learning_curves.h5", "w") do file
        g = create_group(file, "curves") # create a group
        g["training"] =   train_loss_list[1:e]
        g["testing"] =   test_loss_list[1:e]
        attributes(g)["Description"] = "Learning Curves" # an attribute
    end

end




