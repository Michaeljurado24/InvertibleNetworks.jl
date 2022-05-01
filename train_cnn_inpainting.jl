using DrWatson
using InvertibleNetworks
using Random
using Flux
using Flux.Optimise: Optimiser, ExpDecay, update!, ADAM
using LinearAlgebra
using PyPlot
using Distributions
using ImageView, Images
using Zygote
import Random
using BSON: @save
using HDF5
using MLDatasets

# Random seed
Random.seed!(20)

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

model = Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), 1=>16, pad= Flux.SamePad(), relu),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad= Flux.SamePad(), relu),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad= Flux.SamePad(), relu),

    Conv((3, 3), 32=>16, pad= Flux.SamePad(), relu),
    
    Conv((3, 3), 16=>1, pad= Flux.SamePad(), sigmoid)
)


# Plotting dir
exp_name = "train-mnist"
save_dict = @strdict exp_name
save_path = plotsdir(savename(save_dict; digits=6))

# Training hyperparameters
nepochs    = 300
batch_size = 16
lr        = 5f-4

# Architecture parametrs
n_hidden = 64
low = 0.5f0

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


# print(size(X_test))

y_test = deepcopy(X_test)
random_rectangle_draw(X_test) # draw a set number of random rectangles

_, _, _, n_samples_test = size(X_train);

# Split in training/testing
#use all as training set because there is a separate testing set
test_fraction = 1
ntest = Int(floor((n_samples_test*test_fraction)))
nbatches_test = cld(ntest, batch_size)

parameters = Flux.params(model)
opt = ADAMW(lr)

idx = 1

train_loss_list = Array{Float64}(undef, nepochs)
test_loss_list =  Array{Float64}(undef, nepochs)
best_test_loss = typemax(Int)

for e=1:nepochs
    global best_test_loss
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)
    total_train_loss = 0
    for b = 1:nbatches # batch loop
        x_batch = X_train[:, :, :, idx_e[:,b]]
        y_batch = deepcopy(x_batch)
        random_rectangle_draw(x_batch)
        train_loss, back = Zygote.pullback(() -> Flux.Losses.mae(model(x_batch), y_batch), parameters)
        gs = back(one(train_loss))
        total_train_loss += train_loss
        update!(opt, parameters, gs)
        
    end
    train_loss_list[e] = total_train_loss

    # run model on testing data
    total_test_loss = 0
    idx_e = reshape(randperm(ntest), batch_size, nbatches_test)
    for b = 1:nbatches_test # batch loop
        x_batch = X_train[:, :, :, idx_e[:,b]]
        y_batch = deepcopy(x_batch)
        random_rectangle_draw(x_batch)
        total_test_loss += Flux.Losses.mse(model(x_batch), y_batch)
    end
    test_loss_list[e] = total_test_loss
    println("Total test loss", total_test_loss)
    println("Total train loss", total_train_loss)

    if total_test_loss < best_test_loss
        println("Saving best model")
        best_test_loss = total_test_loss
        @save "best_cnn.bson" model

        h5open("learning_curves.h5", "w") do file
            g = create_group(file, "curves") # create a group
            g["training"] =   train_loss_list[1:e]
            g["testing"] =   test_loss_list[1:e]
            attributes(g)["Description"] = "Learning Curves" # an attribute
        end
    else
        print("Test loss stopped improving")
        break

    end
end