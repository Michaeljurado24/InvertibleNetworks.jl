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
using MLDatasets

# Random seed
Random.seed!(20)

# Plotting dir
exp_name = "train-mnist"
save_dict = @strdict exp_name
save_path = plotsdir(savename(save_dict; digits=6))

# Training hyperparameters
nepochs    = 300
batch_size = 16
lr        = 5f-4
lr_step   = 10
λ = 1f-1
noiseLev   = 0.01f0 # Additive noise

# Architecture parametrs
L = 2
K = 6
n_hidden = 64
low = 0.5f0

# Number of digist from mnist to train on 
num_digit = 1
range_dig = 0:num_digit-1

# load in training data
train_x, train_y = MNIST.traindata()

# grab the digist to train on 
inds = findall(x -> x in range_dig, train_y)
train_y = train_y[inds]
train_x = train_x[:,:,inds]

# Use a subset of total training data
num_samples = batch_size*45

# Number of training examples seen during training
total_seen = nepochs*num_samples

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
# X_test = permutedims(X_test, [2, 1, 3, 4])



nx, ny, nc, n_samples = size(X_train);

# Split in training/testing
#use all as training set because there is a separate testing set
train_fraction = 1
ntrain = Int(floor((n_samples*train_fraction)))
nbatches = cld(ntrain, batch_size)

function int(x)
    return Int(trunc(x))
end

function randomly_inpaint_batch(x_batch)
    max_width_mask = trunc(nx * .25)
    
    for b=1:size(x_batch)[4]
        x_coor = int(rand() * nx + 1)
        y_coor = int(rand() * ny + 1)
        start_x = int(max(1, x_coor - max_width_mask))
        end_x = int(min(ny, x_coor + max_width_mask))

        start_y = int(max(1, y_coor - max_width_mask))
        end_y = int(min(ny, y_coor + max_width_mask))

        ImageView.imshow(x_batch[:, :, : , b])
        x_batch[start_x: end_x, start_y: end_y, : , b] -= x_batch[start_x: end_x, start_y:end_y, : , b]
        ImageView.imshow(x_batch[:, :, : , b])
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
parameters = Flux.params(model)
opt = ADAMW(1e-3)
loss(x, y) = Flux.Losses.mse(model(x), y)

idx = 1

for e=1:nepochs
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)
    for b = 1:nbatches # batch loop
        x_batch = X_train[:, :, :, idx_e[:,b]]
        y_batch = deepcopy(x_batch)
        randomly_inpaint_batch(x_batch)
        train_loss, back = Zygote.pullback(() -> Flux.Losses.mse(model(x_batch), y_batch), parameters)
        gs = back(one(train_loss))
        update!(opt, parameters, gs)
        print("did update", train_loss)
    end
    break

end
