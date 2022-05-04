using Flux

using BSON: @load
using MLDatasets
using ImageView
using Revise
using Xsum
using InvertibleNetworks
using LinearAlgebra

# Random seed
Random.seed!(20)

model = create_autoencoder_net()

# Plotting dir
exp_name = "train-mnist"
save_dict = @strdict exp_name
save_path = plotsdir(savename(save_dict; digits=6))

# Training hyperparameters
nepochs    = 300
batch_size = 16
lr        = 5f-4


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

y_test = deepcopy(X_test)
random_rectangle_draw(X_test) # draw a set number of random rectangles

_, _, _, n_samples_test = size(X_train);

# Split in training/testing
#use all as training set because there is a separate testing set
test_fraction = 1
ntest = Int(floor((n_samples_test*test_fraction)))
nbatches_test = cld(ntest, batch_size)


# define architecture with params
L = 2
K = 6
n_hidden = 64
low = 0.5f0
feature_extractor_model = FluxBlock(model[1:3])
G = NetworkGlowCond(1, n_hidden, L, K, feature_extractor_model; split_scales=true, p2=0, k2=1, activation=SigmoidLayer(low=low,high=1.0f0))



