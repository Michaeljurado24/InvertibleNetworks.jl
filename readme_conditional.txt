Test Code Scripts
Info:
    these test scripts just use a linear approximation of the gradient to test that the backwards function is calculating the gradient of the condtion correctly.

1) julia test_glow_layer.jl  # tests the conditional glow layer
2) julia test_glow_network.jl  # tests the conditional glow network


Training Conditional Network Pipeline

1) julia train_cnn_inpainting.jl  # trains feature extractor model on the inpaiting problem. (Can generate conditions Y)
2) julia train_conditional_glow.jl  # trains a cINN using the above model as a condition generator. (This is a flag you can turn on or off!)
