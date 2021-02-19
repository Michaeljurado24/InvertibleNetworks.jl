var documenterSearchIndex = {"docs":
[{"location":"api/#Invertible-Layers","page":"API Reference","title":"Invertible Layers","text":"","category":"section"},{"location":"api/#Types","page":"API Reference","title":"Types","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Modules = [InvertibleNetworks]\nOrder  = [:type]\nFilter = t -> t<:NeuralNetLayer","category":"page"},{"location":"api/#InvertibleNetworks.ActNorm","page":"API Reference","title":"InvertibleNetworks.ActNorm","text":"AN = ActNorm(k; logdet=false)\n\nCreate activation normalization layer. The parameters are initialized during  the first use, such that the output has zero mean and unit variance along  channels for the current mini-batch size.\n\nInput:\n\nk: number of channels\nlogdet: bool to indicate whether to compute the logdet\n\nOutput:\n\nAN: Network layer for activation normalization.\n\nUsage:\n\nForward mode: Y, logdet = AN.forward(X)\nInverse mode: X = AN.inverse(Y)\nBackward mode: ΔX, X = AN.backward(ΔY, Y)\n\nTrainable parameters:\n\nScaling factor AN.s\nBias AN.b\n\nSee also: get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.AdditiveCouplingLayerSLIM","page":"API Reference","title":"InvertibleNetworks.AdditiveCouplingLayerSLIM","text":"CS = AdditiveCouplingLayerSLIM(nx, ny, n_in, n_hidden, batchsize, Ψ; logdet=false, permute=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)\n\nCreate an invertible additive SLIM coupling layer.\n\nInput: \n\nnx, ny: spatial dimensions of input\n\nn_in, n_hidden: number of input and hidden channels\nΨ: link function\nloget: bool to indicate whether to return the logdet (default is false)\npermute: bool to indicate whether to apply a channel permutation (default is false)\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third   operator, k2 is the kernel size of the second operator\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2)\n\nOutput:\n\nCS: Invertible SLIM coupling layer\n\nUsage:\n\nForward mode: Y, logdet = CS.forward(X, D, A)    (if constructed with logdet=true)\nInverse mode: X = CS.inverse(Y, D, A)\nBackward mode: ΔX, X = CS.backward(ΔY, Y, D, A)\nwhere A is a linear forward modeling operator and D is the observed data.\n\nTrainable parameters:\n\nNone in CL itself\nTrainable parameters in residual block CL.RB and 1x1 convolution layer CL.C\n\nSee also: Conv1x1, ResidualBlock, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.AffineCouplingLayerSLIM","page":"API Reference","title":"InvertibleNetworks.AffineCouplingLayerSLIM","text":"CS = AffineCouplingLayerSLIM(nx, ny, n_in, n_hidden, batchsize, Ψ; logdet=false, permute=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)\n\nCreate an invertible affine SLIM coupling layer.\n\nInput:\n\nnx, ny: spatial dimensions of input\nn_in, n_hidden: number of input and hidden channels\nΨ: link function\nloget: bool to indicate whether to return the logdet (default is false)\npermute: bool to indicate whether to apply a channel permutation (default is false)\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third  operator, k2 is the kernel size of the second operator\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2)\n\nOutput:\n\nCS: Invertible SLIM coupling layer\n\nUsage:\n\nForward mode: Y, logdet = CS.forward(X, D, A)    (if constructed with logdet=true)\nInverse mode: X = CS.inverse(Y, D, A)\nBackward mode: ΔX, X = CS.backward(ΔY, Y, D, A)\nwhere A is a linear forward modeling operator and D is the observed data.\n\nTrainable parameters:\n\nNone in CL itself\nTrainable parameters in residual block CL.RB and 1x1 convolution layer CL.C\n\nSee also: Conv1x1, ResidualBlock, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.AffineLayer","page":"API Reference","title":"InvertibleNetworks.AffineLayer","text":"AL = AffineLayer(nx, ny, nc; logdet=false)\n\nCreate a layer for an affine transformation.\n\nInput:\n\nnx, ny,nc`: input dimensions and number of channels\nlogdet: bool to indicate whether to compute the logdet\n\nOutput:\n\nAL: Network layer for affine transformation.\n\nUsage:\n\nForward mode: Y, logdet = AL.forward(X)\nInverse mode: X = AL.inverse(Y)\nBackward mode: ΔX, X = AL.backward(ΔY, Y)\n\nTrainable parameters:\n\nScaling factor AL.s\nBias AL.b\n\nSee also: get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.ConditionalLayerHINT","page":"API Reference","title":"InvertibleNetworks.ConditionalLayerHINT","text":"CH = ConditionalLayerHINT(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, permute=true) (2D)\n\nCH = ConditionalLayerHINT(nx, ny, nz, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, permute=true) (3D)\n\nCreate a conditional HINT layer based on coupling blocks and 1 level recursion.\n\nInput:\n\nnx, ny, nz: spatial dimensions of both X and Y.\nn_in, n_hidden: number of input and hidden channels of both X and Y\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third  operator, k2 is the kernel size of the second operator.\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2)\npermute: bool to indicate whether to permute X and Y. Default is true\n\nOutput:\n\nCH: Conditional HINT coupling layer.\n\nUsage:\n\nForward mode: Zx, Zy, logdet = CH.forward_X(X, Y)\nInverse mode: X, Y = CH.inverse(Zx, Zy)\nBackward mode: ΔX, ΔY, X, Y = CH.backward(ΔZx, ΔZy, Zx, Zy)\nForward mode Y: Zy = CH.forward_Y(Y)\nInverse mode Y: Y = CH.inverse(Zy)\n\nTrainable parameters:\n\nNone in CH itself\nTrainable parameters in coupling layers CH.CL_X, CH.CL_Y, CH.CL_YX and in permutation layers CH.C_X and CH.C_Y.\n\nSee also: CouplingLayerBasic, ResidualBlock, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.ConditionalLayerSLIM","page":"API Reference","title":"InvertibleNetworks.ConditionalLayerSLIM","text":"CI = ConditionalLayerSLIM(nx1, nx2, nx_in, nx_hidden, ny1, ny2, ny_in, ny_hidden, batchsize, Op;\n    type=\"affine\", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)\n\nCreate a conditional SLIM layer based on the HINT architecture.\n\nInput: \n\nnx1, nx2: spatial dimensions of X\nnx_in, nx_hidden: number of input and hidden channels of X\nny1, ny2: spatial dimensions of Y\n\nny_in, ny_hidden: number of input and hidden channels of Y\nOp: Linear forward modeling operator\ntype: string to indicate which type of data coupling layer to use (\"additive\", \"affine\", \"learned\")\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third   operator, k2 is the kernel size of the second operator.\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2)\n\nOutput:\n\nCI: Conditional SLIM coupling layer.\n\nUsage:\n\nForward mode: Zx, Zy, logdet = CI.forward_X(X, Y, Op)\nInverse mode: X, Y = CI.inverse(Zx, Zy, Op)\nBackward mode: ΔX, ΔY, X, Y = CI.backward(ΔZx, ΔZy, Zx, Zy, Op)\nForward mode Y: Zy = CI.forward_Y(Y)\nInverse mode Y: Y = CI.inverse(Zy)\n\nTrainable parameters:\n\nNone in CI itself\nTrainable parameters in coupling layers CI.CL_X, CI.CL_Y, CI.CL_XY and in permutation layers CI.C_X and CI.C_Y.\n\nSee also: CouplingLayerHINT, AffineCouplingLayerSLIM, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.ConditionalResidualBlock","page":"API Reference","title":"InvertibleNetworks.ConditionalResidualBlock","text":"RB = ConditionalResidualBlock(nx1, nx2, nx_in, ny1, ny2, ny_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)\n\nCreate a (non-invertible) conditional residual block, consisting of one dense and three convolutional layers  with ReLU activation functions. The dense operator maps the data to the image space and both tensors are  concatenated and fed to the subsequent convolutional layers.\n\nInput:\n\nnx1, nx2, nx_in: spatial dimensions and no. of channels of input image\nny1, ny2, ny_in: spatial dimensions and no. of channels of input data\nn_hidden: number of hidden channels\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third  operator, k2 is the kernel size of the second operator.\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: strides for the first and third convolution (s1) and the second convolution (s2)\n\nor\n\nOutput:\n\nRB: conditional residual block layer\n\nUsage:\n\nForward mode: Zx, Zy = RB.forward(X, Y)\nBackward mode: ΔX, ΔY = RB.backward(ΔZx, ΔZy, X, Y)\n\nTrainable parameters:\n\nConvolutional kernel weights RB.W0, RB.W1, RB.W2 and RB.W3\nBias terms RB.b0, RB.b1 and RB.b2\n\nSee also: get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.Conv1x1","page":"API Reference","title":"InvertibleNetworks.Conv1x1","text":"C = Conv1x1(k; logdet=false)\n\nor\n\nC = Conv1x1(v1, v2, v3; logdet=false)\n\nCreate network layer for 1x1 convolutions using Householder reflections.\n\nInput:\n\nk: number of channels\nv1, v2, v3: Vectors from which to construct matrix.\nlogdet: if true, returns logdet in forward pass (which is always zero)\n\nOutput:\n\nC: Network layer for 1x1 convolutions with Householder reflections.\n\nUsage:\n\nForward mode: Y, logdet = C.forward(X)\nBackward mode: ΔX, X = C.backward((ΔY, Y))\n\nTrainable parameters:\n\nHouseholder vectors C.v1, C.v2, C.v3\n\nSee also: get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.CouplingLayerBasic","page":"API Reference","title":"InvertibleNetworks.CouplingLayerBasic","text":"CL = CouplingLayerBasic(RB::ResidualBlock; logdet=false)\n\nor\n\nCL = CouplingLayerBasic(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=false) (2D)\n\nCL = CouplingLayerBasic(nx, ny, nz, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, logdet=false) (3D)\n\nCreate a Real NVP-style invertible coupling layer with a residual block.\n\nInput:\n\nRB::ResidualBlock: residual block layer consisting of 3 convolutional layers with ReLU activations.\nlogdet: bool to indicate whether to compte the logdet of the layer\n\nor\n\nnx, ny, nz: spatial dimensions of input\nn_in, n_hidden: number of input and hidden channels\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third  operator, k2 is the kernel size of the second operator.\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s1)\n\nOutput:\n\nCL: Invertible Real NVP coupling layer.\n\nUsage:\n\nForward mode: Y1, Y2, logdet = CL.forward(X1, X2)    (if constructed with logdet=true)\nInverse mode: X1, X2 = CL.inverse(Y1, Y2)\nBackward mode: ΔX1, ΔX2, X1, X2 = CL.backward(ΔY1, ΔY2, Y1, Y2)\n\nTrainable parameters:\n\nNone in CL itself\nTrainable parameters in residual block CL.RB\n\nSee also: ResidualBlock, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.CouplingLayerGlow","page":"API Reference","title":"InvertibleNetworks.CouplingLayerGlow","text":"CL = CouplingLayerGlow(C::Conv1x1, RB::ResidualBlock; logdet=false)\n\nor\n\nCL = CouplingLayerGlow(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1, logdet=false)\n\nCreate a Real NVP-style invertible coupling layer based on 1x1 convolutions and a residual block.\n\nInput:\n\nC::Conv1x1: 1x1 convolution layer\nRB::ResidualBlock: residual block layer consisting of 3 convolutional layers with ReLU activations.\nlogdet: bool to indicate whether to compte the logdet of the layer\n\nor\n\nnx, ny: spatial dimensions of input\nn_in, n_hidden: number of input and hidden channels\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third  operator, k2 is the kernel size of the second operator.\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2)\n\nOutput:\n\nCL: Invertible Real NVP coupling layer.\n\nUsage:\n\nForward mode: Y, logdet = CL.forward(X)    (if constructed with logdet=true)\nInverse mode: X = CL.inverse(Y)\nBackward mode: ΔX, X = CL.backward(ΔY, Y)\n\nTrainable parameters:\n\nNone in CL itself\nTrainable parameters in residual block CL.RB and 1x1 convolution layer CL.C\n\nSee also: Conv1x1, ResidualBlock, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.CouplingLayerHINT","page":"API Reference","title":"InvertibleNetworks.CouplingLayerHINT","text":"H = CouplingLayerHINT(nx, ny, n_in, n_hidden, batchsize;\n    logdet=false, permute=\"none\", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1) (2D)\n\nH = CouplingLayerHINT(nx, ny, nz, n_in, n_hidden, batchsize;\n    logdet=false, permute=\"none\", k1=3, k2=3, p1=1, p2=1, s1=1, s2=1) (3D)\n\nCreate a recursive HINT-style invertible layer based on coupling blocks.\n\nInput:\n\nnx, ny, nz: spatial dimensions of input\nn_in, n_hidden: number of input and hidden channels\nlogdet: bool to indicate whether to return the log determinant. Default is false.\npermute: string to specify permutation. Options are \"none\", \"lower\", \"both\" or \"full\".\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third  operator, k2 is the kernel size of the second operator.\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2)\n\nOutput:\n\nH: Recursive invertible HINT coupling layer.\n\nUsage:\n\nForward mode: Y = H.forward(X)\nInverse mode: X = H.inverse(Y)\nBackward mode: ΔX, X = H.backward(ΔY, Y)\n\nTrainable parameters:\n\nNone in H itself\nTrainable parameters in coupling layers H.CL\n\nSee also: CouplingLayerBasic, ResidualBlock, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.CouplingLayerIRIM","page":"API Reference","title":"InvertibleNetworks.CouplingLayerIRIM","text":"IL = CouplingLayerIRIM(C::Conv1x1, RB::ResidualBlock)\n\nor\n\nIL = CouplingLayerIRIM(nx, ny, n_in, n_hidden, batchsize; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, logdet=false) (2D)\n\nIL = CouplingLayerIRIM(nx, ny, nz, n_in, n_hidden, batchsize; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1, logdet=false) (3D)\n\nCreate an i-RIM invertible coupling layer based on 1x1 convolutions and a residual block. \n\nInput: \n\nC::Conv1x1: 1x1 convolution layer\n\nRB::ResidualBlock: residual block layer consisting of 3 convolutional layers with ReLU activations.\n\nor\n\nnx, ny, nz: spatial dimensions of input\n\nn_in, n_hidden: number of input and hidden channels\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third   operator, k2 is the kernel size of the second operator.\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2)\n\nOutput:\n\nIL: Invertible i-RIM coupling layer.\n\nUsage:\n\nForward mode: Y = IL.forward(X)\nInverse mode: X = IL.inverse(Y)\nBackward mode: ΔX, X = IL.backward(ΔY, Y)\n\nTrainable parameters:\n\nNone in IL itself\nTrainable parameters in residual block IL.RB and 1x1 convolution layer IL.C\n\nSee also: Conv1x1, ResidualBlock!, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.FluxBlock","page":"API Reference","title":"InvertibleNetworks.FluxBlock","text":"FB = FluxBlock(model::Chain)\n\nCreate a (non-invertible) neural network block from a Flux network.\n\nInput: \n\nmodel: Flux neural network of type Chain\n\nOutput:\n\nFB: residual block layer\n\nUsage:\n\nForward mode: Y = FB.forward(X)\nBackward mode: ΔX = FB.backward(ΔY, X)\n\nTrainable parameters:\n\nNetwork parameters given by Flux.parameters(model)\n\nSee also:  Chain, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.HyperbolicLayer","page":"API Reference","title":"InvertibleNetworks.HyperbolicLayer","text":"HyperbolicLayer(nx, ny, n_in, batchsize, kernel, stride, pad; action=0, α=1f0, n_hidden=1)\n\nor\n\nHyperbolicLayer(W, b, nx, ny, batchsize, stride, pad; action=0, α=1f0)\n\nCreate an invertible hyperbolic coupling layer.\n\nInput:\n\nnx, ny, n_in, batchsize: Dimensions of input tensor\nkernel, stride, pad: Kernel size, stride and padding of the convolutional operator\naction: String that defines whether layer keeps the number of channels fixed (0),  increases it by a factor of 4 (or 8 in 3D) (1) or decreased it by a factor of 4 (or 8) (-1).\nW, b: Convolutional weight and bias. W has dimensions of (kernel, kernel, n_in, n_in). b has dimensions of n_in.\nα: Step size for second time derivative. Default is 1.\nn_hidden: Increase the no. of channels by n_hidden in the forward convolution.  After applying the transpose convolution, the dimensions are back to the input dimensions.\n\nOutput:\n\nHL: Invertible hyperbolic coupling layer\n\nUsage:\n\nForward mode: X_curr, X_new = HL.forward(X_prev, X_curr)\nInverse mode: X_prev, X_curr = HL.inverse(X_curr, X_new)\nBackward mode: ΔX_prev, ΔX_curr, X_prev, X_curr = HL.backward(ΔX_curr, ΔX_new, X_curr, X_new)\n\nTrainable parameters:\n\nHL.W: Convolutional kernel\nHL.b: Bias\n\nSee also: get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.LearnedCouplingLayerSLIM","page":"API Reference","title":"InvertibleNetworks.LearnedCouplingLayerSLIM","text":"CS = LearnedCouplingLayerSLIM(nx1, nx2, nx_in, ny1, ny2, ny_in, n_hidden, batchsize; \n    logdet::Bool=false, permute::Bool=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)\n\nCreate an invertible SLIM coupling layer with a learned data-to-image-space map.\n\nInput: \n\nnx1, nx2, nx_in: spatial dimensions and no. of channels of input image\n\nny1, ny2, ny_in: spatial dimensions and no. of channels of input data\nn_hidden: number of hidden units in conditional residual block\nloget: bool to indicate whether to return the logdet (default is false)\npermute: bool to indicate whether to apply a channel permutation (default is false)\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third   operator, k2 is the kernel size of the second operator\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2)\n\nOutput:\n\nCS: Invertible SLIM coupling layer with learned data-to-image map\n\nUsage:\n\nForward mode: Y, logdet = CS.forward(X, D, A)    (if constructed with logdet=true)\nInverse mode: X = CS.inverse(Y, D, A)\nBackward mode: ΔX, X = CS.backward(ΔY, Y, D, A)\nwhere A is a linear forward modeling operator and D is the observed data.\n\nTrainable parameters:\n\nNone in CL itself\nTrainable parameters in residual block CL.RB and 1x1 convolution layer CL.C\n\nSee also: Conv1x1, ResidualBlock, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.ResidualBlock","page":"API Reference","title":"InvertibleNetworks.ResidualBlock","text":"RB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false) (2D)\n\nRB = ResidualBlock(nx, ny, nz, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false) (3D)\n\nor\n\nRB = ResidualBlock(nx, ny, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false) (2D)\n\nRB = ResidualBlock(nx, ny, nz, n_in, n_hidden, batchsize; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1, fan=false) (3D)\n\nCreate a (non-invertible) residual block, consisting of three convolutional layers and activation functions.  The first convolution is a downsampling operation with a stride equal to the kernel dimension. The last  convolution is the corresponding transpose operation and upsamples the data to either its original dimensions  or to twice the number of input channels (for fan=true). The first and second layer contain a bias term.\n\nInput:\n\nnx, ny, nz: spatial dimensions of input\nn_in, n_hidden: number of input and hidden channels\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third  operator, k2 is the kernel size of the second operator.\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2)\nfan: bool to indicate whether the ouput has twice the number of input channels. For fan=false, the last  activation function is a gated linear unit (thereby bringing the output back to the original dimensions).  For fan=true, the last activation is a ReLU, in which case the output has twice the number of channels  as the input.\n\nor\n\nW1, W2, W3: 4D tensors of convolutional weights\nb1, b2: bias terms\nnx, ny: spatial dimensions of input image\n\nOutput:\n\nRB: residual block layer\n\nUsage:\n\nForward mode: Y = RB.forward(X)\nBackward mode: ΔX = RB.backward(ΔY, X)\n\nTrainable parameters:\n\nConvolutional kernel weights RB.W1, RB.W2 and RB.W3\nBias terms RB.b1 and RB.b2\n\nSee also: get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#Constructors","page":"API Reference","title":"Constructors","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Modules = [InvertibleNetworks]\nOrder  = [:function]\nFilter = t -> contains(String(Symbol(typeof(t).instance)), \"Layer\")","category":"page"},{"location":"api/#Invertible-Networks","page":"API Reference","title":"Invertible Networks","text":"","category":"section"},{"location":"api/#Types-2","page":"API Reference","title":"Types","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Modules = [InvertibleNetworks]\nOrder   = [:type]\nFilter = t -> t<:InvertibleNetwork","category":"page"},{"location":"api/#InvertibleNetworks.NetworkConditionalHINT","page":"API Reference","title":"InvertibleNetworks.NetworkConditionalHINT","text":"CH = NetworkConditionalHINT(nx, ny, n_in, batchsize, n_hidden, depth; k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)\n\nCreate a conditional HINT network for data-driven generative modeling based  on the change of variables formula.\n\nInput:\n\nnx, ny, n_in, batchsize: spatial dimensions, number of channels and batchsize of input tensors X and Y\nn_hidden: number of hidden units in residual blocks\ndepth: number network layers\nk1, k2: kernel size for first and third residual layer (k1) and second layer (k2)\np1, p2: respective padding sizes for residual block layers\ns1, s2: respective strides for residual block layers\n\nOutput:\n\nCH: conditioinal HINT network\n\nUsage:\n\nForward mode: Zx, Zy, logdet = CH.forward(X, Y)\nInverse mode: X, Y = CH.inverse(Zx, Zy)\nBackward mode: ΔX, X = CH.backward(ΔZx, ΔZy, Zx, Zy)\n\nTrainable parameters:\n\nNone in CH itself\nTrainable parameters in activation normalizations CH.AN_X[i] and CH.AN_Y[i],\n\nand in coupling layers CH.CL[i], where i ranges from 1 to depth.\n\nSee also: ActNorm, ConditionalLayerHINT!, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.NetworkGlow","page":"API Reference","title":"InvertibleNetworks.NetworkGlow","text":"G = NetworkGlow(nx, ny, n_in, batchsize, n_hidden, L, K; k1=3, k2=1, p1=1, p2=0, s1=1, s2=1)\n\nCreate an invertible network based on the Glow architecture. Each flow step in the inner loop   consists of an activation normalization layer, followed by an invertible coupling layer with  1x1 convolutions and a residual block. The outer loop performs a squeezing operation prior   to the inner loop, and a splitting operation afterwards.\n\nInput: \n\nnx, ny, n_in, batchsize: spatial dimensions, number of channels and batchsize of input tensor\n\nn_hidden: number of hidden units in residual blocks\nL: number of scales (outer loop)\nK: number of flow steps per scale (inner loop)\nk1, k2: kernel size of convolutions in residual block. k1 is the kernel of the first and third \n\noperator, k2 is the kernel size of the second operator.\n\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2)\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2)\n\nOutput:\n\nG: invertible Glow network.\n\nUsage:\n\nForward mode: Y, logdet = G.forward(X)\nBackward mode: ΔX, X = G.backward(ΔY, Y)\n\nTrainable parameters:\n\nNone in G itself\nTrainable parameters in activation normalizations G.AN[i,j] and coupling layers G.C[i,j], where i and j range from 1 to L and K respectively.\n\nSee also: ActNorm, CouplingLayerGlow!, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.NetworkHyperbolic","page":"API Reference","title":"InvertibleNetworks.NetworkHyperbolic","text":"H = NetworkHyperbolic(nx, ny, n_in, batchsize, architecture; k=3, s=1, p=1, logdet=true, α=1f0)\n\nH = NetworkHyperbolic(nx, ny, nz, n_in, batchsize, architecture; k=3, s=1, p=1, logdet=true, α=1f0)\n\nCreate an invertible network based on hyperbolic layers. The network architecture is specified by a tuple  of the form ((action1, nhidden1), (action2, nhidden2), ... ). Each inner tuple corresonds to an additional layer.   The first inner tuple argument specifies whether the respective layer increases the number of channels (set to 1),   decreases it (set to -1) or leaves it constant (set to 0).  The second argument specifies the number of hidden   units for that layer.\n\nInput: \n\nnx, ny, nz, n_in, batchsize: spatial dimensions, number of channels and batchsize of input tensor. nz is optional.\n\nn_hidden: number of hidden units in residual blocks\narchitecture: Tuple of tuples specifying the network architecture; ((action1, nhidden1), (action2, nhidden2))\nk, s, p: Kernel size, stride and padding of convolutional kernels\n\nlogdet: Bool to indicate whether to return the logdet\nα: Step size in hyperbolic network. Defaults to 1\n\nOutput:\n\nH: invertible hyperbolic network.\n\nUsage:\n\nForward mode: Y_prev, Y_curr, logdet = H.forward(X_prev, X_curr)\nInverse mode: X_curr, X_new = H.inverse(Y_curr, Y_new)\nBackward mode: ΔX_curr, ΔX_new, X_curr, X_new = H.backward(ΔY_curr, ΔY_new, Y_curr, Y_new)\n\nTrainable parameters:\n\nNone in H itself\nTrainable parameters in the hyperbolic layers H.HL[j].\n\nSee also: CouplingLayer!, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.NetworkLoop","page":"API Reference","title":"InvertibleNetworks.NetworkLoop","text":"L = NetworkLoop(nx, ny, n_in, n_hidden, batchsize, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1) (2D)\n\nL = NetworkLoop(nx, ny, nz, n_in, n_hidden, batchsize, maxiter, Ψ; k1=4, k2=3, p1=0, p2=1, s1=4, s2=1) (3D)\n\nCreate an invertibel recurrent inference machine (i-RIM) consisting of an unrooled loop  for a given number of iterations.\n\nInput: \n\nnx, ny, nz, n_in, batchsize: spatial dimensions, number of channels and batchsize of input tensor\n\nn_hidden: number of hidden units in residual blocks\nmaxiter: number unrolled loop iterations\nΨ: link function\nk1, k2: stencil sizes for convolutions in the residual blocks. The first convolution  uses a stencil of size and stride k1, thereby downsampling the input. The second  convolutions uses a stencil of size k2. The last layer uses a stencil of size and stride k1, but performs the transpose operation of the first convolution, thus upsampling the output to  the original input size.\np1, p2: padding for the first and third convolution (p1) and the second convolution (p2) in residual block\ns1, s2: stride for the first and third convolution (s1) and the second convolution (s2) in residual block\n\nOutput:\n\nL: invertible i-RIM network.\n\nUsage:\n\nForward mode: η_out, s_out = L.forward(η_in, s_in, d, A)\nInverse mode: η_in, s_in = L.inverse(η_out, s_out, d, A)\nBackward mode: Δη_in, Δs_in, η_in, s_in = L.backward(Δη_out, Δs_out, η_out, s_out, d, A)\n\nTrainable parameters:\n\nNone in L itself\nTrainable parameters in the invertible coupling layers L.L[i], and actnorm layers L.AN[i], where i ranges from 1 to the number of loop iterations.\n\nSee also: CouplingLayerIRIM, ResidualBlock, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.NetworkMultiScaleConditionalHINT","page":"API Reference","title":"InvertibleNetworks.NetworkMultiScaleConditionalHINT","text":"CH = NetworkMultiScaleConditionalHINT(nx, ny, n_in, batchsize, n_hidden,  L, K; split_scales=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)\n\nCreate a conditional HINT network for data-driven generative modeling based  on the change of variables formula.\n\nInput: \n\nnx, ny, n_in, batchsize: spatial dimensions, number of channels and batchsize of input tensors X and Y\n\nn_hidden: number of hidden units in residual blocks\nL: number of scales (outer loop)\nK: number of flow steps per scale (inner loop)\nsplit_scales: if true, split output in half along channel dimension after each scale. Feed one half through the next layers,  while saving the remaining channels for the output.\nk1, k2: kernel size for first and third residual layer (k1) and second layer (k2)\np1, p2: respective padding sizes for residual block layers\n\ns1, s2: respective strides for residual block layers\n\nOutput:\n\nCH: conditional HINT network\n\nUsage:\n\nForward mode: Zx, Zy, logdet = CH.forward(X, Y)\nInverse mode: X, Y = CH.inverse(Zx, Zy)\nBackward mode: ΔX, X = CH.backward(ΔZx, ΔZy, Zx, Zy)\n\nTrainable parameters:\n\nNone in CH itself\nTrainable parameters in activation normalizations CH.AN_X[i] and CH.AN_Y[i], \n\nand in coupling layers CH.CL[i], where i ranges from 1 to depth.\n\nSee also: ActNorm, ConditionalLayerHINT!, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#InvertibleNetworks.NetworkMultiScaleHINT","page":"API Reference","title":"InvertibleNetworks.NetworkMultiScaleHINT","text":"H = NetworkMultiScaleHINT(nx, ny, n_in, batchsize, n_hidden, L, K; split_scales=false, k1=3, k2=3, p1=1, p2=1, s1=1, s2=1)\n\nCreate a multiscale HINT network for data-driven generative modeling based  on the change of variables formula.\n\nInput: \n\nnx, ny, n_in, batchsize: spatial dimensions, number of channels and batchsize of input tensor X\n\nn_hidden: number of hidden units in residual blocks\nL: number of scales (outer loop)\nK: number of flow steps per scale (inner loop)\nsplit_scales: if true, split output in half along channel dimension after each scale. Feed one half through the next layers,  while saving the remaining channels for the output.\nk1, k2: kernel size for first and third residual layer (k1) and second layer (k2)\np1, p2: respective padding sizes for residual block layers\n\ns1, s2: respective strides for residual block layers\n\nOutput:\n\nH: multiscale HINT network\n\nUsage:\n\nForward mode: Z, logdet = H.forward(X)\nInverse mode: X = H.inverse(Z)\nBackward mode: ΔX, X = H.backward(ΔZ, Z)\n\nTrainable parameters:\n\nNone in H itself\nTrainable parameters in activation normalizations H.AN[i], \n\nand in coupling layers H.CL[i], where i ranges from 1 to depth.\n\nSee also: ActNorm, CouplingLayerHINT!, get_params, clear_grad!\n\n\n\n\n\n","category":"type"},{"location":"api/#Constructors-2","page":"API Reference","title":"Constructors","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"Modules = [InvertibleNetworks]\nOrder  = [:function]\nFilter = t -> contains(String(Symbol(typeof(t).instance)), \"Network\")","category":"page"},{"location":"examples/#Simple-examples","page":"Examples","title":"Simple examples","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"We provide usage examples for all the layers and network in our examples subfolder. Each of the example show how to setup and use the building block for simple random variables.","category":"page"},{"location":"examples/#Litterature-applications","page":"Examples","title":"Litterature applications","text":"","category":"section"},{"location":"examples/","page":"Examples","title":"Examples","text":"The following examples show the implementaton of applications from the linked papers with [InvertibleNetworks.jl]:","category":"page"},{"location":"examples/","page":"Examples","title":"Examples","text":"Invertible recurrent inference machines (Putzky and Welling, 2019) (generic example)\nGenerative models with maximum likelihood via the change of variable formula (example)\nGlow: Generative flow with invertible 1x1 convolutions (Kingma and Dhariwal, 2018) (generic example, source)","category":"page"},{"location":"LICENSE/","page":"LICENSE","title":"LICENSE","text":"MIT License","category":"page"},{"location":"LICENSE/","page":"LICENSE","title":"LICENSE","text":"Copyright (c) 2020 SLIM group @ Georgia Institute of Technology","category":"page"},{"location":"LICENSE/","page":"LICENSE","title":"LICENSE","text":"Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:","category":"page"},{"location":"LICENSE/","page":"LICENSE","title":"LICENSE","text":"The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.","category":"page"},{"location":"LICENSE/","page":"LICENSE","title":"LICENSE","text":"THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Memory efficient inverible layers, networks and activation function for Machine learning.","category":"page"},{"location":"#InvertibleNetwroks.jl-documentation","page":"Home","title":"InvertibleNetwroks.jl documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This documentation is work in progress and is beeing actively populated.","category":"page"},{"location":"#About","page":"Home","title":"About","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"InvertibleNetworks.jl Is a package of invertible layers and networks for machine learning. The invertibility allow to backbropagate through the layers and networks without the need for storing the forward sate that is recomputed on the fly inverse propagating through it. This package is the first of its kind in julia.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package is developped and maintained by Felix J. Herrmann's SlimGroup at Georgia Institute of Technology. In particular the main contributors of this package are:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Philipp Witte, Microsoft Corporation (pwitte@microsoft.com)\nGabrio Rizzuti, Utrecht University (g.rizzuti@umcutrecht.nl)\nMathias Louboutin, Georgia Institute of Technology (mlouboutin3@gatech.edu)\nAli Siahkoohi, Georgia Institute of Technology (alisk@gatech.edu)","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install this package you can either directly install it from its url","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add https://github.com/slimgroup/InvertibleNetworks.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"or if you wish to have access to all slimgroup' softwares you can add our registry to have access to our packages in the standard julia way:","category":"page"},{"location":"","page":"Home","title":"Home","text":"] registry add https://Github.com/slimgroup/SLIMregistryJL.git\n] add InvertibleNetworks","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Yann Dauphin, Angela Fan, Michael Auli and David Grangier, \"Language modeling with gated convolutional networks\", Proceedings of the 34th International Conference on Machine Learning, 2017. https://arxiv.org/pdf/1612.08083.pdf\nLaurent Dinh, Jascha Sohl-Dickstein and Samy Bengio, \"Density estimation using Real NVP\",  International Conference on Learning Representations, 2017, https://arxiv.org/abs/1605.08803\nDiederik P. Kingma and Prafulla Dhariwal, \"Glow: Generative Flow with Invertible 1x1 Convolutions\", Conference on Neural Information Processing Systems, 2018. https://arxiv.org/abs/1807.03039\nKeegan Lensink, Eldad Haber and Bas Peters, \"Fully Hyperbolic Convolutional Neural Networks\", arXiv Computer Vision and Pattern Recognition, 2019. https://arxiv.org/abs/1905.10484\nPatrick Putzky and Max Welling, \"Invert to learn to invert\", Advances in Neural Information Processing Systems, 2019. https://arxiv.org/abs/1911.10914\nJakob Kruse, Gianluca Detommaso, Robert Scheichl and Ullrich Köthe, \"HINT: Hierarchical Invertible Neural Transport for Density Estimation and Bayesian Inference\", arXiv Statistics and Machine Learning, 2020. https://arxiv.org/abs/1905.10687","category":"page"},{"location":"#Related-work-and-publications","page":"Home","title":"Related work and publications","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The following publications use [InvertibleNetworks.jl]:","category":"page"},{"location":"","page":"Home","title":"Home","text":"[“Preconditioned training of normalizing flows for variational inference in inverse problems”]\npaper: https://arxiv.org/abs/2101.03709\npresentation\ncode: [FastApproximateInference.jl]\n[\"Parameterizing uncertainty by deep invertible networks, an application to reservoir characterization\"]\npaper: https://arxiv.org/abs/2004.07871\npresentation\ncode: https://github.com/slimgroup/Software.SEG2020\n[\"Generalized Minkowski sets for the regularization of inverse problems\"]\npaper: http://arxiv.org/abs/1903.03942\ncode: [SetIntersectionProjection.jl]","category":"page"},{"location":"#Acknowledgments","page":"Home","title":"Acknowledgments","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package uses functions from NNlib.jl, Flux.jl and Wavelets.jl","category":"page"}]
}
