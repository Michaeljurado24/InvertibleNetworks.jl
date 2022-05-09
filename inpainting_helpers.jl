using Flux

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



function create_autoencoder_net()
    model = Chain(
        Conv((3, 3), 1=>16, pad= Flux.SamePad(), relu),

        Conv((3, 3), 16=>32, pad= Flux.SamePad(), relu),

        Conv((3, 3), 32=>32, pad= Flux.SamePad(), relu), # <- condition extractor output layer

        Conv((3, 3), 32=>16, pad= Flux.SamePad(), relu),

        Conv((3, 3), 16=>1, pad= Flux.SamePad(), sigmoid)
    )
    return model
end