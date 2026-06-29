from inference.infer import SRInference

model = SRInference(
    "checkpoints/best_srresnet.pth"
)

thermal = model.load_image(
    "datasets/raw/landsat9_auto.tif"
)

output = model.predict(thermal)

model.save(
    output,
    "outputs/sr_output.tif"
)