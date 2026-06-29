from inference.pipeline import IRPipeline

pipeline = IRPipeline(
    "checkpoints/best_srresnet.pth",
    "checkpoints/best_colorizer.pth"
)

pipeline.run(
    "sample_input/sample_00187",
    "outputs/prediction.png"
)