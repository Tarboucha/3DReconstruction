from CameraPoseEstimation.pipeline2 import MainPosePipeline

pose_pipeline = MainPosePipeline()


result = pose_pipeline.process_monument_reconstruction(
    matches_pickle_file=matches_pickle,
    output_directory='./results/poses/',
    chosen_images=None  # Auto-select best pair, or specify tuple like ('img1.jpg', 'img2.jpg')
)