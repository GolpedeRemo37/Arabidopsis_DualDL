from Main_pipeline import CellSegmentationPipeline

# Configuration
config = {
    'model_paths': {
        '3d': r"E:\PHD\phd_env\Proyectos\AIRM\Codes\3D_UNet.pt",
        '2d': r"E:\PHD\phd_env\Proyectos\AIRM\Codes\2D_UNet.pt"
    },
    'input_images': [
        r"E:\PHD\phd_env\Proyectos\AIRM\PNAS\PNAS\plant1\processed_tiffs\0hrs_plant1_trim-acylYFP.tif"
    ],
    'output_folders': {
        'segmentation': r"E:\PHD\phd_env\Proyectos\AIRM\Codes\Plant_1_Processed\0 Model segmentation",
        'final': r"E:\PHD\phd_env\Proyectos\AIRM\Codes\Plant_1_Processed\1 Cell Segmentation"
    },
    'processing_params': {
        'stack_depth': 16,
        'cell_dilation_radius': 1,
        'cell_final_dilation_radius': 2,
        'volume_threshold_um3': 1.5
    },
    'pixel_dimensions': [0.5, 0.5, 1.0]  # [pix_x, pix_y, pix_z] in micrometers
    # Use None to read from metadata: 'pixel_dimensions': None
}

# Run the pipeline
if __name__ == "__main__":
    pipeline = CellSegmentationPipeline(config)
    pipeline.run_pipeline()
