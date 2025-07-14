import torch
import numpy as np
import tifffile
import os
import gc
from math import ceil
from scipy import ndimage
from skimage.morphology import ball
from scipy.ndimage import binary_dilation
from skimage.measure import label, regionprops
import re

class CellSegmentationPipeline:
    def __init__(self, config):
        """
        Initialize the pipeline with configuration dictionary
        
        config should contain:
        - model_paths: dict with '3d' and '2d' keys
        - input_images: list of input image paths
        - output_folders: dict with folder names as keys
        - processing_params: dict with processing parameters
        - pixel_dimensions: list [pix_x, pix_y, pix_z] in um (optional, None to read from metadata)
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_device()
        
    def setup_device(self):
        """Setup device and memory configuration"""
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            torch.cuda.set_per_process_memory_fraction(0.8)
    
    def load_models(self):
        """Load 3D and 2D models"""
        print(f"Loading 3D model from {self.config['model_paths']['3d']}")
        self.model_3d = torch.jit.load(self.config['model_paths']['3d'], map_location=self.device)
        self.model_3d.eval()
        
        print(f"Loading 2D model from {self.config['model_paths']['2d']}")
        self.model_2d = torch.jit.load(self.config['model_paths']['2d'], map_location=self.device)
        self.model_2d.eval()
        print("Models loaded successfully")
    
    def load_3d_image(self, image_path):
        """Load and preprocess a 3D image from a TIFF file"""
        try:
            print(f"Loading 3D image from {image_path}")
            full_img = tifffile.imread(image_path)
            print(f"Original image shape: {full_img.shape}")
            
            # Handle 4D images by selecting the first channel
            if len(full_img.shape) == 4:
                print(f"4D image detected with shape {full_img.shape}, using first channel")
                full_img = full_img[:, 0, :, :] if full_img.shape[1] < full_img.shape[0] else full_img[0]
            
            # Normalize to [0, 1]
            print("Normalizing image to [0, 1]...")
            full_img = full_img.astype(np.float32)
            img_min, img_max = full_img.min(), full_img.max()
            full_img = (full_img - img_min) / (img_max - img_min + 1e-8)
            
            return full_img
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def process_stack(self, model, stack, stack_idx, total_stacks):
        """Process a single 3D stack on the GPU"""
        try:
            print(f"Processing stack {stack_idx+1}/{total_stacks}, shape: {stack.shape}")
            
            stack_tensor = torch.from_numpy(stack).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output_tensor = model(stack_tensor)
            
            output_stack = output_tensor.squeeze().cpu().numpy()
            
            del stack_tensor, output_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return output_stack
            
        except Exception as e:
            print(f"Error processing stack {stack_idx+1}: {e}")
            return np.zeros_like(stack)
        
    def process_3d_stacks(self, model, full_img, stack_depth=48, min_depth=16):
        """Process the 3D image by splitting into smaller stacks with padding"""
        depth, height, width = full_img.shape
        print(f"Processing 3D image of shape {full_img.shape} in stacks of depth {stack_depth}")
        
        num_stacks = ceil(depth / stack_depth)
        print(f"Total number of stacks: {num_stacks}")
        
        processed_stacks = []
        
        for stack_idx in range(num_stacks):
            start_idx = stack_idx * stack_depth
            end_idx = min(start_idx + stack_depth, depth)
            
            stack = full_img[start_idx:end_idx]
            original_stack_depth = stack.shape[0]
            
            # Add padding if stack depth is less than minimum required
            if original_stack_depth < min_depth:
                padding_needed = min_depth - original_stack_depth
                print(f"Stack {stack_idx+1} has depth {original_stack_depth}, padding with {padding_needed} slices")
                
                # Pad with zeros at the end
                padded_stack = np.zeros((min_depth, height, width), dtype=stack.dtype)
                padded_stack[:original_stack_depth] = stack
                
                output_stack = self.process_stack(model, padded_stack, stack_idx, num_stacks)
                
                # Remove padding from output
                output_stack = output_stack[:original_stack_depth]
            else:
                output_stack = self.process_stack(model, stack, stack_idx, num_stacks)
            
            processed_stacks.append(output_stack)
            gc.collect()
        
        print("Concatenating processed stacks...")
        result_3d = np.concatenate(processed_stacks, axis=0)
        
        if result_3d.shape != full_img.shape:
            print(f"Warning: Output shape {result_3d.shape} does not match input shape {full_img.shape}")
            result_3d = result_3d[:depth, :height, :width]
        
        print("Binarizing output...")
        result_3d = (result_3d > 0.2).astype(np.uint8) * 255
        
        return result_3d
    
    def process_2d_slices(self, model, full_img):
        """Process each 2D slice using the 2D model"""
        depth, height, width = full_img.shape
        print(f"Processing 2D slices of shape {full_img.shape}")
        processed_slices = []
        
        for z in range(depth):
            print(f"Processing slice {z+1}/{depth}")
            slice_2d = full_img[z]
            
            slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output_tensor = model(slice_tensor)
            
            output_slice = output_tensor.squeeze().cpu().numpy()
            output_slice = (output_slice > 0.2).astype(np.uint8) * 255
            processed_slices.append(output_slice)
            
            del slice_tensor, output_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print("Stacking processed 2D slices...")
        result_2d = np.stack(processed_slices, axis=0)
        return result_2d
    
    def extract_voxel_dimensions(self, image_path):
        """Extract or use provided voxel dimensions"""
        print(f"Determining voxel dimensions for: {image_path}")

        # Check if pixel dimensions are provided in config
        pixel_dimensions = self.config.get('pixel_dimensions', None)

        if pixel_dimensions is not None and len(pixel_dimensions) == 3:
            pixel_size_um_x, pixel_size_um_y, pixel_size_um_z = pixel_dimensions
            print(f"Using provided pixel dimensions: X={pixel_size_um_x:.3f}, Y={pixel_size_um_y:.3f}, Z={pixel_size_um_z:.3f} um")
        else:
            # Fall back to reading from metadata
            print("No pixel dimensions provided in config, reading from metadata...")
            pixel_size_um_x = None
            pixel_size_um_y = None
            pixel_size_um_z = None

            try:
                with tifffile.TiffFile(image_path) as tif:
                    if hasattr(tif, 'ome_metadata') and tif.ome_metadata is not None:
                        pixels = tif.ome_metadata.get('Image', {}).get('Pixels', {})
                        pixel_size_um_x = pixels.get('PhysicalSizeX')
                        pixel_size_um_y = pixels.get('PhysicalSizeY')
                        pixel_size_um_z = pixels.get('PhysicalSizeZ')
                        print(f"Found OME-XML metadata.")

                    if pixel_size_um_x is None or pixel_size_um_y is None:
                        if tif.pages and tif.pages[0].tags.get('ImageDescription'):
                            description = tif.pages[0].tags['ImageDescription'].value
                            
                            match_x = re.search(r"x_resolution=(\d+\.?\d*)", description, re.IGNORECASE)
                            match_y = re.search(r"y_resolution=(\d+\.?\d*)", description, re.IGNORECASE)
                            match_z = re.search(r"z_resolution=(\d+\.?\d*)", description, re.IGNORECASE)
                            match_voxel_size = re.search(r"voxel_size_um=(\d+\.?\d*)", description, re.IGNORECASE)
                            
                            if match_x: pixel_size_um_x = float(match_x.group(1))
                            if match_y: pixel_size_um_y = float(match_y.group(1))
                            if match_z: pixel_size_um_z = float(match_z.group(1))
                            if match_voxel_size and (pixel_size_um_x is None or pixel_size_um_y is None):
                                val = float(match_voxel_size.group(1))
                                pixel_size_um_x = pixel_size_um_x or val
                                pixel_size_um_y = pixel_size_um_y or val
                                pixel_size_um_z = pixel_size_um_z or val
                                
            except Exception as e:
                print(f"Could not read TIFF metadata: {e}")

            # Set defaults if not found
            if pixel_size_um_x is None: pixel_size_um_x = 1.0
            if pixel_size_um_y is None: pixel_size_um_y = 1.0
            if pixel_size_um_z is None: pixel_size_um_z = 1.0

            print(f"Extracted voxel dimensions from metadata: X={pixel_size_um_x:.3f}, Y={pixel_size_um_y:.3f}, Z={pixel_size_um_z:.3f} um")

        return pixel_size_um_x, pixel_size_um_y, pixel_size_um_z
    
    def get_6_connectivity_structure(self):
        """Get 6-connectivity structuring element"""
        return ndimage.generate_binary_structure(3, 1)
    
    def remove_largest_component_2d(self, binary_image_slice):
        """Remove the largest component from a 2D slice"""
        binary_bool = binary_image_slice > 0
        labeled_image = label(binary_bool)
        regions = regionprops(labeled_image)
        
        if not regions:
            return np.zeros_like(binary_image_slice)
        
        largest_region = max(regions, key=lambda x: x.area)
        output_image = np.copy(binary_image_slice)
        output_image[labeled_image == largest_region.label] = 0
        
        return output_image
    
    def process_single_image(self, image_path):
        """Process a single image through the complete pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Get base filename for output files
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Ensure output folders exist
        for folder_path in self.config['output_folders'].values():
            os.makedirs(folder_path, exist_ok=True)
        
        # Load image
        full_img = self.load_3d_image(image_path)
        if full_img is None:
            print(f"Skipping {image_path} due to loading error")
            return False
        
        # Get voxel dimensions
        pixel_size_um_x, pixel_size_um_y, pixel_size_um_z = self.extract_voxel_dimensions(image_path)
        voxel_volume_um3 = pixel_size_um_x * pixel_size_um_y * pixel_size_um_z
        
        # Step 1: Model Segmentation
        print("\n--- Step 1: Model Segmentation ---")
        result_3d = self.process_3d_stacks(self.model_3d, full_img, 
                                           self.config['processing_params']['stack_depth'])
        result_2d = self.process_2d_slices(self.model_2d, full_img)
        
        # Combine results
        print("Combining 3D and 2D segmentation results...")
        combined_result = np.logical_or(result_3d > 0, result_2d > 0).astype(np.uint8) * 255
        
        # Save segmentation result
        segmentation_path = os.path.join(self.config['output_folders']['segmentation'], f"{base_name}_segmentation.tif")
        tifffile.imwrite(segmentation_path, combined_result)
        print(f"Segmentation saved to: {segmentation_path}")
        
        # Step 2: Cell Identification
        print("\n--- Step 2: Cell Identification ---")
        binary_skeleton = (combined_result > 0).astype(np.uint8)
        
        # Dilate membranes
        cell_dilation_radius = self.config['processing_params']['cell_dilation_radius']
        print(f"Dilating membranes with ball of radius {cell_dilation_radius}")
        structuring_element = ball(cell_dilation_radius, dtype=np.uint8)
        dilated_skeleton = binary_dilation(binary_skeleton, structure=structuring_element)
        
        # Pad one pixel in each direction (H+2, W+2) for the ZY and ZX planes
        print("Padding dilated skeleton...")
        padded_dilated_skeleton = np.pad(dilated_skeleton, 
                                         ((0, 0), (1, 1), (1, 1)), 
                                         mode='constant', constant_values=0)
        
        # Invert and remove largest component
        print("Inverting padded dilated volume and removing largest component...")
        inverted_volume_padded = (padded_dilated_skeleton == 0).astype(np.uint8)
        binary_cells_padded = np.zeros_like(inverted_volume_padded, dtype=np.uint8)
        
        for z in range(inverted_volume_padded.shape[0]):
            binary_cells_padded[z, :, :] = self.remove_largest_component_2d(inverted_volume_padded[z, :, :])
        
        # Remove the padding to restore original dimensions (H, W)
        print("Removing padding...")
        binary_cells = binary_cells_padded[:, 1:-1, 1:-1]
        
        # Label cells in 3D
        print("Labeling cells in 3D...")
        s_6_connectivity = self.get_6_connectivity_structure()
        labeled_cells_3d, num_features = ndimage.label(binary_cells, structure=s_6_connectivity)
        print(f"Found {num_features} distinct 3D cells")
        
        # Dilate cells
        print("Dilating cells...")
        dilation_radius = self.config['processing_params']['cell_final_dilation_radius']
        se_ball = ball(dilation_radius).astype(int)
        
        dilated_temp = ndimage.grey_dilation(labeled_cells_3d, structure=se_ball)
        binary_all_cells = (labeled_cells_3d > 0).astype(np.uint8)
        dilated_binary_all_cells = ndimage.binary_dilation(binary_all_cells, structure=se_ball, iterations=1)
        
        dilated_cells_3d = np.zeros_like(labeled_cells_3d, dtype=labeled_cells_3d.dtype)
        dilated_cells_3d[dilated_binary_all_cells] = dilated_temp[dilated_binary_all_cells]
        dilated_cells_3d = np.maximum(dilated_cells_3d, labeled_cells_3d)
        
        # Step 3: Volume Filtering
        print("\n--- Step 3: Volume Filtering ---")
        volume_threshold_um3 = self.config['processing_params']['volume_threshold_um3']
        print(f"Filtering cells with volume < {volume_threshold_um3:.2f} um³")
        
        # Calculate volumes and filter
        regions = regionprops(dilated_cells_3d)
        filtered_cells_3d = np.zeros_like(dilated_cells_3d, dtype=dilated_cells_3d.dtype)
        valid_labels = []
        
        for region in regions:
            volume_um3 = region.area * voxel_volume_um3
            if volume_um3 >= volume_threshold_um3:
                valid_labels.append(region.label)
        
        print(f"Keeping {len(valid_labels)} cells with volume >= {volume_threshold_um3:.2f} um³")
        
        # Set labels of cells below threshold to 0
        mask = np.isin(dilated_cells_3d, valid_labels)
        filtered_cells_3d[mask] = dilated_cells_3d[mask]
        
        # Second labeling to ensure consecutive labels
        print("Relabeling filtered cells...")
        final_labeled_cells_3d, num_final_features = ndimage.label(filtered_cells_3d > 0, structure=s_6_connectivity)
        print(f"After relabeling, found {num_final_features} distinct cells")
        
        # Save final result
        final_output_path = os.path.join(self.config['output_folders']['final'], f"{base_name}_cells.tif")
        tifffile.imwrite(final_output_path, final_labeled_cells_3d, 
                         compression='zlib', photometric='minisblack')
        print(f"Final result saved to: {final_output_path}")
        
        # Generate summary statistics
        final_regions = regionprops(final_labeled_cells_3d)
        
        print(f"\n--- Summary Statistics ---")
        print(f"Total cells identified: {len(final_regions)}")
        if final_regions:
            cell_volumes_voxels = [prop.area for prop in final_regions]
            cell_volumes_um3 = [v * voxel_volume_um3 for v in cell_volumes_voxels]
            print(f"Cell volume range (um³): {min(cell_volumes_um3):.2f} - {max(cell_volumes_um3):.2f}")
            print(f"Average cell volume (um³): {np.mean(cell_volumes_um3):.2f}")
            print(f"Median cell volume (um³): {np.median(cell_volumes_um3):.2f}")
            print(f"Cell size range (voxels): {min(cell_volumes_voxels):.0f} - {max(cell_volumes_voxels):.0f}")
            print(f"Average cell size (voxels): {np.mean(cell_volumes_voxels):.2f}")
            print(f"Median cell size (voxels): {np.median(cell_volumes_voxels):.2f}")
        else:
            print("No cells passed the volume threshold.")
        
        return True
        
    def run_pipeline(self):
        """Run the complete pipeline on all images"""
        print("Starting 3D Cell Segmentation Pipeline")
        print(f"Processing {len(self.config['input_images'])} images")
        
        # Load models once
        self.load_models()
        
        # Process each image
        successful_count = 0
        for image_path in self.config['input_images']:
            if self.process_single_image(image_path):
                successful_count += 1
        
        print(f"\n{'='*60}")
        print(f"Pipeline completed: {successful_count}/{len(self.config['input_images'])} images processed successfully")
        print(f"{'='*60}")
