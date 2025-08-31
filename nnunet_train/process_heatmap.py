from scipy.ndimage import gaussian_filter
import nibabel as nib
import numpy as np
import os


def parse_click_json(json_data):
    """
    Parse click data from the JSON format with version and points structure.

    Args:
        json_data (dict): JSON data containing version and points information.

    Returns:
        dict: Dictionary with 'tumor' and 'background' keys containing coordinate lists.
    """
    clicks = {'tumor': [], 'background': []}

    if 'points' in json_data:
        for point_data in json_data['points']:
            if 'point' in point_data and 'name' in point_data:
                coord = point_data['point']
                name = point_data['name']

                if name == 'tumor':
                    clicks['tumor'].append(coord)
                elif name == 'background':
                    clicks['background'].append(coord)

    return clicks

def generate_combined_click_heatmap(clicks, shape, tumor_sigma=3.0, background_sigma=3.0, tumor_intensity=1.0,
                                    background_intensity=-0.5):
    """
    Generate a combined 3D Gaussian heatmap from tumor and background clicks.
    Tumor clicks are positive values, background clicks are negative values.

    Args:
    clicks (dict): Dictionary with 'tumor' and 'background' keys containing coordinate lists.
    shape (tuple): Shape of the output volume (D, H, W).
    tumor_sigma (float): Standard deviation of the Gaussian for tumor clicks.
    background_sigma (float): Standard deviation of the Gaussian for background clicks.
    tumor_intensity (float): Maximum intensity for tumor clicks (positive).
    background_intensity (float): Maximum intensity for background clicks (negative).

    Returns:
    np.ndarray: 3D volume with combined Gaussian heatmaps at the specified coordinates.
    """

    # Initialize combined heatmap
    combined_heatmap = np.zeros(shape, dtype=np.float32)

    # Process tumor clicks (positive values)
    if clicks.get('tumor'):
        tumor_heatmap = np.zeros(shape, dtype=np.float32)
        for coord in clicks['tumor']:
            if 0 <= coord[0] < shape[0] and 0 <= coord[1] < shape[1] and 0 <= coord[2] < shape[2]:
                tumor_heatmap[tuple(coord)] = tumor_intensity

        # Apply Gaussian smoothing to tumor clicks
        tumor_heatmap = gaussian_filter(tumor_heatmap, sigma=tumor_sigma)
        combined_heatmap += tumor_heatmap

    # Process background clicks (negative values)
    if clicks.get('background'):
        background_heatmap = np.zeros(shape, dtype=np.float32)
        for coord in clicks['background']:
            if 0 <= coord[0] < shape[0] and 0 <= coord[1] < shape[1] and 0 <= coord[2] < shape[2]:
                background_heatmap[tuple(coord)] = abs(background_intensity)

        # Apply Gaussian smoothing to background clicks
        background_heatmap = gaussian_filter(background_heatmap, sigma=background_sigma)
        combined_heatmap -= background_heatmap

    return combined_heatmap


def save_combined_click_heatmap(clicks, output_path, input_pet, tumor_sigma=3.0, background_sigma=3.0,
                                tumor_intensity=1.0, background_intensity=-0.5):
    """
    Save a combined click heatmap as a single channel file.

    Args:
        clicks (dict): Dictionary with 'tumor' and 'background' keys containing coordinate lists.
        output_path (str): Directory to save the heatmap file.
        input_pet (str): Path to the PET image file for reference shape and affine.
        tumor_sigma (float): Standard deviation of the Gaussian for tumor clicks.
        background_sigma (float): Standard deviation of the Gaussian for background clicks.
        tumor_intensity (float): Maximum intensity for tumor clicks (positive).
        background_intensity (float): Maximum intensity for background clicks (negative).
    """
    pet_img = nib.load(input_pet)
    ref_shape = pet_img.shape
    ref_affine = pet_img.affine

    # Generate combined heatmap
    combined_heatmap = generate_combined_click_heatmap(
        clicks, ref_shape, tumor_sigma, background_sigma, tumor_intensity, background_intensity
    )

    # Create NIfTI image
    combined_nifti = nib.Nifti1Image(combined_heatmap, ref_affine)

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save as channel 2 (index 2) - channels 0 and 1 are CT and PET
    output_filename = f'{input_pet.split("/")[-1].split("_0000.nii.gz")[0]}_0002.nii.gz'
    output_filepath = os.path.join(output_path, output_filename)
    nib.save(combined_nifti, output_filepath)

    print(f"Combined click heatmap saved to: {output_filepath}")
    print(f"Tumor clicks: {len(clicks.get('tumor', []))}, Background clicks: {len(clicks.get('background', []))}")
    print(f"Heatmap value range: [{combined_heatmap.min():.3f}, {combined_heatmap.max():.3f}]")


