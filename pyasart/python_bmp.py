# This module has functions to convert and approximate bytes to be supported
# by Python and BMP file readers. Python requires the source code be encoded in UTF8.
# Other encodings are available, but CPython expects the encoding line be in at first line,
# this is impossible due to BMP file specification requirement of signature bytes (b'BM')
# at 0th byte of the file.
# BMP file specification: https://www.ece.ualberta.ca/~elliott/ee552/studentAppNotes/2003_w/misc/bmp_file_format/bmp_file_format.htm
# Python source code encoding: https://peps.python.org/pep-0263/

import struct

from multiprocessing import Manager, freeze_support, cpu_count

import tqdm
import colour
import numpy as np
import numpy.typing as npt

from .optimizer import init_adam_optimizer, step_adam_optimizer
from .clusterizer import kmeans_centroids

NOT_ALLOWED_UTF8_BYTES = [
    0x9,  # '\t' (horizontal tab)
    0xA,  # '\n' (new line)
    0xC,  # '\x0c' (new page)
    0xD   # '\r' (carriage return)
]
NOT_ALLOWED_PIXEL_DATA_BYTES = NOT_ALLOWED_UTF8_BYTES + [
    0x00, # '\x00' (null byte)
    0x22  # '"' (double-quote)
]
TOTAL_K_MEANS_CENTROIDS = 4 * 36


def resize_for_python_bmp(size: int):
    """Resizes width or height to be compible with a Python/BMP file"""
    current_size = size
    new_size = 0
    shift = 0
    while current_size > 0:
        byte = current_size & 0xFF  # Gets the least significant byte
        # Can't have break line, carriage return in size value
        # Also non UTF8 characters are not allowed for a Python/BMP file
        if byte in NOT_ALLOWED_UTF8_BYTES:
            byte = 0x0E
        elif byte >= 0x80:
            byte = 0
            current_size = (current_size & ~0xFF) + 0x100
        new_size += byte << shift
        current_size = current_size >> 8
        shift += 8
    return new_size


def rgb_image_to_bmp(rgb_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Convert an RGB image to the BMP format by rearranging the color channels, applying padding and flipping.

    BMP image format requires pixel data to be arranged in BGR format (not RGB) and to be padded so that
    each row of the image data is a multiple of 4 bytes. Also it flips the array vertically.
    """
    image_data = rgb_image[:, :, ::-1]  # RGB to BGR

    width, height = rgb_image.shape[1], rgb_image.shape[0]

    # Calculate padding needed to make each row a multiple of 4 bytes
    row_bytes = width * 3  # 3 bytes per pixel (RGB)
    padding = (4 - (row_bytes % 4)) % 4
    if padding > 0:
        padded_rgb_array = np.zeros((height, width * 3 + padding), dtype=np.uint8)
        padded_rgb_array[:, :width * 3] = image_data.reshape(height, width * 3)
        image_data = padded_rgb_array
    
    # Flip the array vertically (since scan lines are stored bottom-to-top)
    image_data = np.flipud(image_data)
    
    return image_data


def filesize_for_python_bmp(offset_pixel_data: int, image_total_bytes: int, code_payload: bytes):
    """
    Calculate the BMP header file size including pixel data and additional payload.

    The BMP file size needs to include the size of the pixel data, the file header, and any additional payload.
    This function computes the file size with an extra payload to facilitate a specific use case: making the 
    signature byte "BM", a variable for Python. This approach adds b'=0#' at end of file size bytes to make
    sure the first line is valid for Python.
    """
    filesize = offset_pixel_data + image_total_bytes + len(code_payload)
    filesize = ((filesize >> 24) + 1) << 24
    filesize |= 0x23303D  # Add b'=0#' at filesize bytes
    return filesize


def generate_bmp_from_image_data_and_source_code(rgb_image: npt.NDArray[np.uint8],
                                                 code: str) -> bytes:
    width, height = rgb_image.shape[:-1][::-1]
    image_data = rgb_image_to_bmp(rgb_image)
    total_used_colors = np.unique(rgb_image, axis=-2).shape[0]
    code_payload = code.encode('utf-8')

    # Make the Pixel Data chunk, a multiline string for the Python interpreter
    pixel_data_seperators = b'\n"""', b'"""\n\n'
    header_size = 14
    info_header_size = 40
    offset_pixel_data = header_size + info_header_size + len(pixel_data_seperators[0])
    
    filesize = filesize_for_python_bmp(offset_pixel_data, image_data.size, code_payload)

    bmp_fileformat = '<2sL4xLLLLHHLLLLLL'
    bmp_values = (
        b'BM',  # Signature
        filesize,
        offset_pixel_data,
        info_header_size,
        width,
        height,
        1,  # Planes
        24,  # Bits per pixel
        0,  # Compression
        0,  # ImageSize (0 if Compression = 0)
        1,  # XpixelsPerM
        1,  # YpixelsPerM
        total_used_colors,
        0  # Important Colors (all)
    )

    bmp_data = struct.pack(bmp_fileformat, *bmp_values)
    bmp_data += pixel_data_seperators[0] + image_data.tobytes()
    bmp_data += pixel_data_seperators[1] + code_payload + b'\n'

    return bmp_data

def get_all_valid_RGB_colors() -> npt.NDArray[np.uint32]:
    color_index = np.arange(2**24, dtype=np.uint32)

    rgb_r = color_index >> 16
    rgb_g = (color_index & 0xFF00) >> 8
    rgb_b = color_index & 0xFF

    rgb = np.stack((rgb_r, rgb_g, rgb_b), axis=1)

    return rgb[mask_valid_RGB_colors(rgb)]


def get_all_valid_Lab_colors() -> npt.NDArray[np.float32]:
    return RGB_to_Lab(get_all_valid_RGB_colors())


def get_valid_Lab_centroids() -> npt.NDArray[np.float32]:
    lab_colors = get_all_valid_Lab_colors()
    lab_colors = kmeans_centroids(lab_colors, TOTAL_K_MEANS_CENTROIDS, 1000)
    rgb_colors = Lab_to_RGB(lab_colors)
    return lab_colors[mask_valid_RGB_colors(rgb_colors)].astype(np.float32)


def RGB_to_Lab(rgb) -> npt.NDArray[np.float32]:
    srgb = rgb / 255.0
    xyz = colour.sRGB_to_XYZ(srgb)
    return colour.XYZ_to_Lab(xyz).astype(np.float32)


def Lab_to_RGB(lab) -> npt.NDArray[np.uint8]:
    xyz = colour.Lab_to_XYZ(lab)
    srgb = np.clip(colour.XYZ_to_sRGB(xyz), a_min=0, a_max=1)
    return np.round(srgb * 255).astype(np.uint8)


def mask_valid_RGB_colors(rgb_colors: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """
    Checks if a Numpy array containing RGB colors are valid for a Python/BMP file.

    The allowed RGB colors are those ones that are also the representation of a UTF8 character
    when the color is converted BGR (BMP file specifies RGB values are stored backwards). Also,
    some characters are not allowed as '\"' (double quote), '\r', '\n'. Since the Pixel Data
    is going to be encapsulated in multiline Python string, they would mess up the generation
    of a valid code.
    """
    rgb_colors_reshaped = rgb_colors.reshape(-1, 3)

    rgb_r = rgb_colors_reshaped[:,0]
    rgb_g = rgb_colors_reshaped[:,1]
    rgb_b = rgb_colors_reshaped[:,2]
    
    is_utf8_color = (rgb_r < 128) & (rgb_g < 128) & (rgb_b < 128)
    is_utf8_color |= (
          (rgb_b >= 0b11000000) & (rgb_b <= 0b11011111)
        & (rgb_g >= 0b10000000) & (rgb_g <= 0b10111111)
        & (rgb_r < 128)
        & ((((rgb_b & 0b11111) << 6) + (rgb_g & 0b111111)) >= 0x80)
    )
    is_utf8_color |= (
          (rgb_b < 128)
        & (rgb_g >= 0b11000000) & (rgb_g <= 0b11011111)
        & (rgb_r >= 0b10000000) & (rgb_r <= 0b10111111)
        & ((((rgb_g & 0b11111) << 6) + (rgb_r & 0b111111)) >= 0x80)
    )
    is_utf8_color |= (
        (rgb_b >= 0b11100000) & (rgb_b <= 0b11101111)
        & (rgb_g >= 0b10000000) & (rgb_g <= 0b10111111)
        & (rgb_r >= 0b10000000) & (rgb_r <= 0b10111111)
        & ((((rgb_b & 0b1111) << 12) + ((rgb_g & 0b111111) << 6) + (rgb_r & 0b111111)) >= 0x800)
    )

    for byte_number in NOT_ALLOWED_PIXEL_DATA_BYTES:
       is_utf8_color &= (rgb_r != byte_number) & (rgb_g != byte_number) & (rgb_b != byte_number)

    return is_utf8_color.reshape(*rgb_colors.shape[:-1])


def convert_RGB_image_for_python_bmp(rgb_image: npt.NDArray[np.uint8], learning_rate=0.01, epochs=350, derivate_h=1e-06) -> npt.NDArray[np.uint8]:
    """
    Converts RGB colors in an image to the closest colors in the BMP UTF-8 color space.

    This function takes an RGB image and adjusts the colors that are not compliant with 
    the BMP UTF-8 color space to the closest colors that are compliant. It uses a gradient
    descent-like optimization approach to find the closest colors in the BMP UTF-8 color space.
    The optimization process iterates through random color choices and refines them using 
    gradient information to minimize the color difference (CIEDE2000 Delta E) while ensuring the
    colors conform to the BMP UTF-8 color space.
    """
    progress_bar = tqdm.tqdm(
        total=TOTAL_K_MEANS_CENTROIDS,
        bar_format='{l_bar}{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}, ' '{rate_fmt}{postfix}]',
        desc='Generating...'
    )

    # Check which colors in the RGB image are already BMP UTF-8 compliant
    is_bmp_utf8_color = mask_valid_RGB_colors(rgb_image)

    # If all colors are BMP UTF-8 compliant, return the original image
    if np.all(is_bmp_utf8_color):
        return rgb_image

    # Extract unique non-BMP UTF-8 colors from the image
    non_bmp_utf8_colors = np.unique(rgb_image[~is_bmp_utf8_color], axis=-2)

    # Convert non-BMP UTF-8 colors to Lab color space
    non_bmp_utf8_lab_colors = RGB_to_Lab(non_bmp_utf8_colors)

    closest_colors = np.zeros(non_bmp_utf8_lab_colors.shape)
    delta_E_closest_colors = colour.delta_E(non_bmp_utf8_lab_colors, closest_colors)

    freeze_support()  # For Windows support

    n_workers = cpu_count()

    valid_lab_centroids = get_valid_Lab_centroids()

    progress_bar.update()

    update_unit = (TOTAL_K_MEANS_CENTROIDS - 1) / valid_lab_centroids.shape[0]
    with Manager() as manager:
        with manager.Pool(processes=n_workers) as pool:
            task_params = [
                (non_bmp_utf8_lab_colors, initial_color, epochs, learning_rate, derivate_h)
                for initial_color in valid_lab_centroids
            ]
            task_iterator = pool.imap_unordered(_color_optimizer_worker, task_params)
            for branch_current_color in task_iterator:
                # Update closest_colors based on the results found in the episode
                new_diff = colour.delta_E(non_bmp_utf8_lab_colors, branch_current_color)
                is_closer_than_before = new_diff < delta_E_closest_colors
                delta_E_closest_colors[is_closer_than_before] = new_diff[is_closer_than_before]
                closest_colors[is_closer_than_before] = branch_current_color[is_closer_than_before]
                progress_bar.set_description(f'Generating (E = {np.mean(delta_E_closest_colors):.2f})')
                progress_bar.update(update_unit)

    # Convert the closest Lab colors back to RGB
    rgb_closest_colors = Lab_to_RGB(closest_colors)

    # Replace non-BMP UTF-8 colors in the original image with their closest BMP UTF-8 colors
    result_rgb_image = rgb_image.copy()
    for i, non_bmp_utf8_color in enumerate(non_bmp_utf8_colors):
        mask = np.all(rgb_image == non_bmp_utf8_color, axis=-1)
        result_rgb_image[mask] = rgb_closest_colors[i]

    return result_rgb_image



def delta_E_gradient(color_a, color_b, derivate_h):
    delta_e = colour.delta_E(color_a, color_b)
    grad_L = (colour.delta_E(color_a, color_b + np.array([derivate_h, 0, 0])) - delta_e) / derivate_h
    grad_a = (colour.delta_E(color_a, color_b + np.array([0, derivate_h, 0])) - delta_e) / derivate_h
    grad_b = (colour.delta_E(color_a, color_b + np.array([0, 0, derivate_h])) - delta_e) / derivate_h
    return np.stack((grad_L, grad_a, grad_b), axis=-1)


def _color_optimizer_worker(args):
    input_colors, initial_color, epochs, learning_rate, derivate_h = args

    branch_current_color = np.tile(initial_color, (input_colors.shape[0], 1))
    adam_params = init_adam_optimizer(branch_current_color.shape)

    diff_step = 0

    for epoch in range(epochs):
        # Compute gradients
        gradient = delta_E_gradient(input_colors, branch_current_color, derivate_h)

        # Update colors based on the gradient
        diff_step, adam_params = step_adam_optimizer(*adam_params, gradient, diff_step, epoch, learning_rate)

        # Apply the diff step
        new_branch_current_color = branch_current_color + diff_step
        is_bmp_utf8_current_color = mask_valid_RGB_colors(Lab_to_RGB(new_branch_current_color))

        branch_current_color[is_bmp_utf8_current_color] = new_branch_current_color[is_bmp_utf8_current_color]

    return branch_current_color
