import argparse

import numpy as np

from .drawing import generate_code_image
from .python_bmp import generate_bmp_from_image_data_and_source_code


def generate_bmp_file_from_source_code(code: str, output_file: str):
    code = code.strip()
    image = generate_code_image(code)
    image_data = np.asarray(image)

    bmp_data = generate_bmp_from_image_data_and_source_code(image_data, code)
    
    with open(output_file, 'wb') as file:
        file.write(bmp_data)


def main():
    parser = argparse.ArgumentParser(description='Convert a Python source file to an executable BMP image.')
    parser.add_argument('input_file', help='Path to the Python source file to convert')
    parser.add_argument('output_file', help='Path to save the resulting BMP image')

    args = parser.parse_args()

    # Read the Python source code from the input file
    with open(args.input_file, 'r') as file:
        source_code = file.read()

    # Create the image from the source code
    print('Generating the image, this can take some seconds...')
    generate_bmp_file_from_source_code(source_code, args.output_file)

if __name__ == '__main__':
    main()
