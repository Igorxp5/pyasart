import re
import os
import math

from typing import Tuple

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from .code_style_parser import get_code_token_colors
from .python_bmp import resize_for_python_bmp, mask_valid_RGB_colors, \
    convert_RGB_image_for_python_bmp

IMAGE_MODE = 'RGB'

# Font Constants
FONTS_DIR = os.path.join(os.path.dirname(__file__), 'fonts')
FONT_PATH = os.path.abspath(os.path.join(FONTS_DIR, 'Hack-Regular.ttf'))

# Image drawing constants
MIN_IMAGE_WIDTH = 800

IMAGE_BACKGROUND = 0, 43, 54
IMAGE_HORIZONTAL_PADDING = 25
IMAGE_VERTICAL_PADDING = 25

## Code drawing
TEXT_SIZE = 14
TEXT_FONT = ImageFont.truetype(FONT_PATH, TEXT_SIZE)
INDENT_WIDTH = 40
LINE_SPACING = 4
WHITESPACE_WIDTH = 10

# Window Topbar constants

## Window Button
WINDOW_BUTTONS_TOP = 20
WINDOW_BUTTONS_RIGHT = 15
WINDOW_BUTTON_COLOR = 111, 126, 127
WINDOW_BUTTON_WIDTH = 16
WINDOW_BUTTON_HEIGHT = 16
WINDOW_BUTTON_MARGIN = 10

## Window Topbar
WINDOW_TOPBAR_HEIGHT = WINDOW_BUTTONS_TOP + WINDOW_BUTTON_HEIGHT + WINDOW_BUTTONS_TOP
WINDOW_TOPBAR_BACKGROUND = 0, 31, 38


def get_text_size(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
    _, _, right, bottom = font.getbbox(text)
    width, height = right, bottom
    return math.ceil(width), math.ceil(height)


def draw_code(image: Image.Image, code: str):
    token_colors = get_code_token_colors(code)

    draw = ImageDraw.Draw(image)
    draw.font = TEXT_FONT

    base_y = WINDOW_TOPBAR_HEIGHT + IMAGE_VERTICAL_PADDING
    base_x = IMAGE_HORIZONTAL_PADDING

    last_line_text = ''
    whitespace_regex = re.compile(r'^ +$')
    for token, color in token_colors:
        whitespace_match = whitespace_regex.match(token)
        if whitespace_match:
            base_x += WHITESPACE_WIDTH * token.count(' ')
        elif token == '\n':
            base_y += LINE_SPACING
            if last_line_text == '\n':
                last_line_text = ' '
            base_y += get_text_size(last_line_text, TEXT_FONT)[1]
            base_x = IMAGE_HORIZONTAL_PADDING
            last_line_text = ''
        else:
            draw.text((base_x, base_y), token, fill=color)
            base_x += get_text_size(token, TEXT_FONT)[0]
        last_line_text += token

    base_x = IMAGE_HORIZONTAL_PADDING


def draw_window_buttons(image: Image.Image):
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (image.size[0], WINDOW_TOPBAR_HEIGHT)], fill=WINDOW_TOPBAR_BACKGROUND)

    close_x1 = image.size[0] - WINDOW_BUTTONS_RIGHT - WINDOW_BUTTON_WIDTH
    close_y1 = WINDOW_BUTTONS_TOP
    close_x2 = close_x1 + WINDOW_BUTTON_WIDTH
    close_y2 = close_y1 + WINDOW_BUTTON_HEIGHT
    draw.line([(close_x1, close_y1), (close_x2, close_y2)], fill=WINDOW_BUTTON_COLOR)
    draw.line([(close_x1, close_y2), (close_x2, close_y1)], fill=WINDOW_BUTTON_COLOR)
    
    maximize_x1 = close_x1 - WINDOW_BUTTON_MARGIN - WINDOW_BUTTON_WIDTH
    maximize_y1 = WINDOW_BUTTONS_TOP
    maximize_x2 = maximize_x1 + WINDOW_BUTTON_WIDTH
    maximize_y2 = maximize_y1 + WINDOW_BUTTON_HEIGHT

    draw.rectangle([(maximize_x1, maximize_y1), (maximize_x2, maximize_y2)],
                   fill=WINDOW_TOPBAR_BACKGROUND, outline=WINDOW_BUTTON_COLOR)
    
    minimize_x1 = maximize_x1 - WINDOW_BUTTON_MARGIN - WINDOW_BUTTON_WIDTH
    minimize_y1 = WINDOW_BUTTONS_TOP + WINDOW_BUTTON_HEIGHT // 2
    minimize_x2 = minimize_x1 + WINDOW_BUTTON_WIDTH
    minimize_y2 = minimize_y1

    draw.line([(minimize_x1, minimize_y1), (minimize_x2, minimize_y2)], fill=WINDOW_BUTTON_COLOR)


def get_code_image_size(code: str) -> Tuple[int, int]:
    width = 800
    height = WINDOW_TOPBAR_HEIGHT + IMAGE_VERTICAL_PADDING
    lines = code.splitlines()

    for line in lines:
        whitespaces = line.count(' ')
        tabs = line.count('\t')
        only_text = re.sub(r'[ \t]+', '', line) or ' '
        text_width, text_height = get_text_size(only_text, TEXT_FONT)
        text_width += whitespaces * WHITESPACE_WIDTH + tabs * INDENT_WIDTH
        width = max(width, IMAGE_HORIZONTAL_PADDING * 2 + text_width)
        height += text_height + LINE_SPACING

    height += IMAGE_VERTICAL_PADDING * 2

    width = resize_for_python_bmp(width)
    height = resize_for_python_bmp(height)
    return width, height


def generate_code_image(code: str) -> Image.Image:
    """Generate a PIL.Image.Image object drawing the source code"""
    image_size = get_code_image_size(code)
    image = Image.new(IMAGE_MODE, image_size, color=IMAGE_BACKGROUND)

    draw_code(image, code)
    draw_window_buttons(image)

    image_data = np.asarray(image)
    image_data = convert_RGB_image_for_python_bmp(image_data)

    assert np.all(mask_valid_RGB_colors(image_data)), 'Could not produce a valid Python/BMP image'

    image = Image.fromarray(image_data)

    return image
