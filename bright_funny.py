import gradio as gr
import numpy as np
from PIL import Image, ImageFilter
from random import randrange

def bright_funny(image: np.ndarray) -> np.ndarray:
    print('This is bright and funny effect.')

    new_img = Image.fromarray(image)

    # Crop
    x, y = new_img.size
    matrix = 500
    sample = 4
    images = []

    for i in range(sample):
        x1 = randrange(0, x - matrix)
        y1 = randrange(0, y - matrix)
        # new_img = new_img.crop((x1, y1, x1 + matrix, y1 + matrix))
        images.append(new_img.crop((x1, y1, x1 + matrix, y1 + matrix)))

    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_img = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_img.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    # Bright
    new_img = new_img.point(lambda p: p * 2)
    return np.asarray(new_img)


if __name__ == '__main__':
    gr.Interface(
        fn=bright_funny,
        inputs=['image'],
        outputs=['image']
    ).launch()