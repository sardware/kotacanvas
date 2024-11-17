import gradio as gr
import numpy as np


def first(image: np.ndarray, r, g, b) -> np.ndarray:
    # Here you can manipulate the image
    buffer = image.copy().astype(np.float32)
    image[:, :, 0] = np.round(buffer[:, :, 0] * r / 100).astype(np.uint8)
    image[:, :, 1] = np.round(buffer[:, :, 1] * g / 100).astype(np.uint8)
    image[:, :, 2] = np.round(buffer[:, :, 2] * b / 100).astype(np.uint8)
    return image


if __name__ == '__main__':
    gr.Interface(
        fn=first,
        inputs=['image',
                gr.Slider(minimum=0,
                          maximum=100,
                          value=100,
                          label='R %'),
                gr.Slider(minimum=0,
                          maximum=100,
                          value=100,
                          label='G %'),
                gr.Slider(minimum=0,
                          maximum=100,
                          value=100,
                          label='B %')],
        outputs=['image'],
        live=True
    ).launch()
