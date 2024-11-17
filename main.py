import gradio as gr
import numpy as np


def first(image: np.ndarray) -> np.ndarray:
    # Here you can manipulate the image
    return image


if __name__ == '__main__':
    gr.Interface(
        fn=first,
        inputs=['image'],
        outputs=['image']
    ).launch()
