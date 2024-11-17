import gradio as gr
import numpy as np
from skimage.transform import swirl
from PIL import Image

def apply_swirl(image: np.ndarray, strength: float, angle: float) -> np.ndarray:
    # Get dimensions and center
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = min(height, width) // 2

    # Apply the swirl transformation
    swirled_image = swirl(image, center=center, rotation=angle, strength=strength, radius=radius, preserve_range=True)
    swirled_image = swirled_image.astype(np.uint8)

    # Convert BGR to RGB for display (assuming input is BGR format)
    if swirled_image.shape[2] == 3:
        swirled_image = swirled_image[...,::-1]

    return swirled_image

def main():
    interface = gr.Interface(
        fn=apply_swirl,
        inputs=['image', gr.Slider(0, 10, step=0.1, label="Strength"), gr.Slider(-360, 360, step=1, label="Angle")],
        outputs='image'
    )
    interface.launch()

if __name__ == '__main__':
    main()