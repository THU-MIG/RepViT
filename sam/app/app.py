import os

import gradio as gr
import numpy as np
import torch
from repvit_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import ImageDraw
from utils.tools import box_prompt, format_results, point_prompt
from utils.tools_gradio import fast_process

# Most of our demo code is from [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM). Huge thanks for AN-619.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
sam_checkpoint = "weights/repvit_sam.pt"
model_type = "repvit"

repvit_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
repvit_sam = repvit_sam.to(device=device)
repvit_sam.eval()

mask_generator = SamAutomaticMaskGenerator(repvit_sam)
predictor = SamPredictor(repvit_sam)

# Description
title = "<center><strong><font size='8'>RepViT-SAM<font></strong></center>"

description_e = """This is a demo of [RepViT-SAM](https://github.com/THU-MIG/RepViT).

                   We will provide box mode soon. 

                   Enjoy!
                
              """

description_p = """ Instructions for point mode

                0. Restart by click the Restart button
                1. Select a point with Add Mask for the foreground (Must)
                2. Select a point with Remove Area for the background (Optional)
                3. Click the Start Segmenting.

                Github [link](https://github.com/THU-MIG/RepViT)

              """

examples = [
    ["app/assets/picture3.jpg"],
    ["app/assets/picture4.jpg"],
    ["app/assets/picture6.jpg"],
    ["app/assets/picture1.jpg"],
]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

def segment_with_points(
    image,
    original_image,
    input_size=1024,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_point_label = np.array(global_point_label)

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, image

    nd_image = np.array(original_image.resize((new_w, new_h)))
    predictor.set_image(nd_image)
    masks, scores, logits = predictor.predict(
        point_coords=scaled_points,
        point_labels=scaled_point_label,
        multimask_output=False,
    )

    results = format_results(masks, scores, logits, 0)

    annotations, _ = point_prompt(
        results, scaled_points, scaled_point_label, new_h, new_w
    )
    annotations = np.array([annotations])

    fig = fast_process(
        annotations=annotations,
        image=image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        bbox=None,
        use_retina=use_retina,
        withContours=withContours,
    )

    global_points = []
    global_point_label = []
    # return fig, None
    return fig, original_image.resize((new_w, new_h))


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15 * ((max(image.width, image.height)) / 1024), (255, 255, 0) if label == "Add Mask" else (
        255,
        0,
        255,
    )
    global_points.append([x, y])
    global_point_label.append(1 if label == "Add Mask" else 0)

    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    return image


cond_img_e = gr.Image(label="Input", value=default_example[0], type="pil")
cond_img_p = gr.Image(label="Input with points", value=default_example[0], type="pil")

segm_img_e = gr.Image(label="Segmented Image", interactive=False, type="pil")
segm_img_p = gr.Image(
    label="Segmented Image with points", interactive=True, type="pil"
)

global_points = []
global_point_label = []

input_size_slider = gr.components.Slider(
    minimum=512,
    maximum=1024,
    value=1024,
    step=64,
    label="Input_size",
    info="Our model was trained on a size of 1024",
)

with gr.Blocks(css=css, title="RepViT-SAM") as demo:
    from PIL import Image
    original_image = gr.State(value=Image.open(default_example[0]).convert('RGB'))

    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Tab("Point mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_p.render()

            with gr.Column(scale=1):
                segm_img_p.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    add_or_remove = gr.Radio(
                        ["Add Mask", "Remove Area"],
                        value="Add Mask",
                    )

                    with gr.Column():
                        segment_btn_p = gr.Button(
                            "Start segmenting!", variant="primary"
                        )
                        clear_btn_p = gr.Button("Restart", variant="secondary")

                gr.Markdown("Try some of the examples below ⬇️")

                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_p],
                    fn=lambda x: x,
                    outputs=[original_image],
                    # fn=segment_with_points,
                    # cache_examples=True,
                    examples_per_page=4,
                    run_on_click=True
                )

            with gr.Column():
                # Description
                gr.Markdown(description_p)

    cond_img_p.select(get_points_with_draw, [cond_img_p, add_or_remove], cond_img_p)
    cond_img_p.upload(lambda x: x, inputs=[cond_img_p], outputs=[original_image])

    # segment_btn_e.click(
    #     segment_everything,
    #     inputs=[
    #         cond_img_e,
    #         input_size_slider,
    #         mor_check,
    #         contour_check,
    #         retina_check,
    #     ],
    #     outputs=segm_img_e,
    # )

    segment_btn_p.click(
        segment_with_points, inputs=[cond_img_p, original_image], outputs=[segm_img_p, cond_img_p]
    )

    def clear():
        return None, None

    def clear_text():
        return None, None, None

    # clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])
    clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])

demo.queue()
demo.launch()
