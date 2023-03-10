import os
from pathlib import Path
from glob import glob
import shutil
import torch

import gradio as gr
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from webui import wrap_gradio_gpu_call
from modules import shared, scripts, script_callbacks, ui
from modules import generation_parameters_copypaste as parameters_copypaste
import launch

from laion import LaionAesthetic

script_dir = Path(scripts.basedir())
models_dir = script_dir / "models"
predictors = {}  # name: pipeline

model_list = [
    "sac+logos+ava1-l14-linearMSE",
    # "ava+logos-l14-reluMSE", # not working... 
    "ava+logos-l14-linearMSE"
]

classify_styles = ["Waifu Diffusion style", "Manual"]
wd_style_scores = [6.675, 6, 5, -1]
wd_style_score_dict = {
    6.675: "exceptional",
    6: "best aesthetic",
    5: "normal aesthetic",
    -1: "bad aesthetic"
}
wd_style_score_names = ["exceptional", "best aesthetic", "normal aesthetic", "bad aesthetic"]

wd_score_description = """
Note: The scores used in "Waifu Diffusion style" are from this [release note](https://cafeai.notion.site/WD-1-5-Beta-2-Release-Notes-2852db5a9cdd456ba52fc5730b91acfd):
| Score | Style |
| --- | --- |
| ≥ 6.675 | exceptional |
| ≥ 6 | best aesthetic |
| ≥ 5 | normal aesthetic |
| < 5 | bad aesthetic |
"""

def library_check():
    if not launch.is_installed("clip"):
        launch.run_pip("install git+https://github.com/openai/CLIP.git", "requirements for Laion Aesthetic")

def model_check(name):
    if name not in predictors:
        library_check()

        if name in model_list:
            predictors[name] = LaionAesthetic(model_name=f"{name}.pth", model_path=models_dir)

def unload_models():
    for name in predictors:
        del predictors[name]
    torch.cuda.empty_cache()

def predict_score(image, model_name):
    model_check(model_name)
    score = predictors[model_name].get_score(image)

    return score

def predict_score_with_wd_style(image, model_name, step = None):
    score = predict_score(image, model_name) # normal style
    wd_score = calc_output_folder(score, classify_styles[0], step) # wd style

    return [score, wd_score]

def example_score_folders(style, step = None):
    if style == classify_styles[0]: # wd style
        return wd_style_score_names
    elif style == classify_styles[1]: # manual
        if step is None:
            step = 0.5
        return [str(n) for n in [7, 7-step, 7-step*2, 7-step*3]]

def calc_output_folder(score, style, step = None):
    if style == classify_styles[0]: # wd style
        for s in wd_style_scores:
            if score >= s:
                return wd_style_score_dict[s]
    elif style == classify_styles[1]: # manual
        max = 10
        if step is None:
            step = 0.5
        for s in list(range(max, -1, -step)):
            if score >= s:
                return f"{s}"

def output_dir_previews_update(dir, classify_style, step = None):
    if dir == "":
        return ["", ""]
    folders = example_score_folders(classify_style, step)
    output_dir_previews = "\n".join([f"- {Path(dir)/f}" for f in folders])

    if classify_style == classify_styles[0]:
        return [f"Output dirs will be created like: \n{output_dir_previews}", wd_score_description] 
    else:
        output_dir_previews += f"\n- {Path(dir)/'etc'}"
        return [f"Output dirs will be created like: \n{output_dir_previews}", ""] 


def copy_or_move_files(img_path: Path, to: Path, copy, together):
    img_name = img_path.stem  # hoge.jpg
    if together:
        for p in img_path.parent.glob(f"{img_name}.*"):
            if copy:
                shutil.copy2(p, to / p.name)
            else:
                if os.path.exists(p):
                    p.rename(to / p.name)
                else:
                    print(f"Not found: {p}".encode("utf-8"))
    else:
        if copy:
            shutil.copy2(img_path, to / img_path.name)
        else:
            img_path.rename(to / img_path.name)

def batch_classify(
    input_dir, output_dir, model_name, classify_type, output_style, together, step
):
    print("Batch classifying started")
    try:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        image_paths = [
            p
            for p in input_dir.iterdir()
            if (p.is_file and p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"])
        ]

        print(f"Found {len(image_paths)} images")

        for i, f in enumerate(image_paths):
            if f.is_dir():
                continue

            img = Image.open(f)
            score = predict_score(img, model_name)

            folder_name = calc_output_folder(score, classify_type, step)
            if folder_name is None:
                continue

            item_output_dir = output_dir / folder_name

            if not os.path.exists(item_output_dir):
                os.makedirs(item_output_dir)

            copy_or_move_files(
                f, item_output_dir, output_style == "Copy", together
            )

            copied_or_moved = "copied" if output_style == "Copy" else "moved"

            print(
                f"The aesthetic score for {f.name} is predicted to be {score} and {copied_or_moved} to {item_output_dir}"
            )

        print("All done!")
        return "Done!"
    except Exception as e:
        return f"Error: {e}"
    

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui:
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem(label="Single"):
                    with gr.Row().style(equal_height=False):
                        with gr.Column():
                            # with gr.Tabs():
                            image = gr.Image(
                                source="upload",
                                label="Image",
                                interactive=True,
                                type="pil",
                            )

                            single_model_select = gr.Dropdown(
                                label="Model",
                                choices=model_list,
                                value=model_list[0],
                                interactive=True,
                            )

                            single_start_btn = gr.Button(
                                value="Predict", variant="primary"
                            )

                        with gr.Column():
                            single_aesthetic_score_result = gr.Label(label="Aesthetic Score")
                            single_wd_aesthetic_score_result = gr.Label(label="WD Aesthetic Score")

                            gr.Markdown(wd_score_description)

                with gr.TabItem(label="Batch"):

                    with gr.Row().style(equal_height=False):
                        with gr.Column():
                            input_dir_input = gr.Textbox(
                                label="Image Directory",
                                placeholder="path/to/classify",
                                type="text",
                            )
                            output_dir_input = gr.Textbox(
                                label="Output Directory",
                                placeholder="path/to/output",
                                type="text",
                            )

                            gr.Markdown("")

                            batch_model_select = gr.Dropdown(
                                label="Model",
                                choices=model_list,
                                value=model_list[0],
                                interactive=True,
                            )

                            classify_type_radio = gr.Radio(
                                label="Classify type",
                                choices=classify_styles,
                                value="Manual",
                                interactive=True,
                            )

                            manual_step_slider = gr.Slider(
                                label="Step (Manual classification only)",
                                minimum=0,
                                maximum=10,
                                step=0.01,
                                value=0.5,
                                interactive=True,
                            )

                            gr.Markdown("")

                            output_style_radio = gr.Radio(
                                label="Output style",
                                choices=["Copy", "Move"],
                                value="Copy",
                                interactive=True,
                            )
                            copy_or_move_captions_together = gr.Checkbox(
                                label="Copy or move captions together",
                                value=True,
                                interactive=True,
                            )

                            gr.Markdown("")

                            batch_start_btn = gr.Button(
                                value="Start", variant="primary"
                            )

                        with gr.Column():
                            status_block = gr.Label(label="Status", value="Idle")

                            output_dir_previews_md = gr.Markdown("")

                            wd_score_description_md = gr.Markdown("")

        image.change(
            fn=predict_score_with_wd_style,
            inputs=[image, single_model_select],
            outputs=[single_aesthetic_score_result, single_wd_aesthetic_score_result],
        )
        single_start_btn.click(
            fn=predict_score_with_wd_style,
            inputs=[image, single_model_select],
            outputs=[single_aesthetic_score_result, single_wd_aesthetic_score_result],
        )

        output_dir_input.change(
            fn=output_dir_previews_update,
            inputs=[output_dir_input, classify_type_radio],
            outputs=[output_dir_previews_md, wd_score_description_md],
        )
        classify_type_radio.change(
            fn=output_dir_previews_update,
            inputs=[output_dir_input, classify_type_radio],
            outputs=[output_dir_previews_md, wd_score_description_md],
        )

        batch_start_btn.click(
            fn=batch_classify,
            inputs=[
                input_dir_input,
                output_dir_input,
                batch_model_select,
                classify_type_radio,
                output_style_radio,
                copy_or_move_captions_together,
                manual_step_slider,
            ],
            outputs=[status_block],
        )

    return [(ui, "LAION Aesthetic", "laion_aesthetic")]


script_callbacks.on_ui_tabs(on_ui_tabs)