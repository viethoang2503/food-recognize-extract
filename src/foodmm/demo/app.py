"""Gradio UI. The callbacks are plain functions (tested without a server)."""
from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

NO_EXAMPLES = {"info": "No examples yet: run scripts/build_demo_examples.py to create examples.json"}


def make_handlers(predictor, vlm, examples: Sequence[dict] | None):
    by_name = {e["name"]: e for e in (examples or [])}

    def on_predict(image, text, use_vlm):
        try:
            res = predictor.predict(image, text)
        except ValueError as e:
            return None, None, None, "", {"error": str(e)}
        vlm_out = vlm.extract(image, res["masked_text"]) if (use_vlm and vlm is not None) else None
        return res["image"], res["text"], res["fusion"], res["masked_text"], vlm_out

    def on_example(name):
        ex = by_name.get(name)
        if ex is None:
            return None, "", None, None, None, "", NO_EXAMPLES
        image = Image.open(ex["image_path"]).convert("RGB") if ex.get("image_path") else None
        p = ex["pred"]
        return image, ex["text"], p["image"], p["text"], p["fusion"], p["masked_text"], ex.get("vlm")

    return on_predict, on_example


def build_app(predictor, vlm=None, examples: Sequence[dict] | None = None):
    import gradio as gr

    on_predict, on_example = make_handlers(predictor, vlm, examples)

    def ui_predict(image, text, use_vlm):
        out = on_predict(image, text, use_vlm)
        if out[2] is None and isinstance(out[4], dict) and "error" in out[4]:
            gr.Warning(out[4]["error"])
        return out

    with gr.Blocks(title="Multimodal Food Recognition") as app:
        gr.Markdown("# Food recognition from images and text\nFrozen CLIP ViT-B/16 with the Milestone 2 heads. "
                    "The text is cleaned and every dish name is masked before it reaches the models.")
        with gr.Tab("Predict"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil", label="Food photo")
                    text = gr.Textbox(label="Text (optional): recipe, description...", lines=4)
                    use_vlm = gr.Checkbox(label="Extract information with the VLM (the first run takes a few minutes)",
                                          value=False, visible=vlm is not None)
                    button = gr.Button("Predict", variant="primary")
                with gr.Column():
                    with gr.Row():
                        l_img = gr.Label(num_top_classes=5, label="Image only")
                        l_txt = gr.Label(num_top_classes=5, label="Text only")
                        l_fus = gr.Label(num_top_classes=5, label="Fusion")
                    masked = gr.Textbox(label="Text after masking dish names", interactive=False)
                    vlm_json = gr.JSON(label="VLM output")
            button.click(ui_predict, [image, text, use_vlm], [l_img, l_txt, l_fus, masked, vlm_json])
        with gr.Tab("Examples"):
            if examples:
                choice = gr.Dropdown(choices=[e["name"] for e in examples], label="Choose an example")
                with gr.Row():
                    ex_img = gr.Image(type="pil", label="Photo", interactive=False)
                    ex_text = gr.Textbox(label="Text", lines=6, interactive=False)
                with gr.Row():
                    e_img = gr.Label(num_top_classes=5, label="Image only")
                    e_txt = gr.Label(num_top_classes=5, label="Text only")
                    e_fus = gr.Label(num_top_classes=5, label="Fusion")
                e_masked = gr.Textbox(label="Text after masking dish names", interactive=False)
                e_vlm = gr.JSON(label="VLM output (saved)")
                choice.change(on_example, choice, [ex_img, ex_text, e_img, e_txt, e_fus, e_masked, e_vlm])
            else:
                gr.Markdown(NO_EXAMPLES["info"])
    return app
