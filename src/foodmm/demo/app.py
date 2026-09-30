"""Gradio UI. The callbacks are plain functions (tested without a server)."""
from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

NO_EXAMPLES = {"info": "Chưa có ví dụ: chạy scripts/build_demo_examples.py để tạo examples.json"}


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

    with gr.Blocks(title="Nhận diện món ăn đa phương thức") as app:
        gr.Markdown("# Nhận diện món ăn từ ảnh và text\nCLIP ViT-B/16 đóng băng + các head của Mốc 2. "
                    "Text được làm sạch và che tên món trước khi đưa vào model.")
        with gr.Tab("Dự đoán"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type="pil", label="Ảnh món ăn")
                    text = gr.Textbox(label="Text (tùy chọn): công thức, mô tả...", lines=4)
                    use_vlm = gr.Checkbox(label="Trích xuất thông tin bằng VLM (lần đầu mất vài phút)",
                                          value=False, visible=vlm is not None)
                    button = gr.Button("Dự đoán", variant="primary")
                with gr.Column():
                    with gr.Row():
                        l_img = gr.Label(num_top_classes=5, label="Chỉ ảnh")
                        l_txt = gr.Label(num_top_classes=5, label="Chỉ text")
                        l_fus = gr.Label(num_top_classes=5, label="Fusion")
                    masked = gr.Textbox(label="Text sau khi che tên món", interactive=False)
                    vlm_json = gr.JSON(label="Kết quả VLM")
            button.click(ui_predict, [image, text, use_vlm], [l_img, l_txt, l_fus, masked, vlm_json])
        with gr.Tab("Ví dụ có sẵn"):
            if examples:
                choice = gr.Dropdown(choices=[e["name"] for e in examples], label="Chọn ví dụ")
                with gr.Row():
                    ex_img = gr.Image(type="pil", label="Ảnh", interactive=False)
                    ex_text = gr.Textbox(label="Text", lines=6, interactive=False)
                with gr.Row():
                    e_img = gr.Label(num_top_classes=5, label="Chỉ ảnh")
                    e_txt = gr.Label(num_top_classes=5, label="Chỉ text")
                    e_fus = gr.Label(num_top_classes=5, label="Fusion")
                e_masked = gr.Textbox(label="Text sau khi che tên món", interactive=False)
                e_vlm = gr.JSON(label="Kết quả VLM (đã lưu)")
                choice.change(on_example, choice, [ex_img, ex_text, e_img, e_txt, e_fus, e_masked, e_vlm])
            else:
                gr.Markdown(NO_EXAMPLES["info"])
    return app
