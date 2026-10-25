import os
import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. MODEL ARXITEKTURASI ---
def make_cnn(bn=True):
    c1 = nn.Conv2d(1, 16, 3, padding=1)
    c2 = nn.Conv2d(16, 32, 3, padding=1)
    if bn:
        return nn.Sequential(
            c1, nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            c2, nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(128, 10),
        )
    return nn.Sequential(
        c1, nn.ReLU(), nn.MaxPool2d(2),
        c2, nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(128, 10),
    )

# --- 2. MODELNI YUKLASH ---
device = torch.device("cpu")
model = make_cnn(bn=True)
model_path = os.path.join(os.path.dirname(__file__), "digits_cnn.pt")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# --- 3. TAXMIN QILISH ---
def predict(sketch_data):
    if sketch_data is None:
        return {}

    if isinstance(sketch_data, dict):
        img_data = sketch_data.get("composite", sketch_data.get("image"))
    else:
        img_data = sketch_data

    if img_data is None:
        return {}

    img = Image.fromarray(img_data.astype("uint8")).convert("RGBA")
    alpha = img.split()[-1]
    if alpha.getextrema() == (0, 0):
        img = img.convert("L")
    else:
        img = alpha

    img_np = np.array(img)
    if np.mean(img_np) > 127:
        img = ImageOps.invert(img)

    img = img.resize((8, 8), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)[0]

    return {str(i): float(probs[i]) for i in range(10)}

# --- 4. GRADIO INTERFEYS ---
interface = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(
        brush=gr.Brush(colors=["#FFFFFF"], default_size=20),
        canvas_size=(280, 280),
        type="numpy",
        label="Raqamni chizing (Oq rangda)",
    ),
    outputs=gr.Label(num_top_classes=3, label="Model taxmini"),
    title="Raqamlarni Tanuvchi Mini CNN",
    description="0 dan 9 gacha raqam chizing.",
    live=True,
)

if __name__ == "__main__":
    # Render taqdim etgan PORT orqali ishga tushirish
    port = int(os.environ.get("PORT", 7860))
    interface.launch(server_name="0.0.0.0", server_port=port)
