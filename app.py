import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- 1. MODEL ARXITEKTURASI ---
def make_cnn(bn=True):
    c1 = nn.Conv2d(1, 16, 3, padding=1)
    c2 = nn.Conv2d(16, 32, 3, padding=1)
    if bn:
        return nn.Sequential(
            c1,
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            c2,
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128, 10),
        )
    return nn.Sequential(
        c1,
        nn.ReLU(),
        nn.MaxPool2d(2),
        c2,
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128, 10),
    )

# --- 2. MODELNI XOTIRAGA YUKLASH ---
device = torch.device("cpu")  # VPS da CPU da ishlash uchun
model = make_cnn(bn=True)

model_path = os.path.join(os.path.dirname(__file__), "digits_cnn.pt")
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# --- 3. CHIZILGAN RASMNI QAYTA ISHLASH VA TAXMIN QILISH ---
def predict(sketch_data):
    if sketch_data is None:
        return {}

    # Gradio Sketchpad natijasidan kompozit rasmni olish
    if isinstance(sketch_data, dict):
        img_data = sketch_data.get("composite", sketch_data.get("image"))
    else:
        img_data = sketch_data

    if img_data is None:
        return {}

    # PIL Image formatiga o'tkazish
    img = Image.fromarray(img_data.astype("uint8")).convert("RGBA")

    # Alpha kanali orqali chiziqni ajratib olish
    alpha = img.split()[-1]
    if alpha.getextrema() == (0, 0):
        img = img.convert("L")
    else:
        img = alpha

    # Oq chiziq, qora fon holatiga keltirish
    img_np = np.array(img)
    if np.mean(img_np) > 127:
        img = ImageOps.invert(img)

    # 8x8 pikselga qisqartirish
    img = img.resize((8, 8), Image.Resampling.BILINEAR)

    # Normalizatsiya [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # (1, 1, 8, 8) o'lchamli tenzor
    tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Natijani hisoblash
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)[0]

    return {str(i): float(probs[i]) for i in range(10)}

# --- 4. VEB INTERFEYSNI SOZLASH (GRADIO) ---
interface = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(
        brush=gr.Brush(colors=["#FFFFFF"], default_size=20),
        canvas_size=(280, 280),
        type="numpy",
        label="Raqamni shu yerga chizing (Oq rangda)",
    ),
    outputs=gr.Label(num_top_classes=3, label="Model taxmini"),
    title="Raqamlarni Tanuvchi Mini CNN",
    description="Sichqoncha yoki sensor orqali 0 dan 9 gacha raqam chizing.",
    live=True,
)

# --- 5. ISHGA TUSHIRISH ---
if __name__ == "__main__":
    # VPS va Docker ichidan ulanish imkoni bo'lishi uchun 0.0.0.0 shart
    interface.launch(server_name="0.0.0.0", server_port=7860)