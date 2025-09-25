import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import os
import csv
from Networks import ALTGVT

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_model(model_path, device="cuda"):
    model = ALTGVT.alt_gvt_large(pretrained=False)   # ubah: jangan pretrained=True
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # ubah: strict=False biar fleksibel
    model.to(device)
    model.eval()
    return model

def predict_image(img_path, model, device="cuda", crop_size=512):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    b, c, h, w = img.size()
    rh, rw = crop_size, crop_size
    crop_imgs, crop_masks = [], []
    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
            crop_imgs.append(img[:, :, gis:gie, gjs:gje])
            mask = torch.zeros([b, 1, h, w]).to(device)
            mask[:, :, gis:gie, gjs:gje].fill_(1.0)
            crop_masks.append(mask)

    crop_imgs, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks))

    crop_preds = []
    with torch.no_grad():
        for i in range(0, crop_imgs.size(0), 4):
            # --- ubah: handle output model ---
            out = model(crop_imgs[i:i+4])
            if isinstance(out, tuple):
                crop_pred = out[0]
            else:
                crop_pred = out

            _, _, h1, w1 = crop_pred.size()
            crop_pred = F.interpolate(crop_pred, size=(h1*8, w1*8),
                                      mode='bilinear', align_corners=True) / 64
            crop_preds.append(crop_pred)

    crop_preds = torch.cat(crop_preds, dim=0)

    pred_map = torch.zeros([b, 1, h, w]).to(device)
    idx = 0
    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
            pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
            idx += 1

    mask = crop_masks.sum(dim=0).unsqueeze(0)
    outputs = pred_map / mask

    count = outputs.sum().item()
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path model .pth")
    parser.add_argument("--input", type=str, required=True, help="Path gambar atau folder")
    parser.add_argument("--output-csv", type=str, default="predictions.csv", help="File CSV output")
    parser.add_argument("--crop-size", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model_path, device)

    # Kumpulkan gambar
    if os.path.isdir(args.input):
        img_files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        img_files = [args.input]

    results = []
    for img_path in img_files:
        count = predict_image(img_path, model, device, args.crop_size)
        img_id = os.path.basename(img_path)
        results.append({"image_id": img_id, "predicted_count": round(count, 2)})
        print(f"{img_id} -> {count:.2f} orang")

    # Simpan ke CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "predicted_count"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Predictions saved to {args.output_csv}")
