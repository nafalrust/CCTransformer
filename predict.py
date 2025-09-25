import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
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
    """Load pretrained ALTGVT model"""
    model = ALTGVT.alt_gvt_large(pretrained=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # strict=False supaya fleksibel
    model.to(device)
    model.eval()
    return model

def predict_single_image(model, img_path, device="cuda", crop_size=512, batch_size=4):
    """Prediksi untuk 1 gambar (patch-based inference)"""
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
        for i in range(0, crop_imgs.size(0), batch_size):
            gs, gt = i, min(crop_imgs.size(0), i + batch_size)
            out = model(crop_imgs[gs:gt])
            crop_pred = out[0] if isinstance(out, tuple) else out

            _, _, h1, w1 = crop_pred.size()
            crop_pred = F.interpolate(
                crop_pred,
                size=(h1 * 8, w1 * 8),
                mode="bilinear",
                align_corners=True
            ) / 64
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

    return torch.sum(outputs).item()

def main():
    parser = argparse.ArgumentParser(description="Predict crowd counts with CCTrans (ALTGVT)")
    parser.add_argument("--model-path", type=str, required=True, help="Path ke model .pth")
    parser.add_argument("--input", type=str, required=True, help="Path gambar atau folder")
    parser.add_argument("--output-csv", type=str, default="predictions.csv", help="File CSV output")
    parser.add_argument("--crop-size", type=int, default=512, help="Ukuran crop (harus kelipatan ws)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size untuk patch inference")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    model = load_model(args.model_path, device)

    # Ambil daftar file gambar
    if os.path.isdir(args.input):
        img_files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    else:
        img_files = [args.input]

    results = []
    for img_path in img_files:
        count = predict_single_image(model, img_path, device, args.crop_size, args.batch_size)
        img_id = os.path.basename(img_path)
        results.append({"image_id": img_id, "predicted_count": round(count, 2)})
        print(f"{img_id} -> {count:.2f} orang")

    # Simpan ke CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "predicted_count"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Predictions saved to {args.output_csv}")
    print(f"Total test images: {len(results)}")
    print(f"Average predicted count: {np.mean([r['predicted_count'] for r in results]):.2f}")

if __name__ == "__main__":
    main()
