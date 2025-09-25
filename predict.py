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
    model = ALTGVT.alt_gvt_large(pretrained=False)   # pretrained=False
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # strict=False biar fleksibel
    model.to(device)
    model.eval()
    return model

def predict_image(img_path, model, device="cuda", crop_size=512, batch_size=4, ws=8):
    try:
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        b, c, h, w = img.size()
        print(f"📏 Input image size: {h}x{w}")
        
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
        print(f"📦 Number of crops: {crop_imgs.size(0)}")

        crop_preds = []
        with torch.no_grad():
            for i in range(0, crop_imgs.size(0), batch_size):
                crops = crop_imgs[i:i+batch_size]

                # --- Fix padding: harus divisible by patch_size (4) DAN ws (8) ---
                _, _, h_crop, w_crop = crops.shape
                print(f"🔧 Processing crop {i//batch_size + 1}: {h_crop}x{w_crop}")
                
                # Model menggunakan patch_size=4, jadi ukuran harus kelipatan 4
                patch_size = 4
                # Setelah patch embedding, ukuran jadi h_crop//4 x w_crop//4  
                # Ini harus bisa dibagi ws untuk GroupAttention
                
                # Pastikan ukuran setelah patch embedding bisa dibagi ws
                target_h = ((h_crop + patch_size - 1) // patch_size) * patch_size
                target_w = ((w_crop + patch_size - 1) // patch_size) * patch_size
                
                # Setelah patch embedding: target_h//4 x target_w//4
                # Ini harus bisa dibagi ws
                patch_h, patch_w = target_h // patch_size, target_w // patch_size
                
                # Pastikan patch dimensions bisa dibagi ws
                if patch_h % ws != 0:
                    patch_h = ((patch_h + ws - 1) // ws) * ws
                if patch_w % ws != 0:
                    patch_w = ((patch_w + ws - 1) // ws) * ws
                    
                # Convert back to image dimensions
                target_h = patch_h * patch_size
                target_w = patch_w * patch_size
                
                # Apply padding
                pad_h = target_h - h_crop
                pad_w = target_w - w_crop
                
                if pad_h > 0 or pad_w > 0:
                    print(f"🔥 Padding from {h_crop}x{w_crop} to {target_h}x{target_w}")
                    crops = F.pad(crops, (0, pad_w, 0, pad_h), mode="constant", value=0)

                crop_pred, _ = model(crops)

                # Model output resolution adalah input_size / 8 (karena 4 stage dengan stride 2 each)
                # Tapi kita butuh crop ke ukuran asli sebelum padding
                output_h = h_crop // 8
                output_w = w_crop // 8
                
                # Crop prediction ke ukuran yang sesuai dengan input asli
                crop_pred = crop_pred[:, :, :output_h, :output_w]

                # Resize ke ukuran crop asli
                crop_pred = F.interpolate(
                    crop_pred, size=(h_crop, w_crop), mode='bilinear', align_corners=True
                )

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
        print(f"✅ Prediction complete. Count: {count:.2f}")
        return count
        
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path model .pth")
    parser.add_argument("--input", type=str, required=True, help="Path gambar atau folder")
    parser.add_argument("--output-csv", type=str, default="predictions.csv", help="File CSV output")
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ws", type=int, default=8, help="Window size ALTGVT (default=8)")
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
        print(f"\n🖼️  Processing: {os.path.basename(img_path)}")
        try:
            count = predict_image(img_path, model, device, args.crop_size, args.batch_size, args.ws)
            img_id = os.path.basename(img_path)
            results.append({"image_id": img_id, "predicted_count": round(count, 2)})
            print(f"✅ {img_id} -> {count:.2f} people")
        except Exception as e:
            print(f"❌ Failed to process {os.path.basename(img_path)}: {str(e)}")
            img_id = os.path.basename(img_path)
            results.append({"image_id": img_id, "predicted_count": 0})

    # Simpan ke CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "predicted_count"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Predictions saved to {args.output_csv}")
