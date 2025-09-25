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

                density_map, normalized_density = model(crops)
                # Gunakan density_map (bukan normalized) untuk counting yang akurat
                crop_pred = density_map

                # Model menghasilkan density map dengan resolusi ~8x lebih kecil dari input
                # crop_pred shape: [batch, 1, H/8, W/8] 
                # Tapi karena ada padding, kita harus crop sesuai ukuran asli
                
                # Hitung ukuran output yang sesuai dengan input asli (sebelum padding)
                output_h = h_crop // 8  # Downsampling factor ~8x
                output_w = w_crop // 8
                
                # Crop prediction ke ukuran yang sesuai dengan input asli (sebelum padding)
                if crop_pred.size(2) > output_h or crop_pred.size(3) > output_w:
                    crop_pred = crop_pred[:, :, :output_h, :output_w]

                # JANGAN resize! Model sudah menghasilkan density map yang benar
                # Density map resolution lebih rendah adalah normal untuk crowd counting
                # Resize akan merusak density values dan membuat count tidak akurat

                crop_preds.append(crop_pred)

        # Pastikan semua crop_pred memiliki dimensi yang konsisten
        for i, crop_pred in enumerate(crop_preds):
            if len(crop_pred.shape) == 3:
                # [batch*channels, H, W] -> [batch, channels, H, W]  
                crop_preds[i] = crop_pred.unsqueeze(1)
            elif len(crop_pred.shape) == 2:
                # [H, W] -> [1, 1, H, W]
                crop_preds[i] = crop_pred.unsqueeze(0).unsqueeze(0)
        
        print(f"📦 Processing {len(crop_preds)} crops for assembly...")
        
        # Sekarang concat seharusnya aman
        if len(crop_preds) > 0:
            crop_preds = torch.cat(crop_preds, dim=0)
            print(f"� Final crop_preds shape: {crop_preds.shape}")
        else:
            print("❌ No crops to process!")
            return 0

        # Density map memiliki resolusi lebih rendah (1/8 dari input)
        pred_map_h, pred_map_w = h // 8, w // 8
        pred_map = torch.zeros([b, 1, pred_map_h, pred_map_w]).to(device)
        crop_masks_lowres = torch.zeros([b, 1, pred_map_h, pred_map_w]).to(device)
        
        idx = 0
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                
                # Convert koordinat ke resolusi density map (1/8)
                gis_low, gie_low = gis // 8, gie // 8
                gjs_low, gje_low = gjs // 8, gje // 8
                
                # Pastikan tidak keluar batas
                gis_low = max(0, min(gis_low, pred_map_h))
                gie_low = max(0, min(gie_low, pred_map_h))
                gjs_low = max(0, min(gjs_low, pred_map_w))  
                gje_low = max(0, min(gje_low, pred_map_w))
                
                if gie_low > gis_low and gje_low > gjs_low:
                    # Ambil sesuai ukuran area yang valid
                    pred_h = gie_low - gis_low
                    pred_w = gje_low - gjs_low
                    
                    # Ambil crop yang sesuai dari tensor yang sudah di-concat
                    # Setelah concat, crop_preds berbentuk [total_crops, channels, H, W]
                    current_crop = crop_preds[idx]  # Shape: [channels, H, W] atau [1, H, W]
                    
                    # Pastikan kita punya 4D tensor [1, 1, H, W]
                    while len(current_crop.shape) < 4:
                        current_crop = current_crop.unsqueeze(0)
                    
                    # Sekarang slice dengan aman - ambil hanya 1 batch, 1 channel
                    pred_crop = current_crop[:1, :1, :pred_h, :pred_w]
                    
                    # Pastikan ukuran cocok dengan target region
                    if pred_crop.shape[2] == pred_h and pred_crop.shape[3] == pred_w:
                        pred_map[:, :, gis_low:gie_low, gjs_low:gje_low] += pred_crop
                        crop_masks_lowres[:, :, gis_low:gie_low, gjs_low:gje_low] += 1.0
                    else:
                        print(f"⚠️  Shape mismatch: pred_crop {pred_crop.shape} vs target ({pred_h}, {pred_w})")
                
                idx += 1

        # Normalisasi berdasarkan overlap
        mask = crop_masks_lowres
        mask[mask == 0] = 1.0  # Hindari pembagian dengan 0
        outputs = pred_map / mask

        count = outputs.sum().item()
        
        # Debug info
        print(f"📊 Debug info:")
        print(f"   - Density map size: {outputs.shape}")
        print(f"   - Density map range: [{outputs.min().item():.6f}, {outputs.max().item():.6f}]")
        print(f"   - Total count: {count:.2f}")
        
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
