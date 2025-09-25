#!/usr/bin/env python3
"""
Quick fix untuk IndexError pada predict.py
Membuat versi yang lebih robust untuk handling dimensi tensor
"""

def fix_tensor_indexing_issue():
    """
    Perbaikan untuk masalah IndexError: too many indices for tensor of dimension 3
    """
    
    fix_code = '''
    # Perbaikan untuk bagian crop assembly
    
    # 1. Pastikan crop_preds memiliki dimensi yang konsisten
    for i, crop_pred in enumerate(crop_preds):
        if len(crop_pred.shape) == 3:
            # [batch*channels, H, W] -> [batch, channels, H, W]
            crop_preds[i] = crop_pred.unsqueeze(1)
        elif len(crop_pred.shape) == 2:
            # [H, W] -> [1, 1, H, W]
            crop_preds[i] = crop_pred.unsqueeze(0).unsqueeze(0)
    
    # 2. Concat dengan aman
    crop_preds = torch.cat(crop_preds, dim=0)
    
    # 3. Index dengan pengecekan dimensi
    current_crop = crop_preds[idx]
    
    # Ensure 4D tensor [N, C, H, W]
    while len(current_crop.shape) < 4:
        current_crop = current_crop.unsqueeze(0)
    
    # Slice dengan aman
    pred_crop = current_crop[:1, :1, :pred_h, :pred_w]  # Ambil [1, 1, H, W]
    '''
    
    print("🔧 Fix code untuk IndexError:")
    print(fix_code)

if __name__ == "__main__":
    fix_tensor_indexing_issue()