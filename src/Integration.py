# ---------------------------------------------------------------
# STEP 2 — SEVERITY ASSESSMENT
# ---------------------------------------------------------------

def predict_severity(severity_img_array: np.ndarray,
                     disease: str,
                     confidence: float) -> str:
    if severity_model is not None:
        try:
            # ImageNet normalisation (matches training)
            mean_val = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std_val  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            img    = severity_img_array[0].astype(np.float32) / 255.0
            img    = (img - mean_val) / std_val

            # HWC -> CHW, add batch dim
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                # CoralOrdinal output: raw logits shape [1, 2]
                logits    = severity_model(tensor)[0].cpu()       # shape [2]
                coral_out = torch.sigmoid(logits).numpy()         # [2] probs
                p0        = float(coral_out[0])
                p1        = float(coral_out[1])

            # Standard CORAL 3-class conversion:
            # P(Mild)     = 1 - p0
            # P(Moderate) = p0 - p1
            # P(Severe)   = p1
            class_probs = np.array([
                1.0 - p0,
                p0 - p1,
                p1
            ], dtype=np.float32)
            class_probs = np.clip(class_probs, 0.0, 1.0)

            # Ordinal score -> threshold lookup
            score = float(class_probs @ np.arange(3, dtype=np.float32))
            t     = severity_thresholds          # [0.1705, 1.3933]
            if score < t[0]:
                pred_idx = 0
            elif score < t[1]:
                pred_idx = 1
            else:
                pred_idx = 2

            severity_map = ['mild', 'moderate', 'severe']
            result       = severity_map[pred_idx]
            print(f"   [SEVERITY] {result} | p0={p0:.3f} p1={p1:.3f} | "
                  f"score={score:.3f} | probs={class_probs.round(3)}")
            return result

        except Exception as e:
            print(f"Severity model error: {e}")
            import traceback
            traceback.print_exc()

    # Fallback based on disease-model confidence
    if confidence >= 0.85:   return "mild"
    elif confidence >= 0.70: return "moderate"
    else:                    return "severe"