import torch
import numpy as np
import yaml
import mcubes
import trimesh
import trimesh.transformations as tf
import os
import glob
import pandas as pd
from scipy.spatial import cKDTree
from src.model import FUNSR_Net
from tqdm import tqdm

# Load Config
CONFIG_PATH = "configs/config.yaml"
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(pred_mesh, gt_points, num_samples=30000, threshold=0.05):
    """
    threshold=0.05: Nới lỏng ngưỡng chấp nhận lỗi lên 5% (vì GT là điểm thưa)
    """
    if len(pred_mesh.vertices) == 0:
        return {"CD": 999.0, "HD": 999.0, "F-Score": 0.0}

    # 1. Sample điểm trên mesh dự đoán
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)

    # --- TỰ ĐỘNG CĂN TÂM (CENTERING) ---
    # Giúp loại bỏ sai số do lệch vị trí, chỉ so sánh hình dáng
    gt_center = np.mean(gt_points, axis=0)
    pred_center = np.mean(pred_points, axis=0)

    gt_centered = gt_points - gt_center
    pred_centered = pred_points - pred_center
    # -----------------------------------

    # 2. Xây dựng cây tìm kiếm (KDTree) trên dữ liệu đã căn tâm
    tree_gt = cKDTree(gt_centered)
    tree_pred = cKDTree(pred_centered)

    # 3. Tính khoảng cách
    dist_pred_to_gt, _ = tree_gt.query(pred_centered, k=1)
    dist_gt_to_pred, _ = tree_pred.query(gt_centered, k=1)

    # Chamfer Distance (L2)
    chamfer_l2 = np.mean(dist_pred_to_gt ** 2) + np.mean(dist_gt_to_pred ** 2)

    # Hausdorff Distance
    hd = max(np.max(dist_pred_to_gt), np.max(dist_gt_to_pred))

    # F-Score
    precision = np.mean(dist_pred_to_gt < threshold)
    recall = np.mean(dist_gt_to_pred < threshold)

    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0.0

    return {
        "CD": chamfer_l2,
        "HD": hd,
        "F-Score": f_score,
        "Precision": precision,
        "Recall": recall
    }


def reconstruct_and_evaluate(model_path, gt_path, save_mesh_path, save_gt_path):
    # 1. Load Model
    net = FUNSR_Net(hidden_dim=cfg['model']['hidden_dim']).to(DEVICE)
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint['net'] if (isinstance(checkpoint, dict) and 'net' in checkpoint) else checkpoint

        # Fix _orig_mod
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k
            new_state_dict[new_key] = v
        net.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return None
    net.eval()

    # 2. Inference
    N = 128
    x = np.linspace(-1.1, 1.1, N)
    grid = np.stack(np.meshgrid(x, x, x), -1).reshape(-1, 3)
    sdfs = []
    chunk = 50000

    with torch.no_grad():
        for i in range(0, len(grid), chunk):
            pts = torch.tensor(grid[i:i + chunk], dtype=torch.float32).to(DEVICE)
            val = net(pts)
            sdfs.append(val.cpu().numpy())

    sdfs = np.concatenate(sdfs).reshape(N, N, N)

    try:
        verts, faces = mcubes.marching_cubes(sdfs, 0.0)
        verts = (verts / (N - 1)) * 2.2 - 1.1
        mesh = trimesh.Trimesh(verts, faces)

        # --- 🛠️ CHỈNH SỬA ROTATION (Z=90) ---
        # X=0, Y=0, Z=90 độ
        matrix = tf.euler_matrix(0, 0, np.radians(90), 'sxyz')
        mesh.apply_transform(matrix)
        # ------------------------------------

        mesh.export(save_mesh_path)
        # print(f"   💾 Đã lưu Pred: {save_mesh_path}")
    except ValueError:
        return None

        # 3. Save GT
    gt_points = np.load(gt_path)
    trimesh.PointCloud(gt_points).export(save_gt_path)
    # print(f"   💾 Đã lưu GT: {save_gt_path}")
    # 4. Tính Metrics
    metrics = compute_metrics(mesh, gt_points, threshold=0.05)
    return metrics


def main():
    ckpt_dir = "./checkpoints"
    processed_dir = cfg['paths']['processed_data']
    data_dir = os.path.join(processed_dir, "pointclouds")
    res_dir = "./results"
    os.makedirs(res_dir, exist_ok=True)

    model_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    model_files = [f for f in model_files if "_ckpt" not in f]

    if not model_files: return

    records = []
    print(f"Đang đánh giá {len(model_files)} mẫu (Z-Rot 90, Centered)...")
    print(f"{'Nodule ID':<25} | {'CD(x1k)':<7} | {'HD':<6} | {'F-Score':<7}")
    print("-" * 60)

    for m_path in tqdm(model_files, desc="Evaluating"):
        base_name = os.path.basename(m_path).replace(".pth", "")
        gt_path = os.path.join(data_dir, base_name + ".npy")
        if not os.path.exists(gt_path): continue

        mesh_out = os.path.join(res_dir, base_name + "_pred.obj")
        gt_out = os.path.join(res_dir, base_name + "_gt.obj")

        m = reconstruct_and_evaluate(m_path, gt_path, mesh_out, gt_out)

        if m:
            tqdm.write(f"{base_name:<25} | {m['CD'] * 1000:<7.4f} | {m['HD']:<6.4f} | {m['F-Score']:<7.4f}")

            records.append({
                "Nodule_ID": base_name,
                "Chamfer_Distance": m['CD'],
                "Hausdorff_Distance": m['HD'],
                "F_Score": m['F-Score'],
                "Precision": m['Precision'],
                "Recall": m['Recall'],
                "Pred_Path": mesh_out
            })

    if records:
        df = pd.DataFrame(records)
        print("-" * 60)
        print(f"   REPORT (FINAL):")
        print(f"   Avg CD (x1000): {df['Chamfer_Distance'].mean() * 1000:.4f}")
        print(f"   Avg Hausdorff : {df['Hausdorff_Distance'].mean():.4f}")
        print(f"   Avg F-Score   : {df['F_Score'].mean():.4f}")

        csv_path = os.path.join(res_dir, "full_metrics_final.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n Đã lưu kết quả tại: {csv_path}")


if __name__ == "__main__":
    main()