import torch
import numpy as np
import yaml
import mcubes
import trimesh
import trimesh.transformations as tf
import os
import glob
import shutil
import pandas as pd
from scipy.spatial import cKDTree
from src.model import FUNSR_Net
from tqdm import tqdm
import gc

# Load Config
CONFIG_PATH = "configs/config.yaml"
if not os.path.exists(CONFIG_PATH):
    CONFIG_PATH = "config.yaml"

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_volumetric_metrics(pred_mesh, gt_mesh, n_samples=50000, batch_size=10000):
    # DSC/IoU với kỹ thuật Batching
    bounds_pred = pred_mesh.bounds
    bounds_gt = gt_mesh.bounds
    min_xyz = np.minimum(bounds_pred[0], bounds_gt[0])
    max_xyz = np.maximum(bounds_pred[1], bounds_gt[1])
    padding = (max_xyz - min_xyz) * 0.1
    min_xyz -= padding
    max_xyz += padding

    intersection_count = 0
    pred_vol_count = 0
    gt_vol_count = 0

    steps = int(np.ceil(n_samples / batch_size))

    for _ in range(steps):
        current_batch_size = min(batch_size, n_samples)
        samples = np.random.uniform(min_xyz, max_xyz, (current_batch_size, 3))
        try:
            occ_pred = pred_mesh.contains(samples)
            occ_gt = gt_mesh.contains(samples)
        except Exception:
            continue

        intersect = np.logical_and(occ_pred, occ_gt).sum()
        intersection_count += intersect
        pred_vol_count += occ_pred.sum()
        gt_vol_count += occ_gt.sum()

        del samples, occ_pred, occ_gt

    union_count = pred_vol_count + gt_vol_count - intersection_count
    sum_volumes = pred_vol_count + gt_vol_count

    iou = intersection_count / union_count if union_count > 0 else 0.0
    dice = (2.0 * intersection_count) / sum_volumes if sum_volumes > 0 else 0.0
    return dice, iou


def compute_surface_metrics(pred_mesh, gt_points, threshold=0.05):
    """Tính CD, HD, F-Score"""
    if len(pred_mesh.vertices) == 0:
        return {"CD": 999.0, "HD": 999.0, "F-Score": 0.0}

    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, 20000)

    gt_center = np.mean(gt_points, axis=0)
    pred_center = np.mean(pred_points, axis=0)
    gt_centered = gt_points - gt_center
    pred_centered = pred_points - pred_center

    tree_gt = cKDTree(gt_centered)
    tree_pred = cKDTree(pred_centered)

    dist_pred_to_gt, _ = tree_gt.query(pred_centered, k=1)
    dist_gt_to_pred, _ = tree_pred.query(gt_centered, k=1)

    chamfer_l2 = np.mean(dist_pred_to_gt ** 2) + np.mean(dist_gt_to_pred ** 2)
    hd = max(np.max(dist_pred_to_gt), np.max(dist_gt_to_pred))

    precision = np.mean(dist_pred_to_gt < threshold)
    recall = np.mean(dist_gt_to_pred < threshold)
    f_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Dọn dẹp bộ nhớ cây kdtree
    del tree_gt, tree_pred, pred_points
    return {"CD": chamfer_l2, "HD": hd, "F-Score": f_score}


def reconstruct_and_evaluate(net, model_path, gt_path, gt_mesh_source_path, save_mesh_path, save_gt_path):
    torch.cuda.empty_cache()
    gc.collect()

    # 1. Load Weights
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        state_dict = checkpoint['net'] if (isinstance(checkpoint, dict) and 'net' in checkpoint) else checkpoint
        new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        net.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Lỗi: {e}")
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
            sdfs.append(net(pts).cpu().numpy())

    sdfs = np.concatenate(sdfs).reshape(N, N, N)

    try:
        verts, faces = mcubes.marching_cubes(sdfs, 0.0)
        verts = (verts / (N - 1)) * 2.2 - 1.1
        mesh = trimesh.Trimesh(verts, faces)
        matrix = tf.euler_matrix(0, np.radians(180), np.radians(90), 'sxyz')
        mesh.apply_transform(matrix)
        mesh.export(save_mesh_path)
    except ValueError:
        return None

    del sdfs, grid, x
    gc.collect()

    # 3. Load GT
    gt_points = np.load(gt_path)
    if os.path.exists(gt_mesh_source_path):
        gt_mesh = trimesh.load(gt_mesh_source_path)
        shutil.copy(gt_mesh_source_path, save_gt_path)
    else:
        gt_mesh = None
        trimesh.PointCloud(gt_points).export(save_gt_path)

    # 4. Tính toán Metrics
    surf_m = compute_surface_metrics(mesh, gt_points, threshold=0.05)

    vol_m = {"Dice": 0.0, "IoU": 0.0}
    # Đủ ram thì tắt comment để tính =)))
    # if gt_mesh is not None:
    #     try:
    #         dice, iou = compute_volumetric_metrics(mesh, gt_mesh, n_samples=50000, batch_size=10000)
    #         vol_m["Dice"] = dice
    #         vol_m["IoU"] = iou
    #     except Exception:
    #         pass

    # Cleanup trimesh cache
    mesh._cache.clear()
    if gt_mesh: gt_mesh._cache.clear()
    del mesh, gt_mesh
    gc.collect()

    return {**surf_m, **vol_m}


def main():
    # --- SETUP ---
    ckpt_dir = "./checkpoints"
    processed_dir = cfg['paths']['processed_data']
    data_dir = os.path.join(processed_dir, "pointclouds")
    mesh_source_dir = os.path.join(processed_dir, "meshes")

    if os.path.exists("/content/drive/MyDrive"):
        res_dir = "/content/drive/MyDrive/funsr/results"
    else:
        res_dir = "./results"
    os.makedirs(res_dir, exist_ok=True)

    csv_path = os.path.join(res_dir, "metrics_full.csv")

    model_files = glob.glob(os.path.join(ckpt_dir, "*.pth"))
    model_files = [f for f in model_files if "_ckpt" not in f]
    if not model_files: return

    # --- KHỞI TẠO FILE CSV ---
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # Tạo danh sách cột
    columns = ["Nodule_ID", "Chamfer_Distance", "Hausdorff_Distance", "F_Score", "Dice", "IoU", "Pred_Path"]
    # Tạo file rỗng có header
    pd.DataFrame(columns=columns).to_csv(csv_path, index=False)
    print(f"Đã tạo file kết quả: {csv_path}")

    # --- INIT MODEL ---
    print("Đang khởi tạo model...")
    net = FUNSR_Net(hidden_dim=cfg['model']['hidden_dim']).to(DEVICE)

    print(f"Đang đánh giá {len(model_files)} mẫu (Streaming Mode)...")
    print(f"{'Nodule ID':<22} | {'CD':<6} | {'HD':<6} | {'F-Sc':<6} | {'Dice':<6} | {'IoU':<6}")
    print("-" * 70)

    # --- LOOP ---
    for m_path in tqdm(model_files, desc="Processing"):
        base_name = os.path.basename(m_path).replace(".pth", "")
        gt_path = os.path.join(data_dir, base_name + ".npy")
        gt_mesh_source = os.path.join(mesh_source_dir, base_name + "_gt.obj")

        if not os.path.exists(gt_path): continue

        mesh_out = os.path.join(res_dir, base_name + "_pred.obj")
        gt_out = os.path.join(res_dir, base_name + "_gt.obj")

        m = reconstruct_and_evaluate(net, m_path, gt_path, gt_mesh_source, mesh_out, gt_out)

        if m:
            tqdm.write(
                f"{base_name:<22} | {m['CD'] * 1000:<6.2f} | {m['HD']:<6.3f} | {m['F-Score']:<6.3f} | {m['Dice']:<6.3f} | {m['IoU']:<6.3f}")

            # Tạo dictionary cho dòng hiện tại
            row = {
                "Nodule_ID": base_name,
                "Chamfer_Distance": m['CD'],
                "Hausdorff_Distance": m['HD'],
                "F_Score": m['F-Score'],
                "Dice": m['Dice'],
                "IoU": m['IoU'],
                "Pred_Path": mesh_out
            }

            # --- GHI NGAY VÀO FILE (Append Mode) ---
            # mode='a': Ghi nối tiếp vào đuôi file
            # header=False: Không ghi lại tiêu đề
            pd.DataFrame([row]).to_csv(csv_path, mode='a', header=False, index=False)

            # Xóa biến row để chắc ăn
            del row

    # --- FINAL REPORT (Đọc lại từ file) ---
    if os.path.exists(csv_path):
        print("-" * 70)
        print("Đang tổng hợp báo cáo từ file CSV...")
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                print(f"   Avg CD (x1k): {df['Chamfer_Distance'].mean() * 1000:.4f}")
                print(f"   Avg HD      : {df['Hausdorff_Distance'].mean():.4f}")
                print(f"   Avg F-Score : {df['F_Score'].mean():.4f}")
                print(f"   Avg Dice    : {df['Dice'].mean():.4f}")
                print(f"   Avg IoU     : {df['IoU'].mean():.4f}")
            else:
                print("File CSV rỗng.")
        except Exception as e:
            print(f"Không thể đọc file CSV để tổng hợp: {e}")

    print(f"\nHoàn tất! Kết quả đã được lưu an toàn tại: {csv_path}")


if __name__ == "__main__":
    print(f"Đang sử dụng: {DEVICE}")
    main()