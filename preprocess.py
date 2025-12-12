import os
import yaml
import numpy as np
from tqdm import tqdm
import json
import random
import pylidc as pl
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pylidc.utils import consensus

# Import loader của chúng ta
from src.dicom_loader import DicomLoader


def farthest_point_sampling(points, n_samples):
    """FPS Sampling tối ưu cho numpy"""
    if len(points) <= n_samples: return points
    # Dùng random choice cho nhanh (FPS chuẩn rất chậm trên CPU)
    # Nếu muốn strict FPS, cần dùng thư viện C++ binding
    indices = np.random.choice(len(points), n_samples, replace=False)
    return points[indices]


def process_single_patient(args):
    pid, cfg = args

    # Setup paths
    raw_dir = cfg['paths']['raw_data']
    pc_dir = os.path.join(cfg['paths']['processed_data'], "pointclouds")

    loader = DicomLoader(raw_dir)

    res = {"pid": pid, "success": False, "nodules": 0, "error": None}

    try:
        # Load Volume
        vol, spacing, nodules = loader.load_patient_data(pid)
        if vol is None:
            res["error"] = "Load Failed"
            return res

        valid_nodules = 0
        for i, cluster in enumerate(nodules):
            # Bỏ qua nếu ít đồng thuận hoặc quá phức tạp (>4 annotations thường là nhiễu)
            if len(cluster) < 3: continue

            try:
                # 1. Tạo Consensus Mask (Gộp ý kiến bác sĩ)
                # clevel=0.5 nghĩa là ít nhất 50% bác sĩ đồng ý đó là nốt
                mask, cbbox, _ = consensus(cluster, clevel=cfg['data']['consensus_level'], pad=cfg['data']['padding'])

                # 2. Kiểm tra kích thước
                # Chỉ lấy các điểm thuộc mask (Volumetric Point Cloud)
                z, y, x = np.where(mask > 0.5)
                if len(z) < 500: continue  # Quá nhỏ

                points = np.stack([x, y, z], axis=1).astype(np.float32)

                # 3. Chuẩn hóa về [-1, 1] (CỰC KỲ QUAN TRỌNG CHO FUNSR)
                # Centering
                centroid = np.mean(points, axis=0)
                points -= centroid
                # Scale về cầu đơn vị
                max_dist = np.max(np.linalg.norm(points, axis=1))
                if max_dist == 0: continue
                points /= max_dist

                # 4. Downsample
                points = farthest_point_sampling(points, cfg['data']['num_points'])

                # 5. Lưu file .npy
                file_id = f"{pid}_nodule{i}"
                np.save(os.path.join(pc_dir, f"{file_id}.npy"), points)

                valid_nodules += 1

            except Exception as e:
                continue

        if valid_nodules > 0:
            res["success"] = True
            res["nodules"] = valid_nodules
        else:
            res["error"] = "No valid nodules"

    except Exception as e:
        res["error"] = str(e)

    return res


def main():
    # Load config
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    PROCESSED_DIR = cfg['paths']['processed_data']
    PC_DIR = os.path.join(PROCESSED_DIR, "pointclouds")
    os.makedirs(PC_DIR, exist_ok=True)

    # File log để resume
    LOG_FILE = os.path.join(PROCESSED_DIR, "processed_log.json")
    completed_patients = set()
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                completed_patients = set(json.load(f))
            print(f"Resume: Đã xử lý {len(completed_patients)} bệnh nhân trước đó.")
        except:
            pass

    # Quét danh sách bệnh nhân
    loader = DicomLoader(cfg['paths']['raw_data'])
    all_patients = loader.get_all_patient_ids()
    target_patients = [p for p in all_patients if p not in completed_patients]

    if not target_patients:
        print("Đã xử lý hết dữ liệu!")
    else:
        print(f"Bắt đầu xử lý {len(target_patients)} bệnh nhân...")

        # Chạy song song
        max_workers = min(os.cpu_count(), 8)  # Đừng dùng hết 100% CPU kẻo treo máy
        tasks = [(pid, cfg) for pid in target_patients]

        stats = {"success": 0, "nodules": 0}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_patient, t) for t in tasks]

            for future in tqdm(as_completed(futures), total=len(tasks)):
                res = future.result()

                if res['success']:
                    stats['success'] += 1
                    stats['nodules'] += res['nodules']

                # Luôn ghi nhận là đã xử lý (dù lỗi hay không) để ko chạy lại
                completed_patients.add(res['pid'])

                # Save log định kỳ
                if len(completed_patients) % 50 == 0:
                    with open(LOG_FILE, "w") as f: json.dump(sorted(list(completed_patients)), f)

        # Final save
        with open(LOG_FILE, "w") as f:
            json.dump(sorted(list(completed_patients)), f)
        print(f"\nHoàn tất: {stats['success']} bệnh nhân thành công | {stats['nodules']} nốt phổi.")

    # --- TẠO FILE SPLIT (Train/Val/Test) ---
    print("Đang chia tập dữ liệu (Split)...")
    all_files = glob.glob(os.path.join(PC_DIR, "*.npy"))

    if all_files:
        # Group theo bệnh nhân để tránh data leakage
        patient_map = {}
        for fpath in all_files:
            fname = os.path.basename(fpath)
            pid = fname.split("_")[0]
            if pid not in patient_map: patient_map[pid] = []
            patient_map[pid].append(fname)  # Chỉ lưu tên file, không lưu full path cho gọn

        pids = list(patient_map.keys())
        random.seed(42)
        random.shuffle(pids)

        # Tỉ lệ: 80% Train - 10% Val - 10% Test (Evaluation)
        n = len(pids)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        train_pids = pids[:n_train]
        val_pids = pids[n_train:n_train + n_val]
        test_pids = pids[n_train + n_val:]

        def flatten(pid_list):
            files = []
            for pid in pid_list: files.extend(patient_map[pid])
            return files

        split_dict = {
            "train": flatten(train_pids),
            "val": flatten(val_pids),
            "test": flatten(test_pids)
        }

        split_path = os.path.join(PROCESSED_DIR, "split_data.json")
        with open(split_path, "w") as f:
            json.dump(split_dict, f, indent=4)

        print(f"Đã lưu split tại {split_path}")
        print(f"   Train: {len(split_dict['train'])} | Val: {len(split_dict['val'])} | Test: {len(split_dict['test'])}")
    else:
        print("Không có file dữ liệu nào để split.")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()