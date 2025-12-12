import os
import yaml
import numpy as np
from tqdm import tqdm
import json
import random
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pylidc.utils import consensus


import trimesh
import mcubes
# -----------------------------------

# Import loader
from src.dicom_loader import DicomLoader


def process_single_patient(args):
    pid, cfg = args

    # Setup paths
    raw_dir = cfg['paths']['raw_data']
    processed_dir = cfg['paths']['processed_data']

    # Tạo đường dẫn lưu pointcloud và mesh
    pc_dir = os.path.join(processed_dir, "pointclouds")
    mesh_dir = os.path.join(processed_dir, "meshes")  # <--- Thêm thư mục Mesh

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
            # Bỏ qua nếu ít đồng thuận
            if len(cluster) < 3: continue

            try:
                # 1. Tạo Consensus Mask (Gộp ý kiến bác sĩ)
                # clevel=0.5 nghĩa là ít nhất 50% bác sĩ đồng ý đó là nốt
                # padding quan trọng để marching cubes không bị hở biên
                mask, cbbox, _ = consensus(cluster, clevel=cfg['data']['consensus_level'], pad=cfg['data']['padding'])

                # --- PHẦN SỬA ĐỔI QUAN TRỌNG ---

                # 2. Tạo Mesh từ Mask bằng Marching Cubes
                # (Thay vì dùng np.where lấy voxel)
                try:
                    # Thuật toán Marching Cubes
                    verts, faces = mcubes.marching_cubes(mask, 0.5)

                    # Tạo đối tượng Trimesh
                    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

                    # Kiểm tra lưới rác
                    if len(mesh.vertices) < 100: continue

                except Exception:
                    continue

                # 3. Chuẩn hóa Mesh về [-1, 1] (Normalization)
                # Việc chuẩn hóa Mesh sẽ tự động chuẩn hóa các điểm lấy mẫu sau này

                # Centering (Dời về gốc tọa độ)
                centroid = mesh.centroid
                mesh.vertices -= centroid

                # Scaling (Co về cầu đơn vị)
                # Tìm đỉnh xa nhất
                max_dist = np.max(np.linalg.norm(mesh.vertices, axis=1))
                if max_dist == 0: continue
                # Scale sao cho nằm gọn trong [-1, 1] (chia cho max_dist * 1.05 để có chút lề)
                mesh.vertices /= (max_dist * 1.0)

                # 4. Lưu Ground Truth Mesh (.obj)
                # Để sau này dùng file này so sánh visual với kết quả predict
                file_id = f"{pid}_nodule{i}"
                mesh.export(os.path.join(mesh_dir, f"{file_id}_gt.obj"))

                # 5. Lấy mẫu điểm (Sampling) từ Mesh để Train
                # Dùng sample_surface của trimesh xịn hơn FPS thủ công
                points, _ = trimesh.sample.sample_surface(mesh, cfg['data']['num_points'])

                # Đảm bảo kiểu dữ liệu float32 cho PyTorch
                points = points.astype(np.float32)

                # 6. Lưu Point Cloud (.npy)
                np.save(os.path.join(pc_dir, f"{file_id}.npy"), points)

                valid_nodules += 1

            except Exception as e:
                # print(f"Lỗi nốt {i}: {e}") # Uncomment để debug nếu cần
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
    MESH_DIR = os.path.join(PROCESSED_DIR, "meshes")  # <--- Tạo thêm folder meshes

    os.makedirs(PC_DIR, exist_ok=True)
    os.makedirs(MESH_DIR, exist_ok=True)

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
        print(f"Bắt đầu xử lý {len(target_patients)} bệnh nhân (Tạo cả .npy và _gt.obj)...")

        # Chạy song song
        max_workers = min(os.cpu_count(), 8)
        tasks = [(pid, cfg) for pid in target_patients]

        stats = {"success": 0, "nodules": 0}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_patient, t) for t in tasks]

            for future in tqdm(as_completed(futures), total=len(tasks)):
                res = future.result()

                if res['success']:
                    stats['success'] += 1
                    stats['nodules'] += res['nodules']

                # Luôn ghi nhận là đã xử lý
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
            patient_map[pid].append(fname)

        pids = list(patient_map.keys())
        random.seed(42)
        random.shuffle(pids)

        # Tỉ lệ: 80% Train - 10% Val - 10% Test
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