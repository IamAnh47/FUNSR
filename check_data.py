import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
import yaml
import sys

# Load config để biết đường dẫn
try:
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        DATA_DIR = os.path.join(cfg['paths']['processed_data'], "pointclouds")
except:
    DATA_DIR = "./data/processed/pointclouds"


def visualize_point_cloud(npy_path):
    # Load dữ liệu
    if not os.path.exists(npy_path):
        print(f"❌ Không tìm thấy file: {npy_path}")
        print(f"   (Đang tìm trong: {DATA_DIR})")
        return

    points = np.load(npy_path)

    print(f"\n🔍 Đang kiểm tra: {os.path.basename(npy_path)}")
    print(f"   - Shape: {points.shape}")
    print(f"   - Min coords: {points.min(axis=0)}")
    print(f"   - Max coords: {points.max(axis=0)}")

    if points.max() > 1.1 or points.min() < -1.1:
        print("⚠️ CẢNH BÁO: Dữ liệu chưa được chuẩn hóa về [-1, 1]!")
    else:
        print("✅ Dữ liệu chuẩn hóa tốt.")

    # Vẽ 3D
    sample_idx = np.random.choice(points.shape[0], min(2000, points.shape[0]), replace=False)
    p_sample = points[sample_idx]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    img = ax.scatter(p_sample[:, 0], p_sample[:, 1], p_sample[:, 2],
                     c=p_sample[:, 2], cmap='viridis', s=2)

    ax.set_title(f"Preview: {os.path.basename(npy_path)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Cố định khung nhìn [-1, 1] để không bị méo tỉ lệ
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])

    plt.colorbar(img, ax=ax, label='Z-axis')
    plt.show()


if __name__ == "__main__":

    # sys.argv[0] là tên script, sys.argv[1] là tham số đầu tiên
    if len(sys.argv) > 1:
        filename = sys.argv[1]

        if os.path.exists(filename):
            visualize_point_cloud(filename)

        else:
            full_path = os.path.join(DATA_DIR, filename)
            visualize_point_cloud(full_path)

    else:
        files = glob.glob(os.path.join(DATA_DIR, "*.npy"))

        if not files:
            print(f"Không tìm thấy dữ liệu trong {DATA_DIR}")
        else:
            print(f"Tìm thấy {len(files)} nốt phổi.")
            while True:
                target_file = random.choice(files)
                visualize_point_cloud(target_file)

                ans = input("Bạn có muốn xem nốt khác không? (y/n): ")
                if ans.lower() != 'y':
                    break