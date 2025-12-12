import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
import yaml

# Load config để biết đường dẫn
try:
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        DATA_DIR = os.path.join(cfg['paths']['processed_data'], "pointclouds")
except:
    # Fallback nếu chưa có config hoặc lỗi
    DATA_DIR = "./data/processed/pointclouds"


def visualize_point_cloud(npy_path):
    # Load dữ liệu
    if not os.path.exists(npy_path):
        print(f"Không tìm thấy file: {npy_path}")
        return

    points = np.load(npy_path)

    print(f"\n   Đang kiểm tra: {os.path.basename(npy_path)}")
    print(f"   - Shape: {points.shape} (Mong đợi: N x 3)")
    print(f"   - Min coords: {points.min(axis=0)}")
    print(f"   - Max coords: {points.max(axis=0)}")

    # Kiểm tra chuẩn hóa (Nên nằm trong khoảng -1 đến 1)
    if points.max() > 1.1 or points.min() < -1.1:
        print("CẢNH BÁO: Dữ liệu chưa được chuẩn hóa về [-1, 1]!")
    else:
        print("Dữ liệu đã chuẩn hóa tốt (nằm trong khối đơn vị).")

    # Vẽ 3D (Lấy mẫu 2000 điểm để vẽ cho nhẹ)
    sample_idx = np.random.choice(points.shape[0], min(2000, points.shape[0]), replace=False)
    p_sample = points[sample_idx]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    # Càng đậm màu tức là trục Z càng cao (để dễ nhìn độ sâu)
    img = ax.scatter(p_sample[:, 0], p_sample[:, 1], p_sample[:, 2],
                     c=p_sample[:, 2], cmap='viridis', s=2)

    ax.set_title(f"Preview: {os.path.basename(npy_path)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set giới hạn trục để không bị méo hình
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.colorbar(img, ax=ax, label='Z-axis')
    plt.show()


if __name__ == "__main__":
    # Tìm tất cả file .npy
    files = glob.glob(os.path.join(DATA_DIR, "*.npy"))

    if not files:
        print(f"Không tìm thấy dữ liệu trong {DATA_DIR}")
        print("Hãy chạy 'python preprocess.py' trước!")
    else:
        print(f"Tìm thấy {len(files)} nốt phổi.")

        while True:
            # Chọn ngẫu nhiên 1 file để xem
            target_file = random.choice(files)
            visualize_point_cloud(target_file)

            ans = input("Bạn có muốn xem nốt khác không? (y/n): ")
            if ans.lower() != 'y':
                break