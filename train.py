import torch
import torch.optim as optim
import yaml
import os
import argparse
import time
from tqdm.auto import tqdm
import torch.multiprocessing as mp

from src.model import FUNSR_Net, Discriminator
from src.dataset import FUNSRDataset


def train_worker(args):
    # Nhận thêm biến 'force_retrain'
    points, nodule_id, cfg, gpu_id, force_retrain = args

    # 1. Setup
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    # torch.set_float32_matmul_precision('medium') # Bật nếu dùng GPU 30xx/40xx

    save_path = os.path.join("./checkpoints", f"{nodule_id}.pth")

    # --- LOGIC KIỂM TRA CHECKPOINT ---
    if os.path.exists(save_path):
        if not force_retrain:
            return None  # Bỏ qua âm thầm
        # Nếu force_retrain=True thì cứ chạy tiếp (sẽ ghi đè)

    # 2. Init
    try:
        raw_net = FUNSR_Net(hidden_dim=cfg['model']['hidden_dim']).to(device)
        net = torch.compile(raw_net)
    except:
        net = raw_net

    disc = Discriminator().to(device)

    opt_G = optim.Adam(net.parameters(), lr=cfg['train']['lr'])
    opt_D = optim.Adam(disc.parameters(), lr=cfg['train']['lr'])

    points = points.to(device, non_blocking=True)
    num_points = points.size(0)

    iterations = cfg['train']['iterations']
    batch_size = cfg['train']['batch_size']

    # --- BUFFER CONFIG ---
    buffer_steps = 100
    buffer_size = batch_size * buffer_steps
    n_queries = cfg['train']['queries_per_point']
    sigma = cfg['train']['query_sigma']

    # --- BIẾN ĐỂ TÍNH TỐC ĐỘ (IT/S) ---
    log_interval = 100
    start_train_time = time.time()
    last_log_time = start_train_time
    last_log_step = 0

    # Vòng lặp chính
    for i in range(0, iterations, buffer_steps):
        # 1. Sinh Buffer
        rand_idx = torch.randint(0, num_points, (buffer_size,), device=device)
        p_buffer = points[rand_idx]

        p_target_all = torch.repeat_interleave(p_buffer, n_queries, dim=0)
        noise = torch.randn_like(p_target_all) * sigma
        q_batch_all = p_target_all + noise

        current_chunk_size = batch_size * n_queries

        for step in range(buffer_steps):
            curr_iter = i + step
            if curr_iter >= iterations: break

            start = step * current_chunk_size
            end = start + current_chunk_size

            q_batch = q_batch_all[start:end].requires_grad_(True)
            p_target = p_target_all[start:end]

            # --- TRAINING ---
            sdf, grad = net.get_sdf_and_gradient(q_batch)
            grad_norm = torch.norm(grad, dim=1, keepdim=True) + 1e-6
            q_pulled = q_batch - (grad / grad_norm) * sdf

            l_self = torch.mean((q_pulled - p_target) ** 2)

            move_vec = p_target - q_batch
            move_norm = torch.norm(move_vec, dim=1, keepdim=True) + 1e-6
            dot = torch.sum((grad / grad_norm) * (move_vec / move_norm), dim=1)
            l_scc = torch.mean(1 - dot)

            opt_D.zero_grad()
            d_real = disc(torch.zeros_like(sdf))
            d_fake = disc(sdf.detach())
            l_d = 0.5 * torch.mean((d_real - 0.9) ** 2) + 0.5 * torch.mean(d_fake ** 2)
            l_d.backward()
            opt_D.step()

            opt_G.zero_grad()
            d_fake_g = disc(sdf)
            l_gan = 0.5 * torch.mean((d_fake_g - 0.9) ** 2)

            loss_G = (cfg['train']['lambda_self'] * l_self +
                      cfg['train']['lambda_scc'] * l_scc +
                      cfg['train']['lambda_gan'] * l_gan)

            loss_G.backward()
            opt_G.step()

            # --- LOGGING ---
            if curr_iter % log_interval == 0 and curr_iter > 0:
                current_time = time.time()
                delta_time = current_time - last_log_time
                delta_steps = curr_iter - last_log_step
                speed = delta_steps / delta_time if delta_time > 0 else 0.0

                last_log_time = current_time
                last_log_step = curr_iter

                tqdm.write(f"🔹 {nodule_id}: {curr_iter}/{iterations} | Loss: {loss_G.item():.4f} | ⚡ {speed:.1f} it/s")

            # Auto-save checkpoint mỗi 1000 bước
            if curr_iter % 1000 == 0 and curr_iter > 0:
                torch.save(net.state_dict(), save_path)

    torch.save(net.state_dict(), save_path)
    total_time = time.time() - start_train_time
    return f"DONE: {nodule_id} ({total_time:.1f}s)"


def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='val', help='val hoặc test')
    parser.add_argument('--workers', type=int, default=2, help='Số luồng song song')
    # Thêm cờ --all (mặc định là False, nếu gõ --all thì thành True)
    parser.add_argument('--all', action='store_true', help='Nếu có, sẽ train lại TẤT CẢ kể cả file đã xong')
    args = parser.parse_args()

    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg['paths']['processed_data']
    pc_dir = os.path.join(processed_dir, "pointclouds")
    split_file = os.path.join(processed_dir, "split_data.json")
    os.makedirs("./checkpoints", exist_ok=True)

    dataset = FUNSRDataset(pc_dir, split_file, mode=args.mode)

    print(f"Optimized Training: {len(dataset)} mẫu | {args.workers} workers")
    if args.all:
        print("CHẾ ĐỘ FORCE RETRAIN: Sẽ train lại toàn bộ từ đầu!")
    else:
        print("CHẾ ĐỘ SMART RESUME: Sẽ bỏ qua các file đã hoàn thành.")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tasks = []
    for i in range(len(dataset)):
        points, fname = dataset[i]
        nodule_id = fname.replace(".npy", "")
        # Truyền thêm args.all vào tasks
        tasks.append((points, nodule_id, cfg, 0, args.all))

    with mp.Pool(processes=args.workers) as pool:
        pbar_total = tqdm(total=len(tasks), desc="Total Progress", unit="nodule")
        for result in pool.imap_unordered(train_worker, tasks):
            if result: tqdm.write(result)
            pbar_total.update(1)
        pbar_total.close()

    print("\nĐã hoàn thành toàn bộ!")


if __name__ == "__main__":
    main()