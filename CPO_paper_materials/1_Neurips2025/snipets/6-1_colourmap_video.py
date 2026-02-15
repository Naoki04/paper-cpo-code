import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as np
import numpy as np
import os
import imageio
import shutil
from tqdm import tqdm
import io
from PIL import Image

def generate_kl_evolution_video(
        csv_path: str,
        horizon_length: int,
        interval: int = 400,
        max_envsteps: int = 20 * 10**9,
        save_path: str = "kl_evolution.mp4",
        fps: int = 30,
    ):
        # ラベル
        labels = ["L", "F1", "F2", "F3", "F4", "F5"]
        

        # データ読み込み
        df = pd.read_csv(csv_path)
        num_envs = 24576  # 固定でOKならハードコード。引数にしても良い。

        def epoch_to_envstep(epoch):
            return epoch * num_envs * horizon_length
        
        images = []

        # フレーム画像を作成
        for epoch in tqdm(range(0, len(df), interval)):
            row = df.iloc[epoch]
            tensor = torch.tensor(row.values, dtype=torch.float32).view(6, 6)
            tensor = torch.flip(tensor.flatten(), dims=[0]).view(6, 6)
            data = tensor.numpy()

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(data, cmap='viridis', vmin=0, vmax=3)
            ax.set_xticks(np.arange(6))
            ax.set_yticks(np.arange(6))
            ax.set_xticklabels(labels, fontsize=18)
            ax.set_yticklabels(labels, fontsize=18)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.grid(False)
            ax.tick_params(which='major', length=0)

            for i in range(6):
                row_min_val = float('inf')
                row_min_pos = None
                for j in range(6):
                    val = data[i, j]
                    ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white', fontsize=18)
                    if i != 0 and i != j and val < row_min_val:
                        row_min_val = val
                        row_min_pos = (i, j)
                if row_min_pos:
                    circle = plt.Circle((row_min_pos[1], row_min_pos[0]), 0.4, color='white', fill=False, linewidth=2)
                    ax.add_patch(circle)

            envsteps = epoch_to_envstep(epoch)
            ax.set_title(f"{envsteps/1e9:.1f} / {max_envsteps/1e9:.1f}G Env Steps", fontsize=18)
            

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            image = np.array(img)[..., :3]
            
            # 幅が奇数なら、右に1列追加して偶数幅に
            h, w, _ = image.shape
            if w % 2 != 0:
                image = np.pad(image, ((0, 0), (0, 1), (0, 0)), mode='edge')  # 右に1列複製で追加
                
            images.append(image)
            buf.close()
            plt.close()
            #if envsteps >= max_envsteps:
            #    break

        

        imageio.mimsave(save_path, images, fps=fps, codec='libx264')
        print(f"✅ 動画保存完了: {save_path}")

        


def main():
    data_dir = "/home/mil/shitanda/CPO_NeurIPS2025/data/0_KL/kari"
    csv_files = os.listdir(data_dir)
    csv_files = [f for f in csv_files if f.endswith('.csv')]
    csv_files = sorted(csv_files)
    
    print(csv_files)
    
    
    for csv_file in csv_files:
        csv_path=os.path.join(data_dir, csv_file),
        csv_path = csv_path[0]
        
        if "Hand" in csv_file:
            horizon_length = 8
            interval = 200
        else:
            horizon_length = 16
            interval = 100
        
        print(f"Processing {csv_file}...")
        out_dir = os.path.join(data_dir, "../", "KL_video")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{csv_file[:-4]}.mp4")
        
        generate_kl_evolution_video(
            csv_path=csv_path,
            horizon_length=horizon_length,
            interval=interval,
            max_envsteps=20 * 10**9,
            save_path=save_path,
            fps=60,
        )
    


if __name__ == "__main__":
    main()