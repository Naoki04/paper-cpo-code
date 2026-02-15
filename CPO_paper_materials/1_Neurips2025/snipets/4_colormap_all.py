import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# フォントを大きめに
rcParams.update({'font.size': 16})

labels = [" L ", "F1", "F2", "F3", "F4", "F5"]

def generate_kl_snapshot(data_dir: str, target_step: float):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)

        # Handが入っていれば8, それ以外は16
        horizon_length = 8 if 'Hand' in csv_file else 16
        num_envs = 24576
        target_epoch = int(target_step / (horizon_length * num_envs))

        # CSV読み込み
        df = pd.read_csv(csv_path)
        if target_epoch >= len(df):
            print(f"⚠️ {csv_file}: target epoch {target_epoch} exceeds data length ({len(df)}). Skipping.")
            continue

        row = df.iloc[target_epoch]
        tensor = torch.tensor(row.values, dtype=torch.float32).view(6, 6)
        tensor = torch.flip(tensor.flatten(), dims=[0]).view(6, 6)
        data = tensor.numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(data, cmap='viridis', vmin=0, vmax=3)

        ax.set_xticks(np.arange(6))
        ax.set_yticks(np.arange(6))
        ax.set_xticklabels(labels, fontsize=20)
        ax.set_yticklabels(labels, fontsize=20)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.grid(False)
        ax.tick_params(which='major', length=0)

        # セルに値を表示 & 最小値に○
        for i in range(6):
            row_min_val = float('inf')
            row_min_pos = None
            for j in range(6):
                val = data[i, j]
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='white')

                if i != 0 and i != j and val < row_min_val:
                    row_min_val = val
                    row_min_pos = (i, j)
            if row_min_pos:
                circle = plt.Circle((row_min_pos[1], row_min_pos[0]), 0.4, color='white', fill=False, linewidth=1.0)
                ax.add_patch(circle)

        # タイトル
        #ax.set_title(f"{target_step / 1e9:.1f}G Env Steps")

        # 出力ファイル名
        step_label = f"{int(target_step/1e9)}G"
        output_name = f"{os.path.splitext(csv_file)[0]}_{step_label}.svg"
        os.makedirs("/home/mil/shitanda/CPO_NeurIPS2025/output/kl", exist_ok=True)
        output_path = os.path.join("/home/mil/shitanda/CPO_NeurIPS2025/output/kl", output_name)
        plt.tight_layout()
        plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"✅ Saved snapshot: {output_path}")


data_dir = "/home/mil/shitanda/CPO_NeurIPS2025/data/0_KL/ablation"
generate_kl_snapshot(data_dir=data_dir, target_step=0e9)
generate_kl_snapshot(data_dir=data_dir, target_step=5e9)
generate_kl_snapshot(data_dir=data_dir, target_step=10e9)
generate_kl_snapshot(data_dir=data_dir, target_step=15e9)
#generate_kl_snapshot(data_dir=data_dir, target_step=20e9)

