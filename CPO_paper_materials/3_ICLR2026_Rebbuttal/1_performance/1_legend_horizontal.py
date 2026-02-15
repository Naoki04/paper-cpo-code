
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 凡例情報
colours = {
    "SAPG (0.005)": "green",
    "SAPG (0)": "blue",

    
    
    
}
styles = {
    "SAPG (0.005)": (10, (5, 3, 1, 3, 1, 3)),
    "SAPG (0)": ":",
}

# Line2D オブジェクト作成
legend_elements = [
    Line2D([0, 1], [0, 0],
           color=colours[label],
           linestyle=styles[label],
           linewidth=2.5,
           label=label)
    for label in colours.keys()
]

# 軸なし縦長図
fig, ax = plt.subplots(figsize=(2.2, 1.6))  # 小さめ縦長
ax.axis("off")

# 凡例を上部中央にぴったり置く
legend = ax.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.0),  # x=中央, y=上端
    frameon=False,
    fontsize=11,
    handlelength=2.8,
    borderaxespad=0.0,  # ← この行が空白削減の鍵
)

# 保存（余白最小）
fig.savefig("legend_vertical_tight.svg", format="svg", bbox_inches="tight")


# Line2D オブジェクト作成
legend_elements = [
    Line2D([0, 1], [0, 0],
           color=colours[label],
           linestyle=styles[label],
           linewidth=2.5,
           label=label)
    for label in colours.keys()
]

# 横長で軸なし
fig, ax = plt.subplots(figsize=(2.0, 0.4))  # 横長サイズ
ax.axis("off")

# 横並びの凡例
legend = ax.legend(
    handles=legend_elements,
    loc='center',
    bbox_to_anchor=(0.5, 0.5),  # 図中央に置く
    frameon=False,
    fontsize=11,
    handlelength=2.8,
    borderaxespad=0.0,
    ncol=len(colours)  # ← 横並び
)

# 保存
fig.savefig("/home/mil/shitanda/CPO_paper_materials/3_ICLR2026_Rebbuttal/1_performance/output_plot/legend_horizontal_tight_.svg", format="svg", bbox_inches="tight")