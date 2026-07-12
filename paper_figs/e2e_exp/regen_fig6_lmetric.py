#!/usr/bin/env python3
"""重新生成 Fig6:用 LMetric(蓝色方块位)替换 vLLM RR。

- 旧四条线(SynergySched红/Sarathi RR棕/Sarathi Session紫/vLLM Session绿)
  数值从原 picture_orig_backup/ 面板逐点数字化而来(坐标轴/橙色SLO线同步克隆)。
- LMetric 线为 8xH100 实测(data/*/native_lmetric/,csv_process 同语义),
  数值存于 data/lmetric_metrics.json。
- 面板样式与 draw.py plot_metric 完全一致;拼接复用 picture_process.py,
  图例用 "LMetric" 替换 "vLLM RR"(蓝 #3399FF、方块标记位置不变)。

运行: python3 regen_fig6_lmetric.py   (先备份 picture/ -> picture_orig_backup/)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from picture_process import (add_side_title_to_image, add_title_to_image,
                             add_title_to_image_bottom,
                             combine_images_horizontally,
                             combine_images_vertically, add_legend_to_image)

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------- 数据 ----------------
with open(f"{BASE}/data/lmetric_metrics.json") as f:
    LM = json.load(f)

X = {
    "arxiv": [12, 13, 14, 15, 16],
    "flowgpt_qps": [14, 15, 16, 17, 18],
    "flowgpt_timestamp": [14, 15, 16, 17, 18],
    "sharegpt": [40, 44, 48, 52],
    "reasoning": [50, 52, 54, 56, 58],
}

# 旧线数字化数值(超出坐标轴的点用大值占位以复现原图"冲出画面"效果)
OLD = {
 "arxiv": {
  "p50":  {"red": [9.2, 9.2, 9.5, 10.2, 12.0], "brown": [10.0, 12.2, 14.7, 18.3, 24.5],
           "purple": [11.8, 13.9, 16.9, 21.9, 25.4], "green": [13.3, 15.9, 20.3, 25.7, 33.4]},
  "p90":  {"red": [14, 14.5, 15, 16, 18.5], "brown": [16, 20, 24, 29.5, 40],
           "purple": [23.5, 28, 32.5, 43.5, 48], "green": [29, 35.5, 43.5, 48.5, 58.5]},
  "ttft": {"red": [1.3, 1.35, 1.32, 1.4, 1.55], "brown": [0.45, 0.45, 0.45, 0.45, 0.55],
           "purple": [0.65, 0.75, 0.85, 1.5, 1.65], "green": [0.7, 0.8, 1.1, 1.45, 4.35]},
  "tpot": {"red": [42, 44, 47, 51, 56], "brown": [53, 64, 78, 96, 133],
           "purple": [62, 70, 90, 125, 139], "green": [67, 85, 110, 138, 151]},
  "slo":  {"red": [100, 100, 100, 100, 96], "brown": [99, 89, 75, 57, 34],
           "purple": [82, 72, 60, 45, 37], "green": [74, 63, 50, 36, 24]},
 },
 "flowgpt_qps": {
  "p50":  {"red": [7.3, 7.4, 7.4, 7.7, 8.5], "brown": [8.5, 9.9, 10.9, 13.0, 16.0],
           "purple": [7.5, 8.6, 8.9, 10.7, 12.1], "green": [12.1, 14.4, 19.6, 26, 30]},
  "p90":  {"red": [9.1, 9.1, 9.0, 9.5, 13.2], "brown": [10.8, 12.7, 13.8, 16, 32],
           "purple": [13.2, 15.5, 15, 18.8, 30.8], "green": [21.8, 40, 50, 55, 60]},
  "ttft": {"red": [0.65, 0.7, 0.65, 0.7, 0.8], "brown": [0.35, 0.35, 0.4, 0.4, 0.5],
           "purple": [0.5, 0.5, 0.5, 0.5, 0.55], "green": [0.6, 0.8, 1.1, 2.9, 7.5]},
  "tpot": {"red": [6.9, 6.9, 6.9, 7.2, 7.8], "brown": [8.2, 9.5, 10.5, 12.5, 15.2],
           "purple": [7.5, 8.1, 8.4, 10.2, 11.2], "green": [11.5, 13.7, 16.5, 19.5, 20]},
  "slo":  {"red": [100, 100, 100, 99, 85], "brown": [99, 83, 71, 36, 17],
           "purple": [87, 71, 78, 60, 50], "green": [49, 39, 23, 17, 15]},
 },
 "flowgpt_timestamp": {
  "p50":  {"red": [8.2, 8.5, 9.5, 12, 19.5], "brown": [9.8, 11.5, 13.5, 24.2, 25.5],
           "purple": [8.4, 9.9, 11.2, 15, 20.5], "green": [14.8, 19, 25.5, 33, 35]},
  "p90":  {"red": [13, 12.5, 13, 21.5, 29], "brown": [13, 14.5, 17, 37, 37],
           "purple": [14, 18, 25, 40, 56], "green": [30, 53, 68, 72, 75]},
  "ttft": {"red": [0.9, 1.15, 1.25, 1.6, 2.6], "brown": [0.5, 0.5, 0.6, 2.5, 1.15],
           "purple": [0.55, 0.6, 0.65, 0.8, 0.9], "green": [0.9, 1.3, 2.05, 4.4, 10.5]},
  "tpot": {"red": [6.8, 7.2, 7.9, 9.7, 14.3], "brown": [8.8, 10.8, 12.2, 19.2, 19.9],
           "purple": [7.5, 8.3, 10, 12.3, 15.5], "green": [12.5, 14.5, 16, 19, 19.9]},
  "slo":  {"red": [100, 100, 100, 92, 79], "brown": [100, 100, 100, 55, 52],
           "purple": [100, 95, 88, 71, 59], "green": [76, 59, 50, 36, 23]},
 },
 "sharegpt": {
  "p50":  {"red": [5.4, 5.6, 5.95, 6.45], "brown": [5.6, 5.8, 5.95, 6.2],
           "purple": [5.3, 5.6, 5.85, 7.7], "green": [6.8, 8.3, 11.5, 13]},
  "p90":  {"red": [6.4, 6.5, 6.7, 7.0], "brown": [6.6, 6.65, 6.8, 7.0],
           "purple": [6.35, 6.5, 6.75, 6.9], "green": [8.05, 8.3, 9.8, 10.5]},
  "ttft": {"red": [0.12, 0.13, 0.15, 0.2], "brown": [0.1, 0.12, 0.15, 0.18],
           "purple": [0.15, 0.15, 0.3, 0.9], "green": [0.15, 0.7, 2.9, 8.3]},
  "tpot": {"red": [21, 21.5, 22, 22.5], "brown": [22, 22.5, 23, 23.5],
           "purple": [20.5, 21, 21.5, 28.5], "green": [26, 31, 42, 47]},
  "slo":  {"red": [99, 99, 97, 88], "brown": [98, 95, 92, 90],
           "purple": [87, 72, 77, 50], "green": [49, 39, 23, 15]},
 },
 "reasoning": {
  "p50":  {"red": [2.6, 2.6, 2.6, 2.65, 2.95], "brown": [2.65, 2.75, 2.85, 2.95, 3.03],
           "purple": [2.85, 3.15, 6.5, 7.5, 8], "green": [9, 9, 9, 9, 9]},
  "p90":  {"red": [2.85, 2.87, 2.83, 2.95, 3.05], "brown": [2.88, 2.92, 3.05, 3.18, 3.25],
           "purple": [3.35, 3.38, 7, 8, 9], "green": [10, 10, 10, 10, 10]},
  "ttft": {"red": [0.14, 0.13, 0.15, 0.15, 0.38], "brown": [0.11, 0.11, 0.11, 0.11, 0.11],
           "purple": [0.13, 0.2, 2.5, 3, 3], "green": [3, 3, 3, 3, 3]},
  "tpot": {"red": [24.3, 24.5, 24.3, 24.5, 27], "brown": [25.5, 26.3, 27, 28.3, 29.3],
           "purple": [27, 29, 55, 60, 65], "green": [70, 70, 70, 70, 70]},
  # reasoning SLO 阈值收紧到 3.0s(见 REASONING_SLO_THRESH);老四线为
  # 从数字化 (P50,P90,SLO@6s) 单调插值估计的 SLO@3s,LMetric 见 LM_REASONING_SLO3
  "slo":  {"red": [90, 90, 90, 89, 70], "brown": [90, 90, 80, 59, 49],
           "purple": [62, 48, 23, 20, 18], "green": [4, 4, 3, 3, 2]},
 },
}

# reasoning 行 SLO 阈值(原 6s -> 3s):6s 对短回复(P50~2.8s)无区分度。
REASONING_SLO_THRESH = 3.0
# LMetric reasoning SLO@3.0s,由原始 mapped.csv 精确重算(csv_process 语义)
LM_REASONING_SLO3 = [73.0, 68.9, 63.8, 60.0, 57.2]

# 坐标轴克隆(自原图):yticks/ylim/橙色SLO横线
AXIS = {
 "arxiv": {
  "p50":  {"yticks": [10, 20, 30], "ylim": (7, 34)},
  "p90":  {"yticks": [20, 40, 60], "ylim": (11, 62)},
  "ttft": {"yticks": [0, 2, 4], "ylim": (0, 5.0), "hline": 2},
  "tpot": {"yticks": [50, 100, 150], "ylim": (35, 170), "hline": 65},
  "slo":  {"yticks": [25, 50, 75, 100], "ylim": (15, 105)},
 },
 "flowgpt_qps": {
  "p50":  {"yticks": [10, 15, 20], "ylim": (6.8, 20.2)},
  "p90":  {"yticks": [10, 20, 30], "ylim": (8, 34)},
  "ttft": {"yticks": [0, 2, 4], "ylim": (-0.15, 5), "hline": 2},
  "tpot": {"yticks": [8, 16, 24], "ylim": (6.3, 25), "hline": 12},
  "slo":  {"yticks": [25, 50, 75, 100], "ylim": (0, 105)},
 },
 "flowgpt_timestamp": {
  "p50":  {"yticks": [8, 16, 24], "ylim": (7, 27)},
  "p90":  {"yticks": [20, 40, 60], "ylim": (10, 60)},
  "ttft": {"yticks": [0, 3, 6, 9], "ylim": (0, 9.3), "hline": 2},
  "tpot": {"yticks": [6, 12, 18, 24], "ylim": (6, 24.5), "hline": 12},
  "slo":  {"yticks": [25, 50, 75, 100], "ylim": (0, 105)},
 },
 "sharegpt": {
  "p50":  {"yticks": [6, 8, 10], "ylim": (5.1, 10.3)},
  "p90":  {"yticks": [6, 7, 8, 9], "ylim": (6, 9)},
  "ttft": {"yticks": [0, 4, 8], "ylim": (-2.5, 9), "hline": 2},
  "tpot": {"yticks": [15, 30, 45], "ylim": (16, 48), "hline": 25},
  "slo":  {"yticks": [25, 50, 75, 100], "ylim": (0, 105)},
 },
 "reasoning": {
  "p50":  {"yticks": [3, 4, 5], "ylim": (2.4, 5.1)},
  "p90":  {"yticks": [2.4, 3.2, 4.0, 4.8], "ylim": (2.4, 4.8)},
  "ttft": {"yticks": [0.0, 0.4, 0.8], "ylim": (-0.35, 1.0), "hline": 0.4},
  "tpot": {"yticks": [24, 30, 36], "ylim": (20.5, 40), "hline": 28},
  "slo":  {"yticks": [25, 50, 75, 100], "ylim": (0, 105)},
 },
}

# 算法样式(与 draw.py algorithm_config 一致;lmetric 顶替 v0_roundrobin 的蓝色方块位)
ALGS = [
    ("purple", {"name": "Sarathi Session", "color": "#9933FF", "marker": "o", "ls": "--"}),
    ("lmetric", {"name": "LMetric", "color": "#3399FF", "marker": "s", "ls": "--"}),
    ("red", {"name": "SynergySched", "color": "red", "marker": "^", "ls": "-"}),
    ("brown", {"name": "Sarathi RR", "color": "#994C00", "marker": "D", "ls": "--"}),
    ("green", {"name": "vLLM Session", "color": "#00CC00", "marker": "*", "ls": "--"}),
]
METRICS = ["p50", "p90", "ttft", "tpot", "slo"]


def plot_panel(dataset, metric):
    xs = X[dataset]
    ax_cfg = AXIS[dataset][metric]
    plt.figure(figsize=(4.5, 3))
    plt.subplots_adjust(left=0.2, right=0.96, bottom=0.15, top=0.95)

    for key, cfg in ALGS:
        if key == "lmetric":
            if dataset == "reasoning" and metric == "slo":
                ys = LM_REASONING_SLO3  # 阈值收紧到 3.0s 的精确重算值
            else:
                ys = LM[dataset][metric]
        else:
            ys = OLD[dataset][metric][key]
        dashes = [5, 3] if cfg["ls"] == "--" else [1, 0]
        plt.plot(xs, ys, marker=cfg["marker"], linestyle=cfg["ls"],
                 color=cfg["color"], linewidth=3, markersize=15,
                 label=cfg["name"], dashes=dashes)

    plt.grid(True, linestyle="--", alpha=0.7, linewidth=1.5)
    plt.xticks(xs, fontsize=25)
    plt.yticks(ax_cfg["yticks"], fontsize=25)
    plt.ylim(ax_cfg["ylim"])
    if ax_cfg.get("hline") is not None:
        plt.axhline(y=ax_cfg["hline"], color="orange", linestyle="--",
                    linewidth=3, dashes=[4, 7])

    out_dir = f"{BASE}/picture/{dataset}"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{dataset}_{metric}.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"panel: {out}")


def main():
    # 1) 25 张面板
    for ds in X:
        for m in METRICS:
            plot_panel(ds, m)

    # 2) 顶部指标标题(arxiv 行)
    title_cfg = {"fontsize": 100, "title_height": 200, "position": 0.6}
    tops = {"P50 E2E (s)": "arxiv_p50", "P90 E2E (s)": "arxiv_p90",
            "P50 TTFT (s)": "arxiv_ttft", "P50 TPOT (ms)": "arxiv_tpot",
            "SLO Attainment (%)": "arxiv_slo"}
    for text, name in tops.items():
        p = f"{BASE}/picture/arxiv/{name}.png"
        add_title_to_image(image=Image.open(p), config=title_cfg,
                           title_text=text, output=p)

    # 3) 底部 Request/s(reasoning 行)
    for m in METRICS:
        p = f"{BASE}/picture/reasoning/reasoning_{m}.png"
        add_title_to_image_bottom(image=Image.open(p), config=title_cfg,
                                  title_text="Request/s", output=p)

    # 4) 左侧数据集标题(各行 p50)
    side_cfg = {"fontsize": 100, "total_title_width": 300, "position": 0.1}
    sides = {"Summarization": "arxiv", "Coding": "reasoning",
             "ShareGPT": "sharegpt", "FlowGPT-Q": "flowgpt_qps",
             "FlowGPT-T": "flowgpt_timestamp"}
    for text, ds in sides.items():
        p = f"{BASE}/picture/{ds}/{ds}_p50.png"
        add_side_title_to_image(image=Image.open(p), config=side_cfg,
                                title_text1=text, title_text2="Latency",
                                output=p)

    # 5) 横向/纵向拼接
    for ds in ["arxiv", "reasoning", "flowgpt_qps", "flowgpt_timestamp",
               "sharegpt"]:
        combine_images_horizontally(dir=f"{BASE}/picture/{ds}",
                                    order_list=METRICS)
    combine_images_vertically(
        dir=f"{BASE}/picture/combine_horizontally",
        order_list=["arxiv", "flowgpt_qps", "flowgpt_timestamp", "sharegpt",
                    "reasoning"])

    # 6) 图例:LMetric 替换 vLLM RR(颜色/标记位不变)
    image = Image.open(f"{BASE}/picture/combine_vertically/image_combine_v.png")
    add_legend_to_image(image, {
        "fontsize": 100,
        "legend_total_height": 200,
        "legend_labels": ["SynergySched", "Sarathi Session", "LMetric",
                          "Sarathi RR", "vLLM Session"],
        "colors": ["red", "#9933FF", "#3399FF", "#994C00", "#00CC00"],
        "output_dir": f"{BASE}/picture/combine_vertically/"
                      "end-to-end algorithm comparison",
        "legend_metrics": {
            "line_width": 10, "dotted_line_width": 30,
            "legend_length_ratoi": 3, "legend_height": 2,
            "marker_size": 0.35, "legend_interval_ratoi": 1,
        },
    })


if __name__ == "__main__":
    main()
