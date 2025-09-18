import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


def apply_topconf_style():
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'sans-serif',
        # 优先尝试支持中文的字体，缺失则回退
        'font.sans-serif': [
            'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei', 'SimHei',
            'DejaVu Sans', 'Liberation Sans', 'Arial'
        ],
        'axes.unicode_minus': False,
        'axes.linewidth': 1.2,
        'axes.titlesize': 16,
        'axes.labelsize': 18,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'figure.dpi': 160,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'grid.alpha': 0.25,
        'path.simplify': True,
        'path.simplify_threshold': 0.5,
        'agg.path.chunksize': 10000,
    })


def _map_instance_label(raw: str) -> str:
    mapping = {
        'http://65.49.81.73:8769': 'H100',
        'http://184.105.190.123:8769': '2xA6000',
        'http://184.105.190.57:8769': 'A100',
    }
    return mapping.get(str(raw), str(raw))


def _map_method_label(raw: str) -> str:
    mapping = {
        # 在此处添加方法名映射，例如：
        'latency_based': 'Latency-Based',
        'least_load': 'Least-Loaded',
        'our': 'SynergySched',
        'weight_based': 'Weighted-Based',
    }
    return mapping.get(str(raw), str(raw))


def compute_share_by_instance(mean_csv: Path, var_csv: Path):
    """
    返回：
      - methods: 方法列表
      - method_to_share: dict[method] -> DataFrame(index=dataset, columns=instance) 的比例矩阵
      - method_to_varshare: dict[method] -> DataFrame(index=dataset, columns=instance) 的归一化方差（到比例尺度）
    说明：对每个 dataset+method，先汇总该组内所有实例的 mean 求和得到 total_mean，
         每个实例的请求比例 = mean_instance / total_mean；
         将实例的方差 variance 归一化到比例尺度：var_share = variance / (total_mean**2)（total_mean=0时记为0）。
    """
    mean_df = pd.read_csv(mean_csv)
    var_df = pd.read_csv(var_csv)

    df = pd.merge(mean_df, var_df, on=['dataset', 'method', 'instance'], how='inner')
    # 实例名映射为硬件标签
    df['instance'] = df['instance'].map(_map_instance_label)

    # 计算每个 dataset+method 的总 mean
    totals = (
        df.groupby(['dataset', 'method'], as_index=False)['mean']
        .sum()
        .rename(columns={'mean': 'total_mean'})
    )
    df = pd.merge(df, totals, on=['dataset', 'method'], how='left')

    # 比例与归一化方差
    df['share'] = np.where(df['total_mean'] > 0, df['mean'] / df['total_mean'], 0.0)
    df['var_share'] = np.where(df['total_mean'] > 0, df['variance'] / (df['total_mean'] ** 2), 0.0)

    # 映射方法名（用于图上展示）
    df['method'] = df['method'].map(_map_method_label)
    methods = sorted(df['method'].unique().tolist())
    datasets = sorted(df['dataset'].unique().tolist())
    method_to_share = {}
    method_to_varshare = {}

    for m in methods:
        sub = df[df['method'] == m].copy()
        # 确保列（实例）全集一致
        instances = sorted(sub['instance'].unique().tolist())
        share_pivot = sub.pivot(index='dataset', columns='instance', values='share').reindex(index=datasets).fillna(0.0)
        var_pivot = sub.pivot(index='dataset', columns='instance', values='var_share').reindex(index=datasets).fillna(0.0)
        # 统一列顺序
        share_pivot = share_pivot.reindex(columns=instances)
        var_pivot = var_pivot.reindex(columns=instances)
        method_to_share[m] = share_pivot
        method_to_varshare[m] = var_pivot

    return methods, method_to_share, method_to_varshare


def plot_grouped_stacked(methods, method_to_share, method_to_varshare, out_path_prefix: Path):
    apply_topconf_style()

    # 统一数据集与实例全集
    all_datasets = sorted({ds for m in methods for ds in method_to_share[m].index})
    all_instances = sorted({inst for m in methods for inst in method_to_share[m].columns})

    num_datasets = len(all_datasets)
    num_methods = len(methods)
    num_instances = len(all_instances)

    # 动态尺寸：每个数据集组宽 ~ 0.9，方法数量越多宽度越大
    width = max(10.0, min(22.0, 0.9 * num_datasets * max(1.0, num_methods * 0.8)))
    height = 4.2 + 0.15 * max(0, num_instances - 4)
    fig, ax = plt.subplots(figsize=(width, height))

    x = np.arange(num_datasets)
    group_width = 0.84
    bar_width = group_width / max(num_methods, 1)

    # 实例颜色
    cmap = plt.get_cmap('tab20')
    colors = {inst: cmap(i % 20) for i, inst in enumerate(all_instances)}

    # 方法的纹理样式帮助区分
    hatch_list = ['', '///', 'xxx', '...', '+++', '***']

    max_y = 0.0
    for m_idx, m in enumerate(methods):
        share_pivot = method_to_share[m].reindex(index=all_datasets, columns=all_instances).fillna(0.0)
        var_pivot = method_to_varshare[m].reindex(index=all_datasets, columns=all_instances).fillna(0.0)

        for ds_idx, ds in enumerate(all_datasets):
            bottom = 0.0
            xpos = x[ds_idx] + (m_idx - (num_methods - 1) / 2) * bar_width
            for inst in all_instances:
                h = float(share_pivot.loc[ds, inst])
                v = float(var_pivot.loc[ds, inst])
                if h <= 0 and v <= 0:
                    continue
                bar = ax.bar(
                    xpos,
                    h,
                    width=bar_width * 0.95,
                    color=colors[inst],
                    edgecolor='none',
                    hatch=hatch_list[m_idx % len(hatch_list)],
                    bottom=bottom,
                )
                # 误差线：以段顶端为锚点
                if v > 0:
                    ax.errorbar(
                        xpos,
                        bottom + h,
                        yerr=v,
                        fmt='none',
                        ecolor='black',
                        elinewidth=1.0,
                        capsize=2,
                        capthick=1.0,
                    )
                bottom += h
            max_y = max(max_y, bottom)

    ax.set_xticks(x)
    ax.set_xticklabels(all_datasets, rotation=15, ha='center')
    ax.tick_params(axis='x', labelsize=12)
    ax.set_ylabel('Request share')
    ax.set_xlabel('Dataset')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylim(0, min(1.05, max(0.2, max_y * 1.12)))

    ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.35)

    # 实例图例
    instance_handles = [mpl.patches.Patch(facecolor=colors[inst], edgecolor='none', label=str(inst)) for inst in all_instances]
    leg1 = ax.legend(handles=instance_handles, title='Instance', ncol=min(len(all_instances), 5), loc='upper left', bbox_to_anchor=(0.0, 1.02), fontsize=12, title_fontsize=12, frameon=False)
    ax.add_artist(leg1)

    # 方法图例（使用空白矩形+不同hatch）
    method_handles = [mpl.patches.Patch(facecolor='white', edgecolor='black', hatch=hatch_list[i % len(hatch_list)], label=str(m)) for i, m in enumerate(methods)]
    ax.legend(handles=method_handles, title='Method', ncol=min(len(methods), 5), loc='upper right', bbox_to_anchor=(1.0, 1.02), fontsize=12, title_fontsize=12, frameon=False)

    out_prefix = out_path_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        out_path = Path(f"{out_prefix}.{ext}")
        fig.savefig(out_path)
        print(f"Saved figure: {out_path}")

    plt.close(fig)


def plot_facet_by_method(methods, method_to_share, method_to_varshare, out_path_prefix: Path):
    apply_topconf_style()

    # Global unions to keep consistent order/colors
    all_datasets = sorted({ds for m in methods for ds in method_to_share[m].index})
    all_instances = sorted({inst for m in methods for inst in method_to_share[m].columns})

    num_methods = len(methods)
    num_datasets = len(all_datasets)
    num_instances = len(all_instances)

    # Layout: up to 2x2; if more than 4 methods, expand rows
    ncols = 2 if num_methods > 1 else 1
    nrows = int(np.ceil(num_methods / ncols))
    width = max(10.0, 4.0 * ncols + 0.8 * num_datasets)
    height = max(3.2 * nrows, 3.2)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width, height), squeeze=False, sharey=True)

    # Colors for instances
    cmap = plt.get_cmap('tab20')
    colors = {inst: cmap(i % 20) for i, inst in enumerate(all_instances)}

    x = np.arange(num_datasets)
    group_width = 0.84
    bar_width = group_width / max(num_instances, 1)

    max_y = 0.0
    for idx, m in enumerate(methods):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        share_pivot = method_to_share[m].reindex(index=all_datasets, columns=all_instances).fillna(0.0)
        var_pivot = method_to_varshare[m].reindex(index=all_datasets, columns=all_instances).fillna(0.0)

        for j, inst in enumerate(all_instances):
            shares = share_pivot[inst].to_numpy()
            varshares = var_pivot[inst].to_numpy()
            offsets = (j - (num_instances - 1) / 2) * bar_width
            ax.bar(
                x + offsets,
                shares,
                width=bar_width * 0.95,
                color=colors[inst],
                edgecolor='none',
                label=str(inst) if idx == 0 else None,
                yerr=varshares,
                error_kw=dict(ecolor='black', elinewidth=1.0, capsize=2, capthick=1.0),
            )
            max_y = max(max_y, np.nanmax(shares + varshares))

        ax.set_title(str(m))
        ax.set_xticks(x)
        ax.set_xticklabels(all_datasets, rotation=15, ha='center')
        ax.tick_params(axis='x', labelsize=12)
        if r == nrows - 1:
            ax.set_xlabel('Dataset')
        if c == 0:
            ax.set_ylabel('Request share')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.35)

    for ax in axes.flat:
        ax.set_ylim(0, min(1.05, max(0.2, max_y * 1.12)))

    # Global legend for instances
    instance_handles = [mpl.patches.Patch(facecolor=colors[inst], edgecolor='none', label=str(inst)) for inst in all_instances]
    fig.legend(handles=instance_handles, title='Instance', ncol=min(len(all_instances), 6), loc='upper center', bbox_to_anchor=(0.5, 1.02), fontsize=12, title_fontsize=12, frameon=False)

    out_prefix = out_path_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        out_path = Path(f"{out_prefix}.{ext}")
        fig.savefig(out_path)
        print(f"Saved figure: {out_path}")

    plt.close(fig)


def plot_small_figure(methods, method_to_share, method_to_varshare, dataset: str, out_path_prefix: Path):
    apply_topconf_style()

    # 确定实例全集，保持颜色一致
    all_instances = sorted({inst for m in methods for inst in method_to_share[m].columns})
    cmap = plt.get_cmap('tab20')
    colors = {inst: cmap(i % 20) for i, inst in enumerate(all_instances)}

    # 过滤仅该数据集的数据；若该数据集不存在，则回退为可用的第一个
    available_datasets = sorted({ds for m in methods for ds in method_to_share[m].index})
    ds_name = dataset if dataset in available_datasets else (available_datasets[0] if available_datasets else dataset)

    num_methods = len(methods)
    num_instances = len(all_instances)
    x = np.arange(num_methods)
    # 更小的组宽，增大柱间距，避免任何“堆叠”的视觉误解
    total_group_width = 0.8
    bar_width = total_group_width / max(num_instances, 1)

    # 小图尺寸紧凑
    fig, ax = plt.subplots(figsize=(6.5, 3.4))

    max_y = 0.0
    for j, inst in enumerate(all_instances):
        shares = []
        varshares = []
        for m in methods:
            share_pivot = method_to_share[m]
            var_pivot = method_to_varshare[m]
            val = float(share_pivot.loc[ds_name, inst]) if (ds_name in share_pivot.index and inst in share_pivot.columns) else 0.0
            vval = float(var_pivot.loc[ds_name, inst]) if (ds_name in var_pivot.index and inst in var_pivot.columns) else 0.0
            shares.append(val)
            varshares.append(vval)
        shares = np.array(shares)
        varshares = np.array(varshares)
        offsets = (j - (num_instances - 1) / 2) * bar_width
        bars = ax.bar(
            x + offsets,
            shares,
            width=bar_width * 0.9,
            color=colors[inst],
            edgecolor='black',
            linewidth=0.6,
            alpha=0.95,
            label=str(inst),
            yerr=varshares * 0.3,
            error_kw=dict(ecolor='black', elinewidth=1.0, capsize=2, capthick=1.0, zorder=3),
            zorder=2,
        )
        max_y = max(max_y, np.nanmax(shares + varshares))

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in methods], rotation=15, ha='center')
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Request Ratio (%)')
    ax.set_title('')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, pos: f'{y*100:.0f}'))
    ax.set_ylim(0, 0.9)

    ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.35)
    ax.legend(ncol=min(len(all_instances), 4), fontsize=12, frameon=False, loc='upper right')

    out_prefix = out_path_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        out_path = Path(f"{out_prefix}.{ext}")
        fig.savefig(out_path)
        print(f"Saved figure: {out_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot request share per dataset and method with variance labels.')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent / 'hete_sched_data', help='Root directory with qps_mean.csv and qps_variance.csv')
    parser.add_argument('--mean_csv', type=Path, default=None, help='Path to qps_mean.csv (long table)')
    parser.add_argument('--var_csv', type=Path, default=None, help='Path to qps_variance.csv (long table)')
    parser.add_argument('--out', type=Path, default=None, help='Output figure path prefix (without extension)')
    parser.add_argument('--dataset', type=str, default='flowgpt-timestamp', help='Single dataset to plot for small figure')
    parser.add_argument('--single_only', action='store_true', default=True, help='Only render the small figure for a single dataset')
    args = parser.parse_args()

    mean_csv = args.mean_csv or (args.root / 'qps_mean.csv')
    var_csv = args.var_csv or (args.root / 'qps_variance.csv')
    out_prefix = args.out or (args.root / 'qps_share_bar')

    if not mean_csv.exists() or not var_csv.exists():
        raise SystemExit(f"Missing input CSVs. mean={mean_csv}, var={var_csv}")

    methods, method_to_share, method_to_varshare = compute_share_by_instance(mean_csv, var_csv)
    if len(methods) == 0:
        print('No data to plot.')
        return

    # 输出合并图（堆叠）与分面图两种风格
    if out_prefix.suffix != '':
        out_prefix = out_prefix.with_suffix('')
    stacked_prefix = out_prefix if out_prefix.name else (args.root / 'qps_share_grouped_stacked')
    facet_prefix = (stacked_prefix.parent / 'qps_share_facet')
    if not args.single_only:
        plot_grouped_stacked(methods, method_to_share, method_to_varshare, stacked_prefix)
        plot_facet_by_method(methods, method_to_share, method_to_varshare, facet_prefix)

    # 单数据集小图
    small_prefix = (stacked_prefix.parent / f'qps_share_small_{args.dataset}')
    plot_small_figure(methods, method_to_share, method_to_varshare, args.dataset, small_prefix)


if __name__ == '__main__':
    main()


