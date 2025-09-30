输入: `/home/paperspace/zhangy/vllm-workspace/vllm/exp/profiling_result/scheduler_profiling_chunk_4096.jsonl`
筛选条件: total_scheduled_tokens > 2048, model_run_duration_ms ∈ [42.0, 70.0]
筛选后样本数: 253

## 全局特征影响 (Spearman 与线性回归)
- Spearman Top 特征:
- 线性回归: R^2=0.000, 重要系数(绝对值排序):

## 按精确 total_scheduled_tokens 分组

---
### total_scheduled_tokens = 2204 (n=2)
耗时分布: min=51.790, p25=51.790, p50=59.994, p75=68.199, max=68.199, std=11.603, iqr=16.409
与高耗时组~低耗时组的均值差异(退化为上下半区时也类似):
- num_running_reqs: high_mean=12.000, low_mean=11.000, delta=1.000
- num_waiting_reqs: high_mean=0.000, low_mean=0.000, delta=0.000
- schedule_duration_ms: high_mean=0.493, low_mean=0.645, delta=-0.152
- chunk_count: high_mean=12.000, low_mean=11.000, delta=1.000
- chunk_sum: high_mean=2204.000, low_mean=2204.000, delta=0.000
- chunk_mean: high_mean=183.667, low_mean=200.364, delta=-16.697
- chunk_std: high_mean=632.776, low_mean=661.214, delta=-28.438
- chunk_p95: high_mean=987.400, low_mean=1097.500, delta=-110.100

---
### total_scheduled_tokens = 2062 (n=2)
耗时分布: min=48.286, p25=48.286, p50=55.562, p75=62.838, max=62.838, std=10.290, iqr=14.552
与高耗时组~低耗时组的均值差异(退化为上下半区时也类似):
- num_running_reqs: high_mean=8.000, low_mean=8.000, delta=0.000
- num_waiting_reqs: high_mean=0.000, low_mean=0.000, delta=0.000
- schedule_duration_ms: high_mean=0.806, low_mean=0.894, delta=-0.088
- chunk_count: high_mean=8.000, low_mean=8.000, delta=0.000
- chunk_sum: high_mean=2062.000, low_mean=2062.000, delta=0.000
- chunk_mean: high_mean=257.750, low_mean=257.750, delta=0.000
- chunk_std: high_mean=726.199, low_mean=726.199, delta=0.000
- chunk_p95: high_mean=1336.100, low_mean=1336.100, delta=0.000

---
### total_scheduled_tokens = 2061 (n=3)
耗时分布: min=47.742, p25=47.742, p50=62.102, p75=62.550, max=62.550, std=8.423, iqr=14.808
与高耗时组~低耗时组的均值差异(退化为上下半区时也类似):
- num_running_reqs: high_mean=9.000, low_mean=8.000, delta=1.000
- num_waiting_reqs: high_mean=0.000, low_mean=0.000, delta=0.000
- schedule_duration_ms: high_mean=0.747, low_mean=0.901, delta=-0.154
- chunk_count: high_mean=9.000, low_mean=8.000, delta=1.000
- chunk_sum: high_mean=2061.000, low_mean=2061.000, delta=0.000
- chunk_mean: high_mean=229.000, low_mean=257.625, delta=-28.625
- chunk_std: high_mean=684.000, low_mean=725.845, delta=-41.845
- chunk_p95: high_mean=1232.200, low_mean=1335.450, delta=-103.250

---
### total_scheduled_tokens = 2093 (n=3)
耗时分布: min=48.793, p25=48.793, p50=61.942, p75=62.384, max=62.384, std=7.722, iqr=13.591
与高耗时组~低耗时组的均值差异(退化为上下半区时也类似):
- num_running_reqs: high_mean=7.000, low_mean=10.000, delta=-3.000
- num_waiting_reqs: high_mean=0.000, low_mean=0.000, delta=0.000
- schedule_duration_ms: high_mean=0.546, low_mean=0.947, delta=-0.402
- chunk_count: high_mean=7.000, low_mean=10.000, delta=-3.000
- chunk_sum: high_mean=2093.000, low_mean=2093.000, delta=0.000
- chunk_mean: high_mean=299.000, low_mean=209.300, delta=89.700
- chunk_std: high_mean=788.434, low_mean=658.702, delta=129.731
- chunk_p95: high_mean=1461.200, low_mean=1146.650, delta=314.550

---
### total_scheduled_tokens = 2110 (n=3)
耗时分布: min=49.918, p25=49.918, p50=62.063, p75=63.298, max=63.298, std=7.394, iqr=13.380
与高耗时组~低耗时组的均值差异(退化为上下半区时也类似):
- num_running_reqs: high_mean=12.000, low_mean=8.000, delta=4.000
- num_waiting_reqs: high_mean=0.000, low_mean=0.000, delta=0.000
- schedule_duration_ms: high_mean=0.914, low_mean=0.904, delta=0.010
- chunk_count: high_mean=12.000, low_mean=8.000, delta=4.000
- chunk_sum: high_mean=2110.000, low_mean=2110.000, delta=0.000
- chunk_mean: high_mean=175.833, low_mean=263.750, delta=-87.917
- chunk_std: high_mean=605.640, low_mean=743.169, delta=-137.529
- chunk_p95: high_mean=945.100, low_mean=1367.300, delta=-422.200

## 按 total_scheduled_tokens 分箱
分箱宽度: 64, 统计文件: per_token_bins_variance_w64.csv

---
### tokens_bin = 2240 (range≈[2240, 2294], n=17)
耗时分布: min=52.584, p25=54.615, p50=66.811, p75=67.319, max=69.084, std=6.709, iqr=12.704
与高耗时组~低耗时组的均值差异:
- num_running_reqs: high_mean=10.200, low_mean=9.400, delta=0.800
- num_waiting_reqs: high_mean=0.000, low_mean=0.000, delta=0.000
- schedule_duration_ms: high_mean=0.729, low_mean=0.934, delta=-0.205
- chunk_count: high_mean=10.200, low_mean=9.400, delta=0.800
- chunk_sum: high_mean=2278.200, low_mean=2263.400, delta=14.800
- chunk_mean: high_mean=225.250, low_mean=245.300, delta=-20.050
- chunk_std: high_mean=712.418, low_mean=740.506, delta=-28.088
- chunk_p95: high_mean=1225.560, low_mean=1308.970, delta=-83.410

---
### tokens_bin = 2176 (range≈[2177, 2234], n=36)
耗时分布: min=50.937, p25=63.818, p50=65.464, p75=66.343, max=68.413, std=5.478, iqr=2.525
与高耗时组~低耗时组的均值差异:
- num_running_reqs: high_mean=10.000, low_mean=10.000, delta=0.000
- num_waiting_reqs: high_mean=0.000, low_mean=0.000, delta=0.000
- schedule_duration_ms: high_mean=0.644, low_mean=0.838, delta=-0.195
- chunk_count: high_mean=10.000, low_mean=10.000, delta=0.000
- chunk_sum: high_mean=2205.556, low_mean=2199.000, delta=6.556
- chunk_mean: high_mean=225.213, low_mean=225.776, delta=-0.564
- chunk_std: high_mean=699.619, low_mean=679.714, delta=19.905
- chunk_p95: high_mean=1208.544, low_mean=1222.378, delta=-13.833

---
### tokens_bin = 2304 (range≈[2310, 2362], n=13)
耗时分布: min=53.623, p25=55.886, p50=56.436, p75=59.156, max=69.389, std=5.730, iqr=3.270
与高耗时组~低耗时组的均值差异:
- num_running_reqs: high_mean=9.500, low_mean=10.250, delta=-0.750
- num_waiting_reqs: high_mean=0.000, low_mean=0.000, delta=0.000
- schedule_duration_ms: high_mean=0.714, low_mean=0.847, delta=-0.133
- chunk_count: high_mean=9.500, low_mean=10.250, delta=-0.750
- chunk_sum: high_mean=2327.250, low_mean=2334.250, delta=-7.000
- chunk_mean: high_mean=254.630, low_mean=234.820, delta=19.810
- chunk_std: high_mean=763.280, low_mean=734.486, delta=28.794
- chunk_p95: high_mean=1332.675, low_mean=1251.650, delta=81.025

---
### tokens_bin = 2112 (range≈[2112, 2174], n=43)
耗时分布: min=49.786, p25=63.239, p50=63.938, p75=64.738, max=66.223, std=4.809, iqr=1.500
与高耗时组~低耗时组的均值差异:
- num_running_reqs: high_mean=10.909, low_mean=9.455, delta=1.455
- num_waiting_reqs: high_mean=0.000, low_mean=0.000, delta=0.000
- schedule_duration_ms: high_mean=0.696, low_mean=0.821, delta=-0.125
- chunk_count: high_mean=10.909, low_mean=9.455, delta=1.455
- chunk_sum: high_mean=2152.455, low_mean=2133.182, delta=19.273
- chunk_mean: high_mean=201.052, low_mean=239.876, delta=-38.823
- chunk_std: high_mean=652.945, low_mean=706.055, delta=-53.110
- chunk_p95: high_mean=1080.918, low_mean=1227.673, delta=-146.755

---
### tokens_bin = 2048 (range≈[2050, 2110], n=54)
耗时分布: min=47.742, p25=61.731, p50=62.443, p75=62.977, max=64.434, std=4.965, iqr=1.245
与高耗时组~低耗时组的均值差异:
- num_running_reqs: high_mean=10.286, low_mean=8.500, delta=1.786
- num_waiting_reqs: high_mean=0.000, low_mean=0.000, delta=0.000
- schedule_duration_ms: high_mean=0.682, low_mean=0.776, delta=-0.094
- chunk_count: high_mean=10.286, low_mean=8.500, delta=1.786
- chunk_sum: high_mean=2085.643, low_mean=2080.929, delta=4.714
- chunk_mean: high_mean=214.775, low_mean=250.939, delta=-36.163
- chunk_std: high_mean=661.139, low_mean=717.455, delta=-56.316
- chunk_p95: high_mean=1114.000, low_mean=1296.607, delta=-182.607
