#!/usr/bin/env python3
"""打印 A/B 两线已完成点的 csv_process 指标(在 paper_figs/e2e_exp 下运行)"""
import sys, glob
sys.path.insert(0, '.')
from csv_process import csv_process_dir

for cfg in ['sla_elrar_fair', 'sla_elrar_v2']:
    print('--- %s ---' % cfg)
    print('%-4s %-7s %-7s %-7s %-7s %-8s %-7s' % ('qps','aqps','p50','p90','p99','SLO10%','ttft'))
    for Q in [14, 15, 16, 17, 18]:
        d = 'data/flowgpt_qps/%s/flowgpt_qps_%s_%d' % (cfg, cfg, Q)
        if not glob.glob(d + '/*.csv'):
            print('%-4d (pending)' % Q)
            continue
        m = csv_process_dir(d, 10, 'flowgpt_qps', [30, 30])
        if m:
            print('%-4d %-7.2f %-7.2f %-7.2f %-7.2f %-8.1f %-7.2f' % (
                Q, m['actual_qps'], m['p50'], m['p90'], m['p99'],
                m['slo attainment'], m['ttft']))
