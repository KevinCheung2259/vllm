import pandas as pd
import argparse
import glob
import os

def csv_process_dir(path,e2e_slo,dataset,filter_sample=None):
    """
    处理单个数据目录下的所有csv文件
    """
    if dataset == "flowgpt_timestamp" or dataset == "flowgpt_qps":
        send_time = "send_time"
        end_time = "total_time"
        tpot_calculate = False
    else:
        send_time = "launch_time"
        end_time = "finish_time"
        tpot_calculate = True
    csv_files = glob.glob(os.path.join(path, '*.csv'))

    if not csv_files:
        return
    
    if not filter_sample:
        filter_sample = [30,30]
    # 保存所有数据
    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)
        df['latency'] = df[end_time] - df[send_time]
        if tpot_calculate:
            df['tpot'] = (df['generation_time']*1000)/df['generation_tokens']
        df = df.sort_values(send_time)

        # 过滤数据
        min_time = df[send_time].min()
        max_time = df[send_time].max()

        df = df[
            (df[send_time] > min_time + filter_sample[0]) &  
            (df[send_time] < max_time -filter_sample[1]) 
        ]

        all_data.append(df)

    # 合并
    data = pd.concat(all_data, ignore_index=True)

    return {
        'p50': data['latency'].quantile(0.5),
        'p90': data['latency'].quantile(0.9),
        'p95': data['latency'].quantile(0.95),
        'p99': data['latency'].quantile(0.99),
        'calculate_requests': len(data),
        'slo_attainment_requests':(data['latency'] <= e2e_slo).sum(),
        'slo attainment': ((data['latency'] <= e2e_slo).sum() / len(data)) * 100,
        'actual_qps':len(data)/(data[send_time].max() - data[send_time].min()),
        'ttft':data['ttft'].quantile(0.5),
        'tpot':data['tpot'].quantile(0.5),
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path',required=True,type=str)
    parser.add_argument('--e2e-slo', type=float, default=5)
    parser.add_argument('--dataset', type=str, default="flowgpt_qps")
    parser.add_argument('--filter_sample',type=int,nargs='+' , default=[30,30])

    args = parser.parse_args()

    metrics = csv_process_dir(args.path,args.e2e_slo,args.dataset,args.filter_sample)

    print(f"处理 {args.path} 目录下数据")
    print(f"一共处理 {metrics['calculate_requests']} 个 请求")
    print(f"Actual QPS: {metrics['actual_qps']:.2f} r/s")
    # print(f"P50 Latency: {metrics['p50']:.4f} s")
    # print(f"P90 Latency: {metrics['p90']:.4f} s")
    # print(f"P95 Latency: {metrics['p95']:.4f} s")
    # print(f"P99 Latency: {metrics['p99']:.4f} s")
    print(f"Slo attainment(Latency<={args.e2e_slo}s): {metrics['slo attainment']:.2f}% ({metrics['slo_attainment_requests']}/{metrics['calculate_requests']})\n")