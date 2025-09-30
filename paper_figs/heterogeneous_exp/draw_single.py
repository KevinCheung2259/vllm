import os
from draw import draw

"""
    针对每一张图进行操作
"""



def draw_sigle_image(base_dir):
    # 修改需要细画的图
    items = [
        {
            "dataset":"coder",
            "dir":f"{base_dir}/data/coder",
            "e2e_slo":12.5,
            "configs":[
                {"metrics_configs":[
                    {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50',}],
                "data_fiter":[30,0],},
                {"metrics_configs":[
                    {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90',}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'ttft', 'label': 'TTFT (s)', 'suffix': 'ttft',"horizontal_line_y":0.6}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'tpot', 'label': 'TPOT (ms)', 'suffix': 'tpot',"horizontal_line_y":20}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'slo attainment', 'label': 'slo attainment(12.5s) (%)', 'suffix': 'slo', 'y_lim': (25, 105)}],
                "data_fiter":[30,0],}
                ]
        },
        {
            "dataset":"sharegpt",
            "dir":f"{base_dir}/data/sharegpt",
            "e2e_slo":10,
            "configs":[
                {"metrics_configs":[
                    {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50','y_lim': (2.2, 3.5)}],
                    "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90','y_lim': (4, 16)}],
                    "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'ttft', 'label': 'TTFT (s)', 'suffix': 'ttft',"horizontal_line_y":0.2}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'tpot', 'label': 'TPOT (ms)', 'suffix': 'tpot',"horizontal_line_y":16}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'slo attainment', 'label': 'slo attainment(10s) (%)', 'suffix': 'slo', 'y_lim': (50, 104)},],
                "data_fiter":[30,100],}
            ]
        },
        {
            "dataset":"flowgpt_qps",
            "dir":f"{base_dir}/data/flowgpt_qps",
            "e2e_slo":10,
            "configs":[
                {"metrics_configs":[
                    {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50'}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90','y_lim': (0, 30)}],
                "data_fiter":[0,30],},
                {"metrics_configs":[
                    {'name': 'ttft', 'label': 'TTFT (s)', 'suffix': 'ttft','y_lim': (0, 3.5),"horizontal_line_y":3}],
                "data_fiter":[30,0],},
                {"metrics_configs":[
                    {'name': 'tpot', 'label': 'TPOT (ms)', 'suffix': 'tpot',"horizontal_line_y":5}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'slo attainment', 'label': 'slo attainment(10s) (%)', 'suffix': 'slo', 'y_lim': (10, 100)},],
                "data_fiter":[30,30],}
                ]
        },
        {
            "dataset":"flowgpt_timestamp",
            "dir":f"{base_dir}/data/flowgpt_timestamp",
            "e2e_slo":12,
            "configs":[
                {"metrics_configs":[
                    {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50'}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90','y_lim': (5,29)}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'ttft', 'label': 'TTFT (s)', 'suffix': 'ttft',"horizontal_line_y":3}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'tpot', 'label': 'TPOT (ms)', 'suffix': 'tpot',"horizontal_line_y":5}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'slo attainment', 'label': 'slo attainment(12s) (%)', 'suffix': 'slo', 'y_lim': (25, 102)},],
                "data_fiter":[30,0],}
                ]
        }
    ]

    select_dataset = [
        'sharegpt',
        'coder',
        'flowgpt_qps',
        'flowgpt_timestamp'
    ]
    select_metrics = [
        'p50',
        'p90',
        'ttft',
        'tpot',
        'slo attainment',
    ]
    for item in items:
        if item['dataset'] in select_dataset:
            for config in item['configs']:
                if config['metrics_configs'][0]['name'] in select_metrics:
                    draw(dir=item["dir"],e2e_slo = item['e2e_slo'],
                        metrics_configs=config["metrics_configs"],data_fiter=config["data_fiter"])

if __name__=='__main__':

    # 获取当前文件的完整路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    
    draw_sigle_image(current_dir)