import os
from draw import draw

"""
    针对每一张图进行操作
"""



def draw_sigle_image(base_dir):
    # 修改需要细画的图
    items = [
        {
            "dataset":"arxiv",
            "dir":f"{base_dir}/data/arxiv",
            # "e2e_slo":15,
            "e2e_slo":20,
            "configs":[
                {"metrics_configs":[
                    {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50','title':'E2E p50'}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90','title':'E2E p90'}],
                "data_fiter":[30,45],},
                {"metrics_configs":[
                    {'name': 'ttft', 'label': 'TTFT (s)', 'suffix': 'ttft','title':'TTFT',"horizontal_line_y":2,'y_lim': (0,5)}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'tpot', 'label': 'TPOT (ms)', 'suffix': 'tpot','title':'TPOT',"horizontal_line_y":65,'y_lim': (40,180)}],
                "data_fiter":[30,30],},
                {"metrics_configs":[
                    {'name': 'slo attainment', 'label': 'slo attainment(15s) (%)', 'suffix': 'slo', 'title':'SLO Attainment','custom_yticks':[25,50,75,100]},],
                "data_fiter":[30,60],}
            ]
        },
        {
            "dataset":"reasoning",
            "dir":f"{base_dir}/data/reasoning",
            # "e2e_slo":2.79,
            "e2e_slo":2.95,
            "configs":[
                {"metrics_configs":[
                    {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50', 'y_lim': (2.0, 5)}],
                "data_fiter":[30,110],},
                {"metrics_configs":[
                    {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90','y_lim': (2.4, 4.8)}],
                "data_fiter":[0,190],},
                {"metrics_configs":[
                    {'name': 'ttft', 'label': 'TTFT (s)', 'suffix': 'ttft','y_lim': (-0.2, 1),"horizontal_line_y":0.4}],
                "data_fiter":[30,110],},
                {"metrics_configs":[
                    {'name': 'tpot', 'label': 'TPOT (ms)', 'suffix': 'tpot','y_lim': (23, 40),"horizontal_line_y":28}],
                "data_fiter":[30,110],},
                {"metrics_configs":[
                    {'name': 'slo attainment', 'label': 'slo attainment(5s) (%)', 'suffix': 'slo', 'custom_yticks':[25,50,75,100]}],
                "data_fiter":[0,190],}
                ]
        },
        {
            "dataset":"sharegpt",
            "dir":f"{base_dir}/data/sharegpt",
            "e2e_slo":7,
            "configs":[
                {"metrics_configs":[
                    {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50','y_lim': (5, 10)}],
                    "data_fiter":[30,60],},
                {"metrics_configs":[
                    {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90','y_lim': (6, 9)}],
                    "data_fiter":[30,190],},
                {"metrics_configs":[
                    {'name': 'ttft', 'label': 'TTFT (s)', 'suffix': 'ttft','y_lim': (-2, 9),"horizontal_line_y":2}],
                "data_fiter":[30,60],},
                {"metrics_configs":[
                    {'name': 'tpot', 'label': 'TPOT (ms)', 'suffix': 'tpot','y_lim': (16, 48),"horizontal_line_y":26}],
                "data_fiter":[30,60],},
                {"metrics_configs":[
                    {'name': 'slo attainment', 'label': 'slo attainment(7s) (%)', 'suffix': 'slo', 'custom_yticks':[25,50,75,100]},],
                "data_fiter":[30,190],}
            ]
        },
        {
            "dataset":"flowgpt_qps",
            "dir":f"{base_dir}/data/flowgpt_qps",
            "e2e_slo":12,
            "configs":[
                {"metrics_configs":[
                    {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50','y_lim': (7, 20)}],
                "data_fiter":[240,0],},
                {"metrics_configs":[
                    {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90','y_lim': (8, 34)}],
                "data_fiter":[240,0],},
                {"metrics_configs":[
                    {'name': 'ttft', 'label': 'TTFT (s)', 'suffix': 'ttft','y_lim': (0, 5),"horizontal_line_y":2}],
                "data_fiter":[240,0],},
                {"metrics_configs":[
                    {'name': 'tpot', 'label': 'TPOT (ms)', 'suffix': 'tpot',"horizontal_line_y":12}],
                "data_fiter":[240,0],},
                {"metrics_configs":[
                    {'name': 'slo attainment', 'label': 'slo attainment(10s) (%)', 'suffix': 'slo', 'custom_yticks':[25,50,75,100]},],
                "data_fiter":[240,0],}
                ]
        },
        {
            "dataset":"flowgpt_timestamp",
            "dir":f"{base_dir}/data/flowgpt_timestamp",
            "e2e_slo":26,
            # "e2e_slo":16,
            "configs":[
                {"metrics_configs":[
                    {'name': 'p50', 'label': 'P50 latency(s)', 'suffix': 'p50','y_lim': (6, 30)}],
                "data_fiter":[240,60],},
                {"metrics_configs":[
                    {'name': 'p90', 'label': 'P90 latency(s)', 'suffix': 'p90','y_lim': (10, 65)}],
                "data_fiter":[240,0],},
                {"metrics_configs":[
                    {'name': 'ttft', 'label': 'TTFT (s)', 'suffix': 'ttft','y_lim': (0, 9),"horizontal_line_y":2}],
                "data_fiter":[240,0],},
                {"metrics_configs":[
                    {'name': 'tpot', 'label': 'TPOT (ms)', 'suffix': 'tpot','y_lim': (6, 24),"horizontal_line_y":12}],
                "data_fiter":[240,0],},
                {"metrics_configs":[
                    {'name': 'slo attainment', 'label': 'slo attainment(20s) (%)', 'suffix': 'slo', 'custom_yticks':[25,50,75,100]},],
                "data_fiter":[240,60],}
                ]
        }
    ]

    select_dataset = [
        'arxiv',
        'sharegpt',
        'reasoning',
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