from PIL import Image
import os

from draw_single import draw_sigle_image
from picture_process import add_title_to_image_bottom
from picture_process import add_title_to_image,add_side_title_to_image,get_image_paths
from picture_process import combine_images_horizontally,combine_images_vertically,add_legend_to_image

def add_top_title(base_dir):
    add_top_title_items = {"P50 E2E (s)":f"{base_dir}/picture/arxiv/arxiv_p50.png",
                            "P90 E2E (s)":f"{base_dir}/picture/arxiv/arxiv_p90.png",
                            "P50 TTFT (s)":f"{base_dir}/picture/arxiv/arxiv_ttft.png",  
                            "P50 TPOT (ms)":f"{base_dir}/picture/arxiv/arxiv_tpot.png",
                            "SLO Attainment (%)":f"{base_dir}/picture/arxiv/arxiv_slo.png"}
    
    config = {
        "fontsize":  100,
        "title_height":200,
        "position":0.6
    }
    for item in add_top_title_items:
        add_title_to_image(image = Image.open(add_top_title_items[item]),config = config, title_text = item, 
                           output = add_top_title_items[item])

def add_bottom_title(base_dir):
    add_bottom_title_items = {f"{base_dir}/picture/reasoning/reasoning_p50.png",
                              f"{base_dir}/picture/reasoning/reasoning_p90.png",
                              f"{base_dir}/picture/reasoning/reasoning_ttft.png",  
                              f"{base_dir}/picture/reasoning/reasoning_tpot.png",
                              f"{base_dir}/picture/reasoning/reasoning_slo.png"}
    config = {
        "fontsize":  100,
        "title_height":200,
        "position":0.6
    }
    
    for item in add_bottom_title_items:
        add_title_to_image_bottom(image = Image.open(item),config = config, title_text = "Request/s", 
                           output = item)

def add_side_title(base_dir):
    # 在第一步处理的图片后添加侧标题
    add_side_title_items = {
        "Summarization":f"{base_dir}/picture/arxiv/arxiv_p50.png",
        "Coding":f"{base_dir}/picture/reasoning/reasoning_p50.png",
        "ShareGPT":f"{base_dir}/picture/sharegpt/sharegpt_p50.png",
        "FlowGPT-Q":f"{base_dir}/picture/flowgpt_qps/flowgpt_qps_p50.png",
        "FlowGPT-T":f"{base_dir}/picture/flowgpt_timestamp/flowgpt_timestamp_p50.png",
    }
    config = {
        "fontsize":  100,
        "total_title_width":300,
        "position":0.1 # 与右边距离比例
    }

    for item in add_side_title_items:
        add_side_title_to_image(image = Image.open(add_side_title_items[item]),config = config, title_text1 = item,
                                title_text2 = "Latency", output=add_side_title_items[item])

def combine_horizontally(base_dir):

    dirs = [f"{base_dir}/picture/arxiv",
            f"{base_dir}/picture/reasoning",
            f"{base_dir}/picture/flowgpt_qps",
            f"{base_dir}/picture/flowgpt_timestamp",
            f"{base_dir}/picture/sharegpt"]
    order_list = ['p50','p90','ttft','tpot','slo']
    for dir in dirs:
        combine_images_horizontally(dir = dir,order_list = order_list)

def combine_vertically(base_dir):

    dir = f"{base_dir}/picture/combine_horizontally"
    order_list = ['arxiv','flowgpt_qps','flowgpt_timestamp','sharegpt','reasoning',]
    combine_images_vertically(dir = dir,order_list = order_list)

def add_legend(base_dir):

    image = Image.open(f"{base_dir}/picture/combine_vertically/image_combine_v.png")
    config = {
        "fontsize":100,
        "legend_total_height":200,
        "legend_labels":["SynergySched","Sarathi Session","vLLM RR","Sarathi RR","vLLM Session"],
        "colors":["red","#9933FF","#3399FF","#994C00","#00CC00"],
        "output_dir":f"{base_dir}/picture/combine_vertically/end-to-end algorithm comparison",
        "legend_metrics":{
            "line_width":10,          # 图例的线的宽度
            "dotted_line_width":30,  # 虚线的点的长度
            "legend_length_ratoi":3, # 一个图例线的长度比例
            "legend_height":2,       # 图例与顶部的距离比例
            "marker_size":0.35,         # 图例标记的大小比例    
            "legend_interval_ratoi":1 # 图例之间的间隔比例
        }
    }
    add_legend_to_image(image,config)

if __name__ == "__main__":

    # 获取当前文件的完整路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    base_dir = current_dir

    draw_sigle_image(base_dir)
    add_top_title(base_dir)
    add_bottom_title(base_dir)
    add_side_title(base_dir)
    combine_horizontally(base_dir) # 横向合并
    combine_vertically(base_dir) # 纵向合并
    add_legend(base_dir)