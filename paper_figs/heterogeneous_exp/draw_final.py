from PIL import Image
import os

from draw import draw_image
from picture_process import add_title_to_image,add_side_title_to_image,get_image_paths_from_directory
from picture_process import combine_images_horizontally,combine_images_vertically,add_legend_to_image

def data2images(base_dir):
    # 生成原始数据图片
    data2images_items = {
        "Coder":{"data_dir":f"{base_dir}/result/Coder","e2e-slo":12},
        "ShareGPT":{"data_dir":f"{base_dir}/result/ShareGPT","e2e-slo":10},
        "Flowgpt-qps":{"data_dir":f"{base_dir}/result/Flowgpt-qps","e2e-slo":4.5},
        "Flowgpt-timestamp":{"data_dir":f"{base_dir}/result/Flowgpt-timestamp","e2e-slo":9},
    }

    for item in data2images_items:
        draw_image(data2images_items[item]["data_dir"],data2images_items[item]["e2e-slo"])

def add_top_title(base_dir):
    add_top_title_items = {"p50":f"{base_dir}/picture/ShareGPT/ShareGPT_p50.png",
                                "p90":f"{base_dir}/picture/ShareGPT/ShareGPT_p90.png",
                                "p95":f"{base_dir}/picture/ShareGPT/ShareGPT_p95.png",
                                "p99":f"{base_dir}/picture/ShareGPT/ShareGPT_p99.png",    
                                "slo_attainment":f"{base_dir}/picture/ShareGPT/ShareGPT_slo.png"}
    
    for item in add_top_title_items:
        add_title_to_image(image = Image.open(add_top_title_items[item]), title_text = item, 
                           output = add_top_title_items[item])

def add_side_title(base_dir):
        # 在第一步处理的图片后添加侧标题
    add_side_title_items = {
        "ShareGPT":f"{base_dir}/picture/ShareGPT/ShareGPT_p50.png",
        "Coder":f"{base_dir}/picture/Coder/Coder_p50.png",
        "Flowgpt-qps":f"{base_dir}/picture/Flowgpt-qps/Flowgpt-qps_p50.png",
        "Flowgpt-timestamp":f"{base_dir}/picture/Flowgpt-timestamp/Flowgpt-timestamp_p50.png",
    }

    for item in add_side_title_items:
        add_side_title_to_image(image = Image.open(add_side_title_items[item]), title_text1 = item,
                                title_text2 = "Latency", output=add_side_title_items[item])

def horizontally_merge(base_dir):
    horizontally_merge_items = {
        "ShareGPT":f"{base_dir}/picture/ShareGPT",
        "Coder":f"{base_dir}/picture/Coder",
        "Flowgpt-qps":f"{base_dir}/picture/Flowgpt-qps",
        "Flowgpt-timestamp":f"{base_dir}/picture/Flowgpt-timestamp",
    }

    for item in horizontally_merge_items:
        image_paths = get_image_paths_from_directory(horizontally_merge_items[item])
        combine_images_horizontally(input_paths = image_paths,dataset = item)

def vertically_merge(base_dir):
    image_paths = get_image_paths_from_directory(f"{base_dir}/picture/merge_horizontally")
    combine_images_vertically(input_paths = image_paths)

def add_legend(base_dir):
    add_legend_to_image(Image.open(f"{base_dir}/picture/merge_vertically/merge_vertically.png"))

if __name__ == "__main__":

    base_dir = "."

    data2images(base_dir)
    add_top_title(base_dir)
    add_side_title(base_dir)
    horizontally_merge(base_dir)
    vertically_merge(base_dir)
    add_legend(base_dir)