from PIL import Image, ImageDraw, ImageFont, ImageColor
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

Image.MAX_IMAGE_PIXELS = None

def draw_dashed_line(draw, start, end, color, width, dash_length, gap_length):
    """手动绘制虚线
    
    参数:
        draw: ImageDraw对象
        start: 起点坐标 (x1, y1)
        end: 终点坐标 (x2, y2)
        color: 线条颜色
        width: 线条宽度
        dash_length: 虚线中实线部分的长度
        gap_length: 虚线中间隙部分的长度
    """
    # 计算线段的总长度
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = (dx ** 2 + dy ** 2) ** 0.5
    
    # 如果线段长度为0，直接返回
    if length == 0:
        return
    
    # 计算单位向量
    unit_dx = dx / length
    unit_dy = dy / length
    
    # 计算虚线模式的总长度（实线+间隙）
    pattern_length = dash_length + gap_length
    
    # 绘制虚线
    current_position = 0
    while current_position < length:
        # 计算当前虚线段的起点和终点
        segment_start_x = x1 + unit_dx * current_position
        segment_start_y = y1 + unit_dy * current_position
        
        # 计算当前虚线段的实际长度（不超过剩余长度）
        current_segment_length = min(dash_length, length - current_position)
        
        # 计算当前虚线段的终点
        segment_end_x = segment_start_x + unit_dx * current_segment_length
        segment_end_y = segment_start_y + unit_dy * current_segment_length
        
        # 绘制当前虚线段
        draw.line([(segment_start_x, segment_start_y), (segment_end_x, segment_end_y)], 
                 fill=color, width=width)
        
        # 更新当前位置，跳过间隙
        current_position += pattern_length

def get_image_paths_from_directory(directory):
    """从目录中获取所有图片文件路径并按名称排序"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} 不是有效的目录")
    
    image_paths = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(image_extensions):
            image_paths.append(os.path.join(directory, filename))
    
    if not image_paths:
        raise ValueError(f"在目录 {directory} 中未找到图片文件")
    return image_paths

# 添加新函数：在图片正上方中央添加文字标题
def add_title_to_image(image, title_text="",output=None,position=0.5):
    """在图片正上方中央添加文字标题"""

    # fontsize = 300 # 300是dpi 600时用
    # title_height = 800
    fontsize = 100
    title_height = 300

    # 创建新图片，高度增加标题区域
    new_width, new_height = image.width, image.height + title_height
    new_image = Image.new('RGB', (new_width, new_height), color='white')
    title_image = Image.new('RGB', (new_width, title_height), color='white')
    
    # 绘制标题
    draw = ImageDraw.Draw(title_image)
    
    # 尝试加载合适的字体
    try:
        from PIL import ImageFont
        # 尝试Linux/Mac系统常见字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
    except (IOError, OSError):
        try:
            # 尝试其他可能的字体路径
            font = ImageFont.truetype("Arial.ttf", fontsize)
            print("使用字体: Arial.ttf")
        except (IOError, OSError):
            print(f"使用默认字体大小")
            font = ImageFont.load_default()
    
    # 计算文本居中位置
    text_width = draw.textlength(title_text, font=font)
    text_x = (new_width - text_width) * position
    text_y = (title_height - fontsize) // 2  # 简化处理
    
    # 绘制文本
    draw.text((text_x, text_y), title_text, font=font, fill='black')
    
    # 将标题和原图合并
    new_image.paste(title_image, (0, 0))
    new_image.paste(image, (0, title_height))

    # 保存图片
    if not output:

        output = "./picture/process/add_title_to_image.png"

    new_image.save(output)
    print(f"添加标题的图片已保存至: {output}")
    
    return new_image

def add_side_title_to_image(image, title_text1="Title 1", title_text2="Title 2", position = 0.5, output = None):
    """在图片左侧添加两列从下往上垂直排列的文字标题"""
    
    # fontsize=300 # dpi 600时用
    # total_title_width=1800
    fontsize=100
    total_title_width=600

    # 创建新图片，宽度增加标题区域
    new_width, new_height = image.width + total_title_width, image.height
    new_image = Image.new('RGB', (new_width, new_height), color='white')
    
    # 计算每列标题的宽度（均分总宽度）
    column_width = total_title_width // 2
    
    # 为两列标题分别创建临时图像
    # 第一列标题
    temp_width1 = new_height
    temp_height1 = column_width
    temp_image1 = Image.new('RGB', (temp_width1, temp_height1), color='white')
    
    # 第二列标题
    temp_width2 = new_height
    temp_height2 = column_width
    temp_image2 = Image.new('RGB', (temp_width2, temp_height2), color='white')
    
    # 尝试加载合适的字体
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("Arial.ttf", fontsize)
        except (IOError, OSError):
            print("无法加载指定大小的字体，使用默认字体大小")
            font = ImageFont.load_default()
    
    # 绘制第一列标题
    draw1 = ImageDraw.Draw(temp_image1)
    text_width1 = draw1.textlength(title_text1, font=font)
    text_x1 = (temp_width1 - text_width1) // 2  # 水平居中
    text_y1 = (temp_height1 - fontsize) * position
    draw1.text((text_x1, text_y1), title_text1, font=font, fill='black')
    
    # 绘制第二列标题
    draw2 = ImageDraw.Draw(temp_image2)
    text_width2 = draw2.textlength(title_text2, font=font)
    text_x2 = (temp_width2 - text_width2) // 2  # 水平居中
    text_y2 = (temp_height2 - fontsize) * position
    draw2.text((text_x2, text_y2), title_text2, font=font, fill='black')
    
    # 将两个临时图像逆时针旋转90度，使文字从下往上排列
    rotated_title1 = temp_image1.rotate(90, expand=True)
    rotated_title2 = temp_image2.rotate(90, expand=True)
    
    # 创建一个组合标题图像，用于放置两列标题
    combined_title_height = max(rotated_title1.height, rotated_title2.height)
    combined_title_width = rotated_title1.width + rotated_title2.width
    combined_title = Image.new('RGB', (combined_title_width, combined_title_height), color='white')
    
    # 计算每列标题的垂直居中位置
    title1_y = (combined_title_height - rotated_title1.height) // 2
    title2_y = (combined_title_height - rotated_title2.height) // 2
    
    # 将两列标题粘贴到组合标题图像中
    combined_title.paste(rotated_title1, (0, title1_y))
    combined_title.paste(rotated_title2, (rotated_title1.width, title2_y))
    
    # 将组合标题和原图合并到最终图像
    final_title_y = (new_height - combined_title.height) // 2
    new_image.paste(combined_title, (0, final_title_y))
    new_image.paste(image, (total_title_width, 0))
    

    # 保存图片
    if not output:

        output = "./picture/process/add_side_title_to_image.png"

    new_image.save(output)
    print(f"添加左侧标题的图片已保存至: {output}")

    return new_image
    
def add_legend_to_image(image, legend_labels=None):
    """在图片上方添加图例"""

    # fontsize = 300 # dpi 600 时用
    fontsize = 100
    legend_image_height = 300
    if not legend_labels:
        legend_labels = ["latency based","least loaded","sla elrar","weight based"]  # 如果没有图例标签，直接返回原图
        
    # 创建新图片，高度增加图例区域
    new_width, new_height = image.size[0], image.size[1] + legend_image_height
    print(f'width:{new_width}')
    print(f'height:{new_height}')
    new_image = Image.new('RGB', (new_width, new_height),color="white")
    new_legend_image = Image.new('RGB', (new_width, legend_image_height),color="white")
    
    # 绘制图例
    draw = ImageDraw.Draw(new_legend_image)

        # 尝试使用可调整大小的默认字体逻辑
    try:
        # 尝试获取系统中可用的字体
        from PIL import ImageFont
        # 使用系统默认字体并设置大小
        font = ImageFont.load_default()
        # 如果需要更大的字体，我们可以先获取默认字体的路径（如果可能）
        # 这里采用一个更可靠的方法：尝试加载系统中常见的等宽字体
        try:
            # 尝试Linux/Mac系统常见字体
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontsize)
        except (IOError, OSError):
            try:
                # 尝试其他可能的字体路径
                font = ImageFont.truetype("Arial.ttf", fontsize)
            except (IOError, OSError):
                # 如果都失败了，创建一个临时字体对象
                # 注意：这不会真正改变默认字体的大小，但可以避免错误
                print(f"警告: 无法加载指定大小的字体，使用默认字体大小")
                # 使用PIL的ImageFont.FreeTypeFont方法尝试创建字体
                try:
                    from PIL import ImageFont
                    # 这是一个尝试创建可调整大小字体的方法
                    font = ImageFont.truetype(font=font, size=fontsize)
                except:
                    # 如果所有尝试都失败，使用默认字体
                    font = ImageFont.load_default()
    except:
        # 如果出现任何错误，回退到默认字体
        font = ImageFont.load_default()

    # 计算每个标签的位置
    total_text_width = 0
    for label in legend_labels:
        text_width = draw.textlength(label, font=font)
        total_text_width += text_width + 800  # 每个标签之间增加间距
    # print(f'total_text_width:{total_text_width}')
    # 居中绘制所有标签
    start_x = (new_width - total_text_width) // 2 - 600 # 往左偏移500
    print(f'start_x:{start_x}')
    current_x = start_x 
    
    # 绘制图例标签
    for i, label in enumerate(legend_labels):
        # 绘制简单的彩色方块作为图例标识
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow']
        color = colors[i % len(colors)]
        
        # 设置图例大小和位置
        legend_height = int(fontsize * 1)
        legend_width = int(fontsize * 4)
        legend_center_y = 60 + legend_height // 2
        # print(f'legend_height:{legend_height},legend_width:{legend_width},legend_center_y:{legend_center_y}')
        
        # 绘制折线：两点一线，中间一个点或三角形
        # 左边点
        left_point = (current_x, legend_center_y)
        # 右边点
        right_point = (current_x + legend_width, legend_center_y)
        # 中间点
        mid_point = ((left_point[0] + right_point[0]) // 2, legend_center_y)
        
        # 绘制连接线
        if i == 2:
            # 实线
            draw.line([left_point, right_point], fill=color, width=10)
        else :
            # 虚线
            draw_dashed_line(draw, left_point, right_point, color, 10, 60, 60)
        # 根据索引选择不同的中间标记（点或三角形）
        marker_size = int(fontsize * 0.3)
        # 根据索引选择不同的中间标记（圆形点、正方形、三角形、菱形）
        marker_type = i % 4
        if marker_type == 0:
            # 绘制圆形点
            draw.ellipse([
                (mid_point[0] - marker_size, mid_point[1] - marker_size),
                (mid_point[0] + marker_size, mid_point[1] + marker_size)
            ], fill=color)
        elif marker_type == 1:
            # 绘制正方形
            draw.rectangle([
                (mid_point[0] - marker_size, mid_point[1] - marker_size),
                (mid_point[0] + marker_size, mid_point[1] + marker_size)
            ], fill=color)
        elif marker_type == 2:
            # 绘制三角形
            triangle_points = [
                (mid_point[0], mid_point[1] - marker_size),
                (mid_point[0] - marker_size, mid_point[1] + marker_size),
                (mid_point[0] + marker_size, mid_point[1] + marker_size)
            ]
            draw.polygon(triangle_points, fill=color)
        else:
            # 绘制菱形
            diamond_points = [
                (mid_point[0], mid_point[1] - marker_size),
                (mid_point[0] + marker_size, mid_point[1]),
                (mid_point[0], mid_point[1] + marker_size),
                (mid_point[0] - marker_size, mid_point[1])
            ]
            draw.polygon(diamond_points, fill=color)
        
        # 绘制标签文本，位置与图例对齐
        text_y = 60 + (legend_height - fontsize) // 2
        draw.text((current_x + legend_width + 15, text_y), label, font=font, fill='black')
        
        # 更新下一个标签的位置，增加间距
        current_x += legend_width + draw.textlength(label, font=font) + 1000
    
    # 将原始图片粘贴到图例下方
    new_image.paste(new_legend_image, (0, 0))
    new_image.paste(image, (0, legend_image_height))
    output_dir = "./heterogeneous_exp_comparison_paper.png"
    pdf_output_path = "./heterogeneous_exp_comparison_paper.pdf"
    new_image.save(output_dir)
    new_legend_image.save("./lengend.png")
    new_image.convert('RGB').save(pdf_output_path, "PDF", resolution=100.0)
    print(f"添加图例的照片保存在: {output_dir}")
    return new_image

def combine_images_horizontally(input_paths, output_path = None,dataset = None):
    """将多张图片横向合并为一张图片"""
    # 打开所有图片并确保它们是RGB模式
    images = []
    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"图片文件不存在: {path}")
        img = Image.open(path).convert('RGB')
        images.append(img)

    # 计算合并后图片的尺寸
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    # 创建新图片
    combined_img = Image.new('RGB', (total_width, max_height))

    # 粘贴每张图片
    x_offset = 0
    for img in images:
        combined_img.paste(img, (x_offset, 0))
        x_offset += img.width

        # 保存结果
    if not output_path:
        output_dir = "./picture/merge_horizontally"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{str(dataset)}_image_h.png"
    # 保存结果
    combined_img.save(output_path)
    print(f"横向合并完成，图片已保存至: {output_path}")


def combine_images_vertically(input_paths, output_path=None):
    """将多张图片纵向合并为一张图片，并可指定首先粘贴的图片
    
    参数:
        input_paths: 输入图片路径列表
        output_path: 输出路径，默认为None
        first_image: 首先粘贴的图片，默认为None
    """
    # 打开所有图片并确保它们是RGB模式，同时调整尺寸
    images = []
    image_paths_with_objects = []
    
    for path in input_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"图片文件不存在: {path}")
        img = Image.open(path).convert('RGB')
        image_paths_with_objects.append((path, img))
    
    # 计算合并后图片的尺寸
    widths, heights = zip(*(i[1].size for i in image_paths_with_objects))
    max_width = max(widths)
    total_height = sum(heights)
    
    # 创建新图片
    combined_img = Image.new('RGB', (max_width, total_height))
    
    # 粘贴图片：先粘贴指定的图片，再粘贴剩下的
    y_offset = 0
    
    first_image_path = "./picture/merge_horizontally/ShareGPT_image_h.png"
    # 首先处理指定的图片（如果有）
    if first_image_path:
        # 查找并粘贴指定的图片
        found = False
        for path, img in image_paths_with_objects[:]:
            if path == first_image_path:
                # 计算水平居中位置
                x_offset = (max_width - img.width) // 2
                combined_img.paste(img, (x_offset, y_offset))
                y_offset += img.height
                # 从列表中移除已粘贴的图片
                image_paths_with_objects.remove((path, img))
                found = True
                print(f"纵向合并开始,首先粘贴指定图片: {first_image_path}")
                break
        
        if not found:
            print(f"警告: 未找到指定的图片路径: {first_image_path}，将按照原顺序粘贴所有图片")
    
    # 粘贴剩下的图片
    for path, img in image_paths_with_objects:
        # 计算水平居中位置
        x_offset = (max_width - img.width) // 2
        combined_img.paste(img, (x_offset, y_offset))
        y_offset += img.height
    
    # 保存结果
    if not output_path:
        output_dir = f"./picture/merge_vertically"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/merge_vertically.png"
    
    combined_img.save(output_path)
    print(f"纵向合并完成，图片已保存至: {output_path}")
    return combined_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='将目录下所有图片合并为一张图片（横向或纵向）')
    parser.add_argument('--direction','-d', required=True, choices=['h', 'v'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    
    try:

        # 获取目录中的所有图片路径
        image_paths = get_image_paths_from_directory(args.input)
        dataset = os.path.basename(args.input)
        print(f"找到 {len(image_paths)} 张图片，准备{ '横向' if args.direction == 'h' else '纵向' }合并...")
        
        # 根据方向调用相应的合并函数
        if args.direction == 'h':
            combine_images_horizontally(image_paths, args.output,dataset)
        else:
            merged_img = combine_images_vertically(image_paths, args.output)

    except Exception as e:
        print(f"发生错误: {str(e)}")
        exit(1)
"""
    示例用法：
    python merge_picture.py -d h --input ./result/Flowgpt-timestamp
"""