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

def get_image_paths(dir = None,order_list = None):
    """从目录中获取所有图片文件路径并按名称排序"""

    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

    if not os.path.isdir(dir):
        raise NotADirectoryError(f"错误：{dir} 不是有效的目录")
    
    image_paths = []
    all_files = sorted(os.listdir(dir))

    if order_list:
        # 按指定顺序读取
        matched = set()
        for order in order_list:
            for filename in all_files:
                if f'{order}' in filename and filename.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(dir, filename))
                    all_files.remove(filename)
                    matched.add(filename)

        # # 添加未匹配的剩余文件（保持原排序）
        # for filename in all_files:
        #     if filename not in matched and filename.lower().endswith(image_extensions):
        #         image_paths.append(os.path.join(dir, filename))
    else:
        """按原顺序读取"""
        for filename in all_files:
            if filename.lower().endswith(image_extensions):
                image_paths.append(os.path.join(dir, filename))
    
    if not image_paths:
        raise ValueError(f"在目录 {dir} 中未找到任何所需的图片文件")

    return image_paths

# 添加新函数：在图片正上方中央添加文字标题
def add_title_to_image(image,config,title_text = "",output=None):
    """在图片正上方中央添加文字标题"""

    # 创建新图片，高度增加标题区域
    new_width, new_height = image.width, image.height + config['title_height']
    new_image = Image.new('RGB', (new_width, new_height), color='white')
    title_image = Image.new('RGB', (new_width, config['title_height']), color='white')
    
    # 绘制标题
    draw = ImageDraw.Draw(title_image)
    
    # 尝试加载合适的字体
    try:
        from PIL import ImageFont
        # 尝试Linux/Mac系统常见字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", config['fontsize'])
    except (IOError, OSError):
        try:
            # 尝试其他可能的字体路径
            font = ImageFont.truetype("Arial.ttf", config['fontsize'])
            print("使用字体: Arial.ttf")
        except (IOError, OSError):
            print(f"使用默认字体大小")
            font = ImageFont.load_default()
    
    # 计算文本居中位置
    text_width = draw.textlength(title_text, font=font)
    text_x = (new_width - text_width) * config['position']
    text_y = (config['title_height'] - config['fontsize']) // 2  # 简化处理
    
    # 绘制文本
    draw.text((text_x, text_y), title_text, font=font, fill='black')
    
    # 将标题和原图合并
    new_image.paste(title_image, (0, 0))
    new_image.paste(image, (0, config['title_height']))

    # 保存图片
    if not output:

        # 获取当前文件的完整路径
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        output = f"{current_dir}/picture/process/add_title_to_image.png"

    new_image.save(output)
    print(f"添加顶部标题的照片保存在: {output}")
    
    return new_image
def add_title_to_image_bottom(image,config,title_text = "",output=None):
    """在图片正下方添加文字标题"""

    # 创建新图片，高度增加标题区域
    new_width, new_height = image.width, image.height + config['title_height']
    new_image = Image.new('RGB', (new_width, new_height), color='white')
    title_image = Image.new('RGB', (new_width, config['title_height']), color='white')
    
    # 绘制标题
    draw = ImageDraw.Draw(title_image)
    
    # 尝试加载合适的字体
    try:
        from PIL import ImageFont
        # 尝试Linux/Mac系统常见字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", config['fontsize'])
    except (IOError, OSError):
        try:
            # 尝试其他可能的字体路径
            font = ImageFont.truetype("Arial.ttf", config['fontsize'])
            print("使用字体: Arial.ttf")
        except (IOError, OSError):
            print(f"使用默认字体大小")
            font = ImageFont.load_default()
    
    # 计算文本居中位置
    text_width = draw.textlength(title_text, font=font)
    text_x = (new_width - text_width) * config['position'] # 文字位置
    text_y = (config['title_height'] - config['fontsize']) // 2  # 简化处理
    
    # 绘制文本
    draw.text((text_x, text_y), title_text, font=font, fill='black')
    
    # 将标题和原图合并
    new_image.paste(image, (0, 0))
    new_image.paste(title_image, (0, image.height))

    # 保存图片
    if not output:
        
        # 获取当前文件的完整路径
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)

        output = f"{current_dir}/picture/process/add_title_to_image_bottom.png"

    new_image.save(output)
    print(f"添加底部标题的照片保存在: {output}")
    
    return new_image
    
def add_side_title_to_image(image,config, title_text1="Title 1", title_text2="Title 2", output = None):
    """在图片左侧添加两列从下往上垂直排列的文字标题"""
    

    # 创建新图片，宽度增加标题区域
    new_width, new_height = image.width + config['total_title_width'], image.height
    new_image = Image.new('RGB', (new_width, new_height), color='white')
    
    # 计算每列标题的宽度（均分总宽度）
    column_width = config['total_title_width'] // 2
    
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
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", config['fontsize'])
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("Arial.ttf", config['fontsize'])
        except (IOError, OSError):
            print("无法加载指定大小的字体，使用默认字体大小")
            font = ImageFont.load_default()
    
    # 绘制第一列标题
    draw1 = ImageDraw.Draw(temp_image1)
    text_width1 = draw1.textlength(title_text1, font=font)
    text_x1 = (temp_width1 - text_width1) // 2  # 水平居中
    text_y1 = (temp_height1 - config['fontsize']) * config['position']
    draw1.text((text_x1, text_y1), title_text1, font=font, fill='black')
    
    # 绘制第二列标题
    draw2 = ImageDraw.Draw(temp_image2)
    text_width2 = draw2.textlength(title_text2, font=font)
    text_x2 = (temp_width2 - text_width2) // 2  # 水平居中
    text_y2 = (temp_height2 - config['fontsize']) * config['position']
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
    new_image.paste(image, (config['total_title_width'], 0))
    

    # 保存图片
    if not output:
        
        # 获取当前文件的完整路径
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        output = f"{current_dir}/picture/process/add_side_title_to_image.png"

    new_image.save(output)
    print(f"添加左侧标题的照片保存在: {output}")

    return new_image
    
def add_legend_to_image(image,config):
    """在图片上方添加图例"""

    if not config.get("legend_labels"):
        print("legend labels is None")
        return
    
    if not config.get("colors"):
        print("color is None")
        return
        
    # 创建新图片，高度增加图例区域
    new_width, new_height = image.size[0], image.size[1] + config['legend_total_height']
    new_image = Image.new('RGB', (new_width, new_height),color="white")
    new_legend_image = Image.new('RGB', (new_width, config['legend_total_height']),color="white")
    
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
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", config['fontsize'])
        except (IOError, OSError):
            try:
                # 尝试其他可能的字体路径
                font = ImageFont.truetype("Arial.ttf", config['fontsize'])
            except (IOError, OSError):
                # 如果都失败了，创建一个临时字体对象
                # 注意：这不会真正改变默认字体的大小，但可以避免错误
                print(f"警告: 无法加载指定大小的字体，使用默认字体大小")
                # 使用PIL的ImageFont.FreeTypeFont方法尝试创建字体
                try:
                    from PIL import ImageFont
                    # 这是一个尝试创建可调整大小字体的方法
                    font = ImageFont.truetype(font=font, size=config['fontsize'])
                except:
                    # 如果所有尝试都失败，使用默认字体
                    font = ImageFont.load_default()
    except:
        # 如果出现任何错误，回退到默认字体
        font = ImageFont.load_default()

    # 计算每个标签的位置
    total_text_width = 0
    for label in config['legend_labels']:
        text_width = draw.textlength(label, font=font)
        total_text_width += text_width  # 每个标签之间增加间距

    legend_metrics = config['legend_metrics']

    
    # 设置图例参数和位置

    legend_height = int(config['fontsize'] * legend_metrics['legend_height'])
    legend_length = int(config['fontsize'] * legend_metrics['legend_length_ratoi'])
    legend_center_y = legend_height // 2
    legend_interval = legend_length * legend_metrics['legend_interval_ratoi']
    # 居中绘制所有标签
    start_x = (new_width - total_text_width - (legend_length + legend_interval + 20)*len(config['legend_labels']) ) // 2

    current_x = start_x +500

    # 开始绘制图例标签
    for i, label in enumerate(config['legend_labels']):
        # 绘制简单的彩色方块作为图例标识

        color = config['colors'][i % len(config['colors'])]
        

        
        # 绘制折线：两点一线，中间一个点或三角形
        # 左边点
        left_point = (current_x, legend_center_y)
        # 右边点
        right_point = (current_x + legend_length, legend_center_y)
        # 中间点
        mid_point = ((left_point[0] + right_point[0]) // 2, legend_center_y)
        
        # 绘制连接线
        if color == "red":
            draw.line([left_point, right_point], fill=color, width=legend_metrics['line_width'])
        else :
            draw_dashed_line(draw, left_point, right_point, color, legend_metrics['line_width'],
                            legend_metrics['dotted_line_width'], legend_metrics['dotted_line_width'])

        # 根据索引选择不同的中间标记（点或三角形）
        marker_size = int(config['fontsize'] * legend_metrics['marker_size'])
        # 根据索引选择不同的中间标记（圆形点、正方形、三角形、菱形）
        marker_type = i % 8
        if marker_type == 0:
            # 绘制三角形
            triangle_points = [
                (mid_point[0], mid_point[1] - marker_size),
                (mid_point[0] - marker_size, mid_point[1] + marker_size),
                (mid_point[0] + marker_size, mid_point[1] + marker_size)
            ]
            draw.polygon(triangle_points, fill=color)
        elif marker_type == 1:
            # 绘制圆形点
            draw.ellipse([
                (mid_point[0] - marker_size, mid_point[1] - marker_size),
                (mid_point[0] + marker_size, mid_point[1] + marker_size)
            ], fill=color)
        elif marker_type == 2:
            # 绘制正方形
            draw.rectangle([
                (mid_point[0] - marker_size, mid_point[1] - marker_size),
                (mid_point[0] + marker_size, mid_point[1] + marker_size)
            ], fill=color)
        elif marker_type == 3:
            # 绘制菱形
            diamond_points = [
                (mid_point[0], mid_point[1] - marker_size),
                (mid_point[0] + marker_size, mid_point[1]),
                (mid_point[0], mid_point[1] + marker_size),
                (mid_point[0] - marker_size, mid_point[1])
            ]
            draw.polygon(diamond_points, fill=color)
        elif marker_type == 4:
            # 绘制星号（*）
            star_points = [
                (mid_point[0], mid_point[1]-marker_size),
                (mid_point[0]+marker_size*0.3, mid_point[1]-marker_size*0.3),
                (mid_point[0]+marker_size, mid_point[1]),
                (mid_point[0]+marker_size*0.3, mid_point[1]+marker_size*0.3),
                (mid_point[0], mid_point[1]+marker_size),
                (mid_point[0]-marker_size*0.3, mid_point[1]+marker_size*0.3),
                (mid_point[0]-marker_size, mid_point[1]),
                (mid_point[0]-marker_size*0.3, mid_point[1]-marker_size*0.3)
            ]
            draw.polygon(star_points, fill=color)
        elif marker_type == 5:
            # 绘制五边形（p）
            pentagon_points = [
                (mid_point[0], mid_point[1]-marker_size),
                (mid_point[0]+marker_size*0.95, mid_point[1]-marker_size*0.3),
                (mid_point[0]+marker_size*0.6, mid_point[1]+marker_size*0.8),
                (mid_point[0]-marker_size*0.6, mid_point[1]+marker_size*0.8),
                (mid_point[0]-marker_size*0.95, mid_point[1]-marker_size*0.3)
            ]
            draw.polygon(pentagon_points, fill=color)
        elif marker_type == 6:
            draw.polygon(star_points, fill=color)
            # 绘制叉号（x）
            draw.line([
                (mid_point[0]-marker_size, mid_point[1]-marker_size),
                (mid_point[0]+marker_size, mid_point[1]+marker_size)
            ], fill=color, width=15)
            draw.line([
                (mid_point[0]+marker_size, mid_point[1]-marker_size),
                (mid_point[0]-marker_size, mid_point[1]+marker_size)
            ], fill=color, width=15)
        elif marker_type == 7:
            # 绘制六边形（h）
            hexagon_points = [
                (mid_point[0], mid_point[1]-marker_size),
                (mid_point[0]+marker_size*0.87, mid_point[1]-marker_size*0.5),
                (mid_point[0]+marker_size*0.87, mid_point[1]+marker_size*0.5),
                (mid_point[0], mid_point[1]+marker_size),
                (mid_point[0]-marker_size*0.87, mid_point[1]+marker_size*0.5),
                (mid_point[0]-marker_size*0.87, mid_point[1]-marker_size*0.5)
            ]
            draw.polygon(hexagon_points, fill=color)
        
        # 绘制标签文本，位置与图例对齐
        text_y = (legend_height - config['fontsize']) // 2
        # 20 是图例与文字间距
        draw.text((current_x + legend_length +20, text_y), label, font=font, fill='black')
        
        # 更新下一个标签的位置，增加间距
        current_x += legend_length + draw.textlength(label, font=font) + legend_interval
    
    # 将原始图片粘贴到图例下方
    new_image.paste(new_legend_image, (0, 0))
    new_image.paste(image, (0, config['legend_total_height']))

    # 保存图片
    new_image.save(f"{config['output_dir']}.png")
    new_image.convert('RGB').save(f"{config['output_dir']}.pdf", "PDF", resolution=100.0)
    print(f"包含图例的照片保存在: {config['output_dir']}")
    return new_image

def combine_images_horizontally(dir = None, output_path = None,order_list = None):
    """将多张图片横向合并为一张图片"""

    images_path = get_image_paths(dir,order_list)
    images = []
    for path in images_path:
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
        
        # 获取当前文件的完整路径
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)

        output_dir = f"{current_dir}/picture/combine_horizontally"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/{str(os.path.basename(dir))}_image_combine_h.png"

    # 保存结果
    combined_img.save(output_path)
    print(f"{dir} 下图片 横向 合并完成，图片保存至: {output_path}")

    return combined_img

def combine_images_vertically(dir = None, output_path = None,order_list = None):
    """将目录中多张图片纵向合并为一张图片，指定顺序"""

    # 打开所有图片并确保它们是RGB模式，同时调整尺寸

    images = []
    
    images_path = get_image_paths(dir,order_list)
    for path in images_path:
        img = Image.open(path).convert('RGB')
        images.append(img)
    
    # 计算合并后图片的尺寸
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)
    
    # 创建新图片
    combined_img = Image.new('RGB', (max_width, total_height))
    
    # 粘贴图片
    y_offset = 0
    
    for img in images:
        # 计算水平居中位置，有些图片偏窄
        x_offset = (max_width - img.width) // 2
        combined_img.paste(img, (x_offset, y_offset))
        y_offset += img.height
    
    # 保存结果
    if not output_path:
        # 获取当前文件的完整路径
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)

        output_dir = f"{current_dir}/picture/combine_vertically"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/image_combine_v.png"
    
    combined_img.save(output_path)
    print(f"{dir} 下图片 纵向 合并完成，图片保存至: {output_path}")
    return combined_img