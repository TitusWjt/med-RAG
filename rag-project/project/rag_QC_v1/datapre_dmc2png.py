import os
import pydicom
from PIL import Image
import numpy as np


def apply_window_level(image, window, level):
    lower = level - window / 2
    upper = level + window / 2
    return np.clip((image - lower) / (upper - lower) * 255.0, 0, 255).astype(np.uint8)


def convert_dicom_to_png(dicom_file_path, output_folder):
    try:
        # 读取DICOM文件
        ds = pydicom.dcmread(dicom_file_path)

        # 将DICOM像素数据转换为numpy数组
        image_array = ds.pixel_array.astype(float)

        # 尝试从DICOM标签中读取窗宽和窗位
        try:
            # 检查窗宽和窗位是否为MultiValue，如果是，则取第一个值
            if isinstance(ds.WindowWidth, pydicom.multival.MultiValue):
                window_width = float(ds.WindowWidth[0])
            else:
                window_width = float(ds.WindowWidth)

            if isinstance(ds.WindowCenter, pydicom.multival.MultiValue):
                window_level = float(ds.WindowCenter[0])
            else:
                window_level = float(ds.WindowCenter)

            # 应用窗宽和窗位
            image_array = apply_window_level(image_array, window_width, window_level)
        except AttributeError:
            # 如果DICOM文件中没有窗宽和窗位信息，则执行简单的归一化
            image_array = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255.0).astype(
                np.uint8)

        # 构建PNG文件的输出路径
        file_name = os.path.splitext(os.path.basename(dicom_file_path))[0] + '.png'
        png_file_path = os.path.join(output_folder, file_name)

        # 保存为PNG文件
        Image.fromarray(image_array).save(png_file_path)
        print(f'Converted {dicom_file_path} to {png_file_path}')
    except Exception as e:
        print(f"Error converting {dicom_file_path}: {e}")


def process_directory(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            convert_dicom_to_png(os.path.join(root, file), output_folder)


# 示例：将一个目录中的文件转换为PNG
input_folder = '/Users/titus.w/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/a6759ea3b6a5ef8471bdb3a7e3028adc/Message/MessageTemp/ef63bb39c2abf470d358d978bffc199e/File/锁骨质控/锁骨质控/锁骨质控/锁骨未包全'  # 文件所在的输入目录路径
output_folder = '/Users/titus.w/Downloads/X-ray/锁骨质控/锁骨未包全'  # 输出PNG文件的文件夹路径

process_directory(input_folder, output_folder)
