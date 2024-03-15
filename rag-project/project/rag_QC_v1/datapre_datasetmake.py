import os

# 图片目录和输出目录路径，这里需要根据你的实际情况进行修改
image_directory = '/Users/titus.w/Downloads/医学影像质控数据集(图文对)/脚掌未包全'
output_directory = '/Users/titus.w/Downloads/医学影像质控数据集(图文对)/tmp'

# 确保输出目录存在
os.makedirs(output_directory, exist_ok=True)

# 遍历图片目录中的所有文件
for image_name in os.listdir(image_directory):
    if image_name.lower().endswith('.png'):
        # 创建单独的文件夹
        folder_name = "脚掌未包全_" + os.path.splitext(image_name)[0]
        folder_path = os.path.join(output_directory, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # 移动图片到新的文件夹
        src_image_path = os.path.join(image_directory, image_name)
        dst_image_path = os.path.join(folder_path, image_name)
        os.rename(src_image_path, dst_image_path)

        # 创建txt文件
        txt_file_name = '脚掌未包全.txt'
        txt_content = """脚掌未包全
可能原因:
摆位失误：患者的足部摆位可能未按照协议要求，导致脚掌的部分区域未能出现在X光影像上。
影像采集范围过小：如果影像采集范围设定得过小，可能无法覆盖整个脚掌。
患者不配合：患者可能因为不适或理解问题，未能保持必要的静止状态。
解决措施:
重新评估并调整摆位协议，确保脚掌的所有部位都能被包含在采集范围内。
在拍摄前确保患者理解所需的摆位，并在必要时使用固定装置帮助患者保持不动。
提高技术人员对于不同体位拍摄技术的熟练度，确保能够根据患者实际情况灵活调整采集范围。
"""

        with open(os.path.join(folder_path, txt_file_name), 'w', encoding='utf-8') as txt_file:
            txt_file.write(txt_content)
        print(f'Folder and txt file created for image: {image_name}')
