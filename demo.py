"""
跑道检测演示脚本


自动处理input文件夹下的所有图像和视频文件
"""
import os
import glob
from runway_detector import process_image, process_video


def main():
    """主函数：处理input文件夹下的所有文件"""
    
    # 创建输出目录
    os.makedirs("output", exist_ok=True)
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # 获取所有输入文件
    input_files = []
    
    # 查找图像文件
    for ext in image_extensions:
        input_files.extend(glob.glob(f"input/*{ext}"))
        input_files.extend(glob.glob(f"input/*{ext.upper()}"))
    
    # 查找视频文件
    for ext in video_extensions:
        input_files.extend(glob.glob(f"input/*{ext}"))
        input_files.extend(glob.glob(f"input/*{ext.upper()}"))
    
    if not input_files:
        print("错误: input文件夹中没有找到任何文件")
        return
    
    print(f"找到 {len(input_files)} 个文件需要处理\n")
    
    # 处理每个文件
    for input_path in input_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join("output", f"result_{filename}")
        
        ext = os.path.splitext(input_path)[1].lower()
        
        print(f"正在处理: {filename}")
        
        if ext in image_extensions:
            process_image(input_path, output_path)
        elif ext in video_extensions:
            process_video(input_path, output_path)
        else:
            print(f"跳过不支持的文件: {filename}")
        
        print()
    
    print("所有文件处理完成！")


if __name__ == "__main__":
    main()


