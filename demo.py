"""
跑道检测演示脚本

自动处理input文件夹下的所有图像和视频文件，并计算无人机偏离距离
"""
import os
import glob
import argparse
from runway_detector import process_image, process_video


def main():
    """主函数：处理input文件夹下的所有文件"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='跑道检测与无人机偏离分析')
    parser.add_argument('--pixels-per-meter', type=float, default=100.0,
                       help='像素到米的转换比例（默认：100像素=1米）')
    parser.add_argument('--drone-x', type=int, default=None,
                       help='无人机X坐标（默认：图像中心）')
    parser.add_argument('--drone-y', type=int, default=None,
                       help='无人机Y坐标（默认：图像中心）')
    parser.add_argument('--log-offsets', action='store_true',
                       help='记录视频处理时每帧的偏离数据到CSV文件')
    parser.add_argument('--input', type=str, default='input',
                       help='输入文件夹路径（默认：input）')
    parser.add_argument('--output', type=str, default='output',
                       help='输出文件夹路径（默认：output）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("跑道检测与无人机偏离分析系统")
    print("=" * 60)
    print(f"配置参数:")
    print(f"  像素/米比例: {args.pixels_per_meter:.1f} px/m")
    print(f"  无人机位置: {'图像中心' if args.drone_x is None else f'({args.drone_x}, {args.drone_y})'}")
    print(f"  记录偏离日志: {'是' if args.log_offsets else '否'}")
    print("=" * 60)
    print()
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    # 支持的视频格式
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # 获取所有输入文件
    input_files = []
    
    # 查找图像文件
    for ext in image_extensions:
        input_files.extend(glob.glob(f"{args.input}/*{ext}"))
        input_files.extend(glob.glob(f"{args.input}/*{ext.upper()}"))
    
    # 查找视频文件
    for ext in video_extensions:
        input_files.extend(glob.glob(f"{args.input}/*{ext}"))
        input_files.extend(glob.glob(f"{args.input}/*{ext.upper()}"))
    
    # 去除重复文件（同一文件可能被小写和大写扩展名都匹配到）
    input_files = list(set(input_files))
    
    if not input_files:
        print(f"错误: {args.input}文件夹中没有找到任何文件")
        return
    
    print(f"找到 {len(input_files)} 个文件需要处理\n")
    
    # 处理每个文件
    for input_path in input_files:
        filename = os.path.basename(input_path)
        output_path = os.path.join(args.output, f"result_{filename}")
        
        ext = os.path.splitext(input_path)[1].lower()
        
        print(f"正在处理: {filename}")
        
        if ext in image_extensions:
            process_image(input_path, output_path, 
                        pixels_per_meter=args.pixels_per_meter,
                        drone_x=args.drone_x, 
                        drone_y=args.drone_y)
        elif ext in video_extensions:
            process_video(input_path, output_path,
                        pixels_per_meter=args.pixels_per_meter,
                        drone_x=args.drone_x,
                        drone_y=args.drone_y,
                        log_offsets=args.log_offsets)
        else:
            print(f"跳过不支持的文件: {filename}")
        
        print()
    
    print("=" * 60)
    print("所有文件处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()


