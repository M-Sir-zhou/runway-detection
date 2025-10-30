import cv2
import numpy as np
import os
from typing import Tuple, List, Optional


class RunwayDetector:
    """跑道检测器，用于检测黑色跑道边缘并计算中线位置"""
    
    def __init__(self, lower_threshold: Tuple[int, int, int] = (0, 0, 0), 
                 upper_threshold: Tuple[int, int, int] = (50, 50, 50),
                 debug: bool = False):
        """
        初始化检测器
        
        Args:
            lower_threshold: HSV颜色空间下黑色跑道的下界
            upper_threshold: HSV颜色空间下黑色跑道的上界
            debug: 是否启用调试模式（显示采样点）
        """
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.debug = debug
        self.center_points = []
        self.raw_center_points = []  # 保存原始未平滑的中心点
    
    def _morphological_skeleton(self, image: np.ndarray) -> np.ndarray:
        """
        使用形态学操作提取骨架
        
        Args:
            image: 二值图像
            
        Returns:
            骨架图像
        """
        skeleton = np.zeros(image.shape, dtype=np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        temp = image.copy()
        
        while True:
            # 腐蚀
            eroded = cv2.erode(temp, element)
            # 开运算
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            # 从腐蚀结果中减去开运算结果
            subset = eroded - opened
            # 累加到骨架
            skeleton = cv2.bitwise_or(skeleton, subset)
            # 更新temp
            temp = eroded.copy()
            
            # 如果腐蚀后全为0，停止
            if cv2.countNonZero(temp) == 0:
                break
        
        return skeleton
        
    def detect_runway(self, image: np.ndarray) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        检测跑道边缘并计算中线
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            edges: 检测到的跑道边缘点列表
            centerline: 中线点 (x坐标)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 转换为HSV色彩空间用于检测暗色区域
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 使用更精确的HSV范围来检测黑色/深色跑道
        # HSV中黑色: V值较低（暗度）
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 75])  # 降低V值阈值从80到75，更严格
        
        # 创建掩码以检测暗色区域
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # 形态学操作，去除噪点并连接区域（使用最小核避免边缘扩张）
        kernel_tiny = np.ones((3, 3), np.uint8)
        
        # 只使用一次闭运算连接断开区域
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_tiny)
        # 只使用一次开运算去除小噪点  
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_tiny)
        
        # 不使用高斯模糊，避免边缘模糊
        # mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [], None
        
        # 按面积排序，找到最大的暗色区域（假设为跑道）
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # 筛选出较大的轮廓（至少占图像面积的5%）
        min_area = image.shape[0] * image.shape[1] * 0.05
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not valid_contours:
            return [], None
        
        # 获取最大的轮廓作为跑道
        runway_contour = valid_contours[0]
        
        # 不简化轮廓，保持原始精度
        # epsilon = 0.01 * cv2.arcLength(runway_contour, True)
        # simplified_contour = cv2.approxPolyDP(runway_contour, epsilon, True)
        
        edges_list = [runway_contour]  # 直接使用原始轮廓
        
        # 计算跑道的中线
        # 使用轮廓点对方法：直接从轮廓提取左右边界点对
        centerline = None
        
        # 创建跑道掩码
        runway_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(runway_mask, [runway_contour], -1, 255, -1)
        
        # 保存用于诊断
        self.debug_distance = None
        self.debug_skeleton = runway_mask.copy()
        
        # 方法：扫描每一行，从mask中找到最左和最右的点
        # 这是最直接、最准确的方法
        x_min, y_min, width, height = cv2.boundingRect(runway_contour)
        y_max = y_min + height
        
        center_points = []
        
        # 使用纯几何中点 - 数学上完美对称
        for y in range(y_min, y_max, 1):
            row = runway_mask[y, :]
            x_indices = np.where(row > 0)[0]
            
            if len(x_indices) >= 2:
                left_x = float(x_indices[0])
                right_x = float(x_indices[-1])
                
                # 纯几何中点
                center_x = (left_x + right_x) / 2.0
                center_points.append((center_x, float(y)))
        
        if len(center_points) < 10:
            print("  警告: 无法提取足够的中心线点")
            return edges_list, None
        
        print(f"  提取 {len(center_points)} 个中心点（逐行扫描）")
        
        # 保存原始中心点用于调试
        self.raw_center_points = center_points.copy()
        
        # 如果有足够的中心点，使用平滑处理
        if len(center_points) >= 10:
            center_y_coords = np.array([p[1] for p in center_points], dtype=float)
            center_x_coords = np.array([p[0] for p in center_points], dtype=float)
            
            # 移除重复的y坐标（保留每个y的平均x值）
            unique_y = np.unique(center_y_coords)
            if len(unique_y) < len(center_y_coords):
                new_x = []
                new_y = []
                for y_val in unique_y:
                    mask = center_y_coords == y_val
                    new_x.append(np.mean(center_x_coords[mask]))
                    new_y.append(y_val)
                center_x_coords = np.array(new_x)
                center_y_coords = np.array(new_y)
            
            # 极简平滑：最大限度保持原始几何中点的准确性
            from scipy.ndimage import uniform_filter1d, gaussian_filter1d
            from scipy.signal import medfilt, savgol_filter
            
            # 只做最基本的去噪，不改变几何位置
            # Savitzky-Golay滤波是最好的选择：去噪同时保持形状
            if len(center_x_coords) > 20:
                window = min(15, len(center_x_coords) // 8 * 2 + 1)  # 小窗口
                if window >= 5:
                    try:
                        center_x_coords = savgol_filter(center_x_coords, window, polyorder=3)
                    except:
                        pass
            
            # 使用参数化B样条拟合
            try:
                from scipy.interpolate import splprep, splev
                
                # 确保点按顺序排列
                sort_idx = np.argsort(center_y_coords)
                center_x_sorted = center_x_coords[sort_idx]
                center_y_sorted = center_y_coords[sort_idx]
                
                # 使用极小的平滑度，紧密跟随原始数据
                smooth_factor = len(center_points) * 0.5  # 最小平滑因子，几乎不偏离原始点
                
                # 参数化B样条拟合
                tck, u = splprep([center_x_sorted, center_y_sorted], 
                                s=smooth_factor, 
                                k=min(3, len(center_points)-1))
                
                # 生成密集的采样点
                u_fine = np.linspace(0, 1, 3000)
                x_fine, y_fine = splev(u_fine, tck)
                
                # 保存为查找表
                self.centerline_type = 'parametric_spline'
                self.param_spline = tck
                self.spline_x_lookup = x_fine
                self.spline_y_lookup = y_fine
                self.spline_y_range = (y_fine.min(), y_fine.max())
                
                centerline = 'parametric_spline'
                
            except Exception as e:
                print(f"  警告: 参数化样条拟合失败 ({e})，使用备用方法")
                # 备用方案：使用简单的y->x样条
                try:
                    from scipy.interpolate import UnivariateSpline
                    
                    sort_idx = np.argsort(center_y_coords)
                    center_y_sorted = center_y_coords[sort_idx]
                    center_x_sorted = center_x_coords[sort_idx]
                    
                    # 使用较小的平滑度
                    spline = UnivariateSpline(center_y_sorted, center_x_sorted, 
                                             s=len(center_points) * 5.0,  # 从20降到5
                                             k=min(3, len(center_points)-1))
                    
                    self.centerline_type = 'spline'
                    self.spline_func = spline
                    self.spline_y_range = (center_y_sorted.min(), center_y_sorted.max())
                    centerline = 'spline'
                    
                except Exception as e2:
                    print(f"  警告: 样条拟合也失败了 ({e2})，使用多项式")
                    # 最终备用：多项式拟合
                    sort_idx = np.argsort(center_y_coords)
                    center_y_sorted = center_y_coords[sort_idx]
                    center_x_sorted = center_x_coords[sort_idx]
                    
                    degree = min(5, len(center_points) - 1)
                    coeffs = np.polyfit(center_y_sorted, center_x_sorted, degree)
                    centerline = coeffs
                    self.centerline_type = 'poly'
        else:
            centerline = None
            self.centerline_type = None
        
        # 保存center_points用于调试可视化
        self.center_points = center_points
        
        return edges_list, centerline
    
    def visualize(self, image: np.ndarray, edges: List[np.ndarray], 
                  centerline: Optional[np.ndarray]) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image: 原始图像
            edges: 检测到的边缘
            centerline: 中线系数
            
        Returns:
            可视化后的图像
        """
        result = image.copy()
        
        # 绘制边缘
        cv2.drawContours(result, edges, -1, (0, 255, 0), 3)
        
        # 绘制中线
        if centerline is not None:
            height = image.shape[0]
            width = image.shape[1]
            points = []
            
            # 判断centerline类型
            if hasattr(self, 'centerline_type') and self.centerline_type == 'parametric_spline':
                # 使用参数化样条曲线 - 改进的插值方法
                if hasattr(self, 'spline_x_lookup') and hasattr(self, 'spline_y_lookup'):
                    # 对每个y坐标，使用二分查找或插值找到最佳x
                    y_lookup = self.spline_y_lookup
                    x_lookup = self.spline_x_lookup
                    
                    # 创建从y到索引的映射（使用插值）
                    from scipy.interpolate import interp1d
                    
                    # 确保y_lookup是单调的（如果不是，需要处理）
                    # 对于非单调情况，我们按y排序并取平均
                    y_min, y_max = y_lookup.min(), y_lookup.max()
                    
                    # 为每个图像y坐标找到对应的x
                    for y in range(0, height):
                        if y_min <= y <= y_max:
                            # 找到lookup表中最接近的点
                            distances = np.abs(y_lookup - y)
                            # 取最近的几个点的平均
                            closest_indices = np.argsort(distances)[:5]
                            x = np.mean(x_lookup[closest_indices])
                        elif y < y_min:
                            # 外推：使用起点
                            x = x_lookup[0]
                        else:
                            # 外推：使用终点
                            x = x_lookup[-1]
                        
                        x = max(0, min(int(round(x)), width - 1))
                        points.append((x, y))
                        
            elif hasattr(self, 'centerline_type') and self.centerline_type == 'spline':
                # 使用样条插值
                y_min, y_max = self.spline_y_range
                
                for y in range(0, height):
                    if y_min <= y <= y_max:
                        x = self.spline_func(y)
                    elif y < y_min:
                        x = self.spline_func(y_min)
                    else:
                        x = self.spline_func(y_max)
                    
                    x = max(0, min(int(round(x)), width - 1))
                    points.append((x, y))
                    
            else:
                # 使用多项式
                for y in range(0, height):
                    x = np.polyval(centerline, y)
                    x = max(0, min(int(round(x)), width - 1))
                    points.append((x, y))
            
            # 绘制中线（使用抗锯齿线条）
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i+1])
                    cv2.line(result, pt1, pt2, (255, 0, 0), 3)
            
            # 绘制采样点（调试模式）
            if self.debug and hasattr(self, 'center_points'):
                # 绘制原始采样点（黄色小圆点）
                if hasattr(self, 'raw_center_points'):
                    for i, point in enumerate(self.raw_center_points):
                        if i % 10 == 0:  # 每10个点绘制一个
                            cv2.circle(result, tuple(map(int, point)), 3, (0, 255, 255), -1)
                
                # 绘制拟合曲线上的点（紫色）
                if len(points) > 0:
                    for i, point in enumerate(points):
                        if i % 50 == 0:  # 每50个点绘制一个
                            cv2.circle(result, point, 2, (255, 0, 255), -1)
        
        return result


def process_image(input_path: str, output_path: str, save_diagnostics: bool = False) -> bool:
    """处理单张图像"""
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return False
    
    # 读取图像
    image = cv2.imread(input_path)
    if image is None:
        print(f"错误: 无法读取图像 {input_path}")
        return False
    
    print(f"处理图像: {input_path}")
    print(f"  图像尺寸: {image.shape}")
    
    # 创建检测器（设置debug=True可以看到采样点）
    detector = RunwayDetector(debug=False)
    
    # 检测跑道
    edges, centerline = detector.detect_runway(image)
    
    # 输出检测信息
    if edges:
        print(f"  [+] 检测到 {len(edges)} 个轮廓")
    else:
        print(f"  [-] 未检测到跑道轮廓")
    
    if centerline is not None:
        print(f"  [+] 成功计算出中线")
    else:
        print(f"  [-] 未能计算出中线")
    
    # 可视化结果
    result = detector.visualize(image, edges, centerline)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    
    # 保存诊断图像（如果需要）
    if save_diagnostics and hasattr(detector, 'debug_skeleton'):
        base_name = os.path.splitext(output_path)[0]
        
        # 保存骨架图
        if detector.debug_skeleton is not None:
            skeleton_colored = cv2.cvtColor(detector.debug_skeleton, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(f"{base_name}_skeleton.png", skeleton_colored)
            print(f"  [+] 骨架图已保存: {base_name}_skeleton.png")
        
        # 保存距离变换图（如果存在）
        if hasattr(detector, 'debug_distance') and detector.debug_distance is not None:
            dist_colored = cv2.normalize(detector.debug_distance, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            dist_colored = cv2.applyColorMap(dist_colored, cv2.COLORMAP_JET)
            cv2.imwrite(f"{base_name}_distance.png", dist_colored)
            print(f"  [+] 距离变换图已保存: {base_name}_distance.png")
    
    print(f"  [+] 结果已保存到: {output_path}")
    return True


def process_video(input_path: str, output_path: str) -> bool:
    """处理视频文件"""
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return False
    
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {input_path}")
        return False
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"处理视频: {input_path}")
    print(f"  视频尺寸: {width}x{height}, FPS: {fps}, 总帧数: {total_frames}")
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 创建检测器
    detector = RunwayDetector()
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测跑道
        edges, centerline = detector.detect_runway(frame)
        
        # 可视化结果
        result = detector.visualize(frame, edges, centerline)
        
        # 写入帧
        out.write(result)
        
        frame_count += 1
        if frame_count % 30 == 0 or frame_count == 1:
            print(f"  已处理帧: {frame_count}/{total_frames}")
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"  [+] 成功处理视频: {input_path} -> {output_path}")
    print(f"  共处理 {frame_count} 帧")
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python runway_detector.py <输入文件路径> <输出文件路径>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # 根据文件扩展名选择处理方式
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(input_path, output_path)
    elif ext in ['.mp4', '.avi', '.mov']:
        process_video(input_path, output_path)
    else:
        print(f"错误: 不支持的文件格式 {ext}")

