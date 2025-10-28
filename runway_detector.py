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
        
        # 使用更宽的HSV范围来检测黑色/深色跑道
        # HSV中黑色: V值较低（暗度）
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])  # V值不超过80表示较暗
        
        # 创建掩码以检测暗色区域
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # 形态学操作，去除噪点并连接区域
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((9, 9), np.uint8)
        kernel_large = np.ones((15, 15), np.uint8)
        
        # 先闭运算连接断开的区域
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        # 再开运算去除小噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_medium)
        
        # 高斯模糊减小噪声
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
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
        
        # 简化轮廓
        epsilon = 0.01 * cv2.arcLength(runway_contour, True)
        simplified_contour = cv2.approxPolyDP(runway_contour, epsilon, True)
        
        edges_list = [simplified_contour]
        
        # 计算跑道的中线
        # 方法：沿着跑道的弯曲方向采样，对每个采样点找到垂直方向的左右边界
        centerline = None
        
        # 获取轮廓的边界框
        x_min, y_min, width, height = cv2.boundingRect(runway_contour)
        x_max = x_min + width
        y_max = y_min + height
        
        # 第一步：先用y轴采样得到粗略的中心路径
        rough_center_points = []
        for y in range(max(0, y_min), min(image.shape[0], y_max + 1), 10):
            left_edge = None
            right_edge = None
            
            for x in range(max(0, x_min), min(image.shape[1], x_max + 1)):
                if cv2.pointPolygonTest(runway_contour, (x, y), True) >= 0:
                    left_edge = x
                    break
            
            for x in range(min(image.shape[1], x_max), max(0, x_min), -1):
                if cv2.pointPolygonTest(runway_contour, (x, y), True) >= 0:
                    right_edge = x
                    break
            
            if left_edge is not None and right_edge is not None:
                rough_center_x = (left_edge + right_edge) / 2
                rough_center_points.append((rough_center_x, y))
        
        if len(rough_center_points) < 3:
            return edges_list, None
        
        # 第二步：对粗略中心点进行平滑
        rough_y = np.array([p[1] for p in rough_center_points])
        rough_x = np.array([p[0] for p in rough_center_points])
        
        # 平滑粗略中心线
        from scipy.ndimage import gaussian_filter1d
        rough_x = gaussian_filter1d(rough_x, sigma=2.0)
        
        # 第三步：沿着粗略中心线，对每个点找垂直方向的左右边界
        center_points = []
        
        for i in range(len(rough_center_points) - 1):
            x1, y1 = rough_center_points[i]
            x2, y2 = rough_center_points[i + 1]
            
            # 计算切线方向（归一化）
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length < 1e-6:
                continue
            dx /= length
            dy /= length
            
            # 垂直方向（逆时针旋转90度）
            normal_x = -dy
            normal_y = dx
            
            # 从中心点沿垂直方向扫描左右边界
            max_search = 200  # 最大搜索距离
            
            left_dist = None
            right_dist = None
            
            # 向一个方向搜索边界
            for d in range(1, max_search):
                test_x = rough_x[i] + normal_x * d
                test_y = rough_y[i] + normal_y * d
                if not (0 <= test_x < image.shape[1] and 0 <= test_y < image.shape[0]):
                    break
                if cv2.pointPolygonTest(runway_contour, (test_x, test_y), True) < 0:
                    left_dist = d - 1
                    break
            
            # 向另一个方向搜索边界
            for d in range(1, max_search):
                test_x = rough_x[i] - normal_x * d
                test_y = rough_y[i] - normal_y * d
                if not (0 <= test_x < image.shape[1] and 0 <= test_y < image.shape[0]):
                    break
                if cv2.pointPolygonTest(runway_contour, (test_x, test_y), True) < 0:
                    right_dist = d - 1
                    break
            
            # 如果找到左右边界，计算真正的中点
            if left_dist is not None and right_dist is not None:
                # 计算左右边界的中点
                # left点位置: rough_x[i] + normal_x * left_dist
                # right点位置: rough_x[i] - normal_x * right_dist
                left_x = rough_x[i] + normal_x * left_dist
                left_y = rough_y[i] + normal_y * left_dist
                right_x = rough_x[i] - normal_x * right_dist
                right_y = rough_y[i] - normal_y * right_dist
                
                # 计算中点
                mid_x = (left_x + right_x) / 2
                mid_y = (left_y + right_y) / 2
                center_points.append((mid_x, mid_y))
        
        # 按照y坐标排序
        if center_points:
            center_points = sorted(center_points, key=lambda p: p[1])
        
        # 如果有足够的中心点，进行智能曲线拟合
        if len(center_points) >= 10:  # 至少需要10个点
            center_y_coords = np.array([p[1] for p in center_points])
            center_x_coords = np.array([p[0] for p in center_points])
            
            # 第一步：对原始采样点进行强度平滑（移动平均）
            from scipy.ndimage import uniform_filter1d, gaussian_filter1d
            
            # 使用较大的移动平均平滑采样点，窗口大小为7-11
            window_size = min(11, len(center_x_coords) // 2)
            if window_size >= 5:
                # 先用较大的窗口平滑
                center_x_coords = uniform_filter1d(center_x_coords, size=window_size, mode='nearest')
                # 再用高斯滤波进一步平滑
                center_x_coords = gaussian_filter1d(center_x_coords, sigma=1.5, mode='nearest')
            
            # 计算权重：近处的点权重更大
            y_normalized = (center_y_coords - center_y_coords.min()) / (center_y_coords.max() - center_y_coords.min() + 1e-6)
            weights = (1.0 - y_normalized) ** 2  # 上部权重更大
            weights = weights / weights.sum() * len(center_points)
            
            # 首先尝试使用分段样条插值（更自然）
            try:
                from scipy.interpolate import UnivariateSpline
                
                # 使用自适应平滑度的样条插值
                # s参数控制平滑度：越小越紧贴数据点，越大越平滑
                # 根据跑道形状自动调整
                # 计算数据点的方差来估计合理的平滑度
                x_variance = np.var(center_x_coords)
                # 使用方差的倍数作为平滑度参数
                smooth_factor = x_variance * 0.5  # 调整这个系数来控制平滑度
                
                # 尝试多个平滑度参数，选择最平滑但误差可接受的
                best_spline = None
                best_error = float('inf')
                
                # 增大平滑度搜索范围，优先选择更平滑的
                for s_factor in [1.0, 3.0, 5.0, 10.0, 20.0, 50.0]:
                    try:
                        test_spline = UnivariateSpline(center_y_coords, center_x_coords, s=x_variance * s_factor, k=min(3, len(center_points)-1))
                        predicted_x = test_spline(center_y_coords)
                        error = np.mean((center_x_coords - predicted_x) ** 2)
                        # 允许更大的误差以获得更平滑的效果
                        if error < x_variance * 10:  # 增大误差容忍度
                            if error < best_error:
                                best_error = error
                                best_spline = test_spline
                    except:
                        continue
                
                # 如果找到合适的spline就用它，否则用更大的平滑度
                if best_spline is not None:
                    spline = best_spline
                else:
                    # 使用更大的平滑度参数以减少抖动
                    spline = UnivariateSpline(center_y_coords, center_x_coords, s=x_variance * 10.0, k=min(3, len(center_points)-1))
                
                # 将样条保存为多项式分段形式（用于快速计算）
                self.centerline_type = 'spline'
                self.spline_func = spline
                self.spline_y_range = (center_y_coords.min(), center_y_coords.max())
                
                centerline = 'spline'  # 标记为样条类型
                
            except ImportError:
                # 如果没有scipy，回退到加权多项式拟合
                best_coeffs = None
                best_error = float('inf')
                best_degree = 2
                
                for degree in [3, 4, 5]:  # 提高最高阶数
                    try:
                        coeffs = np.polyfit(center_y_coords, center_x_coords, degree, w=weights)
                        predicted_x = np.polyval(coeffs, center_y_coords)
                        error = np.mean(weights * (center_x_coords - predicted_x) ** 2)
                        
                        if error < best_error:
                            best_error = error
                            best_coeffs = coeffs
                            best_degree = degree
                    except:
                        continue
                
                centerline = best_coeffs
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
            points = []
            
            # 判断centerline类型
            if hasattr(self, 'centerline_type') and self.centerline_type == 'spline':
                # 使用样条插值
                y_min, y_max = self.spline_y_range
                
                for y in range(0, height, 1):
                    if y_min <= y <= y_max:
                        x = self.spline_func(y)
                    elif y < y_min:
                        # 向上外推：使用第一个点的斜率
                        x = self.spline_func(y_min)
                    else:
                        # 向下外推：使用最后一个点的斜率
                        x = self.spline_func(y_max)
                    
                    x = max(0, min(int(x), image.shape[1] - 1))
                    points.append((x, y))
                    
            else:
                # 使用多项式
                for y in range(0, height, 1):
                    x = np.polyval(centerline, y)
                    x = max(0, min(int(x), image.shape[1] - 1))
                    points.append((x, y))
            
            # 绘制中线（使用抗锯齿线条）
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i+1])
                    cv2.line(result, pt1, pt2, (255, 0, 0), 3)
            
            # 绘制采样点（调试模式）
            if self.debug and hasattr(self, 'center_points'):
                # 绘制原始采样点（黄色）
                for point in self.center_points[::10]:  # 每10个点绘制一个
                    cv2.circle(result, tuple(map(int, point)), 4, (0, 255, 255), -1)
                # 绘制拟合曲线上的点（紫色）
                if len(points) > 0:
                    for point in points[::50]:  # 每50个点绘制一个
                        cv2.circle(result, point, 3, (255, 0, 255), -1)
        
        return result


def process_image(input_path: str, output_path: str) -> bool:
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

