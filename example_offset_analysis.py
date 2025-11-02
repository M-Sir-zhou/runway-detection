# -*- coding: utf-8 -*-
"""
Drone Offset Analysis Example

Demonstrates how to use the runway detection system to calculate drone offset from centerline
"""
import cv2
import numpy as np
from runway_detector import RunwayDetector


def analyze_single_image():
    """Analyze drone offset in a single image"""
    print("=" * 60)
    print("Example 1: Single Image Offset Analysis")
    print("=" * 60)
    
    # Create detector (120 pixels = 1 meter)
    detector = RunwayDetector(pixels_per_meter=120.0)
    
    # Read image
    image = cv2.imread('input/test_image_01.jpg')
    if image is None:
        print("Error: Cannot read image")
        return
    
    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")
    
    # Detect runway
    print("Detecting runway...")
    edges_list, centerline = detector.detect_runway(image)
    
    if centerline is None:
        print("Error: Centerline not detected")
        return
    
    # Assume drone at image center
    drone_x = width // 2
    drone_y = height // 2
    
    # Calculate offset
    offset_pixels, offset_meters = detector.calculate_drone_offset(
        image, drone_x, drone_y
    )
    
    # Output results
    direction = "right" if offset_pixels > 0 else "left" if offset_pixels < 0 else "on centerline"
    print(f"\nResults:")
    print(f"  Drone position: ({drone_x}, {drone_y})")
    print(f"  Offset distance: {abs(offset_pixels):.1f} pixels")
    print(f"  Offset distance: {abs(offset_meters):.3f} meters")
    print(f"  Offset direction: {direction}")
    
    # Visualize and save
    result = detector.visualize(image, edges_list, centerline,
                               show_drone_offset=True,
                               drone_x=drone_x, drone_y=drone_y)
    cv2.imwrite('output/example_offset_analysis.jpg', result)
    print(f"\nResult saved to: output/example_offset_analysis.jpg")


def analyze_multiple_positions():
    """Analyze offsets at multiple positions (simulating drone movement)"""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Position Offset Analysis")
    print("=" * 60)
    
    # Create detector
    detector = RunwayDetector(pixels_per_meter=100.0)
    
    # Read image
    image = cv2.imread('input/test_image_01.jpg')
    if image is None:
        print("Error: Cannot read image")
        return
    
    height, width = image.shape[:2]
    
    # Detect runway
    print("Detecting runway...")
    edges_list, centerline = detector.detect_runway(image)
    
    if centerline is None:
        print("Error: Centerline not detected")
        return
    
    # Simulate drone at different positions
    positions = [
        (width // 4, height // 2, "Left position"),
        (width // 2, height // 2, "Center position"),
        (3 * width // 4, height // 2, "Right position"),
        (width // 2, height // 4, "Top position"),
        (width // 2, 3 * height // 4, "Bottom position"),
    ]
    
    print("\nPosition Analysis:")
    print("-" * 60)
    
    for drone_x, drone_y, desc in positions:
        offset_pixels, offset_meters = detector.calculate_drone_offset(
            image, drone_x, drone_y
        )
        direction = "R" if offset_pixels > 0 else "L" if offset_pixels < 0 else "C"
        
        print(f"{desc:18s} | ({drone_x:4d},{drone_y:4d}) | "
              f"{abs(offset_meters):6.3f}m ({direction})")


if __name__ == "__main__":
    # Run examples
    analyze_single_image()
    analyze_multiple_positions()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
