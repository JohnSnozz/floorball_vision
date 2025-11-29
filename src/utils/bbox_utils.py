"""Bounding Box Utilities for Floorball Vision.

Common bounding box operations and conversions.
"""

from typing import List, Tuple

import numpy as np


def xyxy_to_xywh(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from (x1, y1, x2, y2) to (x, y, width, height).

    Args:
        bbox: Bounding box as (x1, y1, x2, y2)

    Returns:
        Bounding box as (x, y, width, height)
    """
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2 - x1, y2 - y1)


def xywh_to_xyxy(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from (x, y, width, height) to (x1, y1, x2, y2).

    Args:
        bbox: Bounding box as (x, y, width, height)

    Returns:
        Bounding box as (x1, y1, x2, y2)
    """
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def yolo_to_xyxy(
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format (center_x, center_y, width, height) normalized to pixel coordinates.

    Args:
        bbox: YOLO format bbox (all values 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Bounding box as (x1, y1, x2, y2) in pixels
    """
    cx, cy, w, h = bbox

    x1 = int((cx - w / 2) * img_width)
    y1 = int((cy - h / 2) * img_height)
    x2 = int((cx + w / 2) * img_width)
    y2 = int((cy + h / 2) * img_height)

    return (x1, y1, x2, y2)


def xyxy_to_yolo(
    bbox: Tuple[int, int, int, int],
    img_width: int,
    img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert pixel coordinates to YOLO format (center_x, center_y, width, height) normalized.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2) in pixels
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        YOLO format bbox (all values 0-1)
    """
    x1, y1, x2, y2 = bbox

    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    return (cx, cy, w, h)


def get_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Get center point of bounding box (x1, y1, x2, y2).

    Args:
        bbox: Bounding box as (x1, y1, x2, y2)

    Returns:
        Center point as (x, y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """
    Calculate area of bounding box (x1, y1, x2, y2).

    Args:
        bbox: Bounding box as (x1, y1, x2, y2)

    Returns:
        Area in square pixels
    """
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bounding box as (x1, y1, x2, y2)
        bbox2: Second bounding box as (x1, y1, x2, y2)

    Returns:
        IoU value (0-1)
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = get_bbox_area(bbox1)
    area2 = get_bbox_area(bbox2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def scale_bbox(
    bbox: Tuple[float, float, float, float],
    scale_x: float,
    scale_y: float
) -> Tuple[float, float, float, float]:
    """
    Scale a bounding box.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        scale_x: Scale factor for x coordinates
        scale_y: Scale factor for y coordinates

    Returns:
        Scaled bounding box
    """
    x1, y1, x2, y2 = bbox
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
