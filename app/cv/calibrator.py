import math

SEGMENT_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def coords_to_segment_ring(x_mm: float, y_mm: float) -> tuple:
    """Convert board coordinates (mm, origin=centre) to (segment, ring, score)."""
    r = math.sqrt(x_mm ** 2 + y_mm ** 2)

    # Bull zones
    if r <= 6.35:
        return (25, "bullseye", 50)
    if r <= 15.9:
        return (25, "bull", 25)
    if r > 170:
        return (0, "miss", 0)

    # Ring by radius
    if 99 <= r <= 107:
        ring = "triple"
    elif 162 <= r <= 170:
        ring = "double"
    else:
        ring = "single"

    # Segment by angle: atan2 gives angle from +x axis; convert to clockwise from top
    angle_deg = math.degrees(math.atan2(y_mm, x_mm))
    board_angle = (90 - angle_deg) % 360
    seg_idx = int((board_angle + 9) / 18) % 20
    segment = SEGMENT_ORDER[seg_idx]

    if ring == "triple":
        score = segment * 3
    elif ring == "double":
        score = segment * 2
    else:
        score = segment

    return (segment, ring, score)
