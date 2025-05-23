import cv2

def draw_bbox(
    img,
    bbox,
    bbox_color,
    text=None,
    text_color=None,
    text_bg_color=(128, 128, 128),
    thickness=2,
):
    """
    draw bbox and text on image
    """
    x1, y1, w, h = [int(i) for i in bbox]
    x2, y2 = x1 + w, y1 + h
    # For bounding box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness)

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    if text:
        font_scale = 2
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

        # Prints the text.
        img = cv2.rectangle(img, (x1, y1 - h - 5), (x1 + w, y1), text_bg_color, -1)
        img = cv2.putText(
            img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2
        )

    return img