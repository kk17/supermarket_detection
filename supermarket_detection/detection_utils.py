import logging

def is_overlapping_1d(line1, line2):
    """line are in [min, max] format

    >>> is_overlapping_1d((3, 8),(2,5))
    True
    >>> is_overlapping_1d((3, 8),(9,10))
    False
    """
    min1, max1 = line1
    min2, max2 = line2
    return max1 >= min2 and max2 >= min1


def are_boxes_overlaping(box1, box2):
    """ Bboxes are in [y_min, x_min, y_max, x_max] format

    >>> are_boxes_overlaping((1, 1, 6, 6), (2, 2, 7, 7))
    True
    >>> are_boxes_overlaping((1, 1, 6, 6), (0, 2, 5, 7))
    True
    >>> are_boxes_overlaping((1, 1, 6, 6), (3, 7, 4, 12))
    False
    """
    tly1, tlx1, bry1, brx1 = box1
    tly2, tlx2, bry2, brx2 = box2
    return is_overlapping_1d((tlx1, brx1), (tlx2, brx2)) and is_overlapping_1d(
        (tly1, bry1), (tly2, bry2))


def merge_two_boxes(box1, box2):
    """ Bboxes are in [y_min, x_min, y_max, x_max] format

    >>> merge_two_boxes((1, 1, 6, 6), (2, 2, 7, 7))
    (1, 1, 7, 7)
    >>> merge_two_boxes((1, 1, 6, 6), (0, 2, 5, 7))
    (0, 1, 6, 7)
    """
    tly1, tlx1, bry1, brx1 = box1
    tly2, tlx2, bry2, brx2 = box2
    xmin, xmax = min(tlx1, tlx2), max(brx1, brx2)
    ymin, ymax = min(tly1, tly2), max(bry1, bry2)
    return (ymin, xmin, ymax, xmax)


def bb_intersection_over_union(box1, box2):
    """ Bboxes are in [y_min, x_min, y_max, x_max] format

    >>> bb_intersection_over_union((1, 1, 6, 6), (2, 2, 7, 7))
    0.47058823529411764
    >>> bb_intersection_over_union((1, 1, 6, 6), (0, 2, 5, 7))
    0.47058823529411764
    >>> bb_intersection_over_union((1, 1, 6, 6), (3, 7, 4, 12))
    0
    >>> bb_intersection_over_union((1, 1, 6, 6), (1, 1, 6, 6))
    1.0
    >>> bb_intersection_over_union((0, 0, 0.5, 0.5), (0, 0, 0.5, 0.5))
    1.0
    >>> bb_intersection_over_union((0, 0, 0.5, 0.5), (0.5, 0.5, 1, 1))
    0
    >>> bb_intersection_over_union((0, 0, 0.5, 0.5), (0.5, 0.5, 1, 1))
    0
    """
    ## box0: [0.50290835 0.00301954 0.93094087 0.503685  ] box1:[0.01598904 0.16670051 0.6182177  0.81539595] iou: 0.06863983311526284
    # determine the (x, y)-coordinates of the intersection rectangle
    tly1, tlx1, bry1, brx1 = box1
    tly2, tlx2, bry2, brx2 = box2
    h1, w1 = bry1 - tly1, brx1 - tlx1
    h2, w2 = bry2 - tly2, brx2 - tlx2
    xa, xb = max(tlx1, tlx2), min(brx1, brx2)
    ya, yb = max(tly1, tly2), min(bry1, bry2)
    # print(f'xa, xb:{xa}, {xb}')
    # print(f'ya, yb:{ya}, {yb}')

    # compute the area of intersection rectangle
    interArea = abs(max((xb - xa, 0)) * max((yb - ya), 0))
    # print(f'interArea:{interArea}')
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(w1 * h1 + w2 * h2 - interArea)

    # return the intersection over union value
    return iou


def merge_bounding_boxes(boxes, class_names, scores, merge_min_iou_thresh):
    """
    >>> boxes=[(1, 1, 6, 6), (0, 8, 5, 10), (0, 2, 7, 7)]
    >>> class_names=['a', 'b', 'c']
    >>> scores=[0.8, 0.7, 0.6]
    >>> merge_bounding_boxes(boxes, class_names, scores, 0.3)
    ([(0, 1, 7, 7), (0, 8, 5, 10)], ['a', 'b'], [0.8, 0.7])
    >>> merge_bounding_boxes(boxes, class_names, scores, 0.6)
    ([(1, 1, 6, 6), (0, 8, 5, 10), (0, 2, 7, 7)], ['a', 'b', 'c'], [0.8, 0.7, 0.6])
    """
    n = len(boxes)
    merged = [False] * n
    boxes = list(boxes)
    scores = list(scores)
    for i in range(n):
        if merged[i]:
            continue
        for j in range(n):
            if i == j or merged[j]:
                continue
            iou = bb_intersection_over_union(boxes[i], boxes[j])
            logging.debug(f'Class {class_names[0]} merge_min_iou_thresh {merge_min_iou_thresh}, box{i}: {boxes[i]} box{j}:{boxes[j]} iou: {iou}')
            if iou >= merge_min_iou_thresh:
                boxes[i] = merge_two_boxes(boxes[i], boxes[j])
                merged[j] = True
                scores[i] = max(scores[i], scores[j])

    _boxes = [boxes[i] for i in range(n) if not merged[i]]
    _class_names = [class_names[i] for i in range(n) if not merged[i]]
    _scores = [scores[i] for i in range(n) if not merged[i]]
    return _boxes, _class_names, _scores


if __name__ == "__main__":
    import doctest
    doctest.testmod()
