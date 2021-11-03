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
    """ Bboxes are in [top-left-x, top-left-y, width, height] format

    >>> are_boxes_overlaping((1, 1, 5, 5), (2, 2, 5, 5))
    True
    >>> are_boxes_overlaping((1, 1, 5, 5), (0, 2, 5, 5))
    True
    >>> are_boxes_overlaping((1, 1, 5, 5), (3, 7, 1, 5))
    False
    """
    # Bboxes are in [top-left-x, top-left-y, width, height] format
    tlx1, tly1, w1, h1 = box1
    tlx2, tly2, w2, h2 = box2
    brx1, bry1 = tlx1 + w1, tly1 + h1
    brx2, bry2 = tlx2 + w2, tly2 + h2
    return is_overlapping_1d((tlx1, brx1), (tlx2, brx2)) and is_overlapping_1d(
        (tly1, bry1), (tly2, bry2))


def merge_two_boxes(box1, box2):
    """ Bboxes are in [top-left-x, top-left-y, width, height] format

    >>> merge_two_boxes((1, 1, 5, 5), (2, 2, 5, 5))
    (1, 1, 6, 6)
    >>> merge_two_boxes((1, 1, 5, 5), (0, 2, 5, 5))
    (0, 1, 6, 6)
    """
    tlx1, tly1, w1, h1 = box1
    tlx2, tly2, w2, h2 = box2
    brx1, bry1 = tlx1 + w1, tly1 + h1
    brx2, bry2 = tlx2 + w2, tly2 + h2

    xmin, xmax = min(tlx1, tlx2), max(brx1, brx2)
    ymin, ymax = min(tly1, tly2), max(bry1, bry2)
    w = xmax - xmin
    h = ymax - ymin
    return (xmin, ymin, w, h)


def bb_intersection_over_union(box1, box2):
    # determine the (x, y)-coordinates of the intersection rectangle
    tlx1, tly1, w1, h1 = box1
    tlx2, tly2, w2, h2 = box2
    brx1, bry1 = tlx1 + w1, tly1 + h1
    brx2, bry2 = tlx2 + w2, tly2 + h2
    xa, xb = max(tlx1, tlx2), min(brx1, brx2)
    ya, yb = max(tly1, tly2), min(bry1, bry2)

    # compute the area of intersection rectangle
    interArea = abs(max((xb - xa, 0)) * max((yb - ya), 0))
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
    >>> boxes=[(1, 1, 5, 5), (0, 8, 5, 2), (0, 2, 5, 5)]
    >>> class_names=['a', 'b', 'c']
    >>> scores=[0.8, 0.7, 0.6]
    >>> merge_bounding_boxes(boxes, class_names, scores)
    ([(0, 1, 6, 6), (0, 8, 5, 2)], ['a', 'b'], [0.8, 0.7])
    """
    n = len(boxes)
    merged = [False] * n
    boxes = list(boxes)
    scores = list(scores)
    for i in range(n):
        if merged[i]:
            continue
        for j in range(i + 1, n):
            if merged[j]:
                continue
            if are_boxes_overlaping(
                    boxes[i], boxes[j]) and bb_intersection_over_union(
                        boxes[i], boxes[j]) >= merge_min_iou_thresh:
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
