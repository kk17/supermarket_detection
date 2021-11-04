
def crop_by_bounding_box(image_np, b):
    y_min, x_min, y_max, x_max = b
    sp = image_np.shape
    int_y_min, int_y_max, int_x_min, int_x_max = int(y_min * sp[0]), int(y_max * sp[0]), int(x_min * sp[1]), int(x_max * sp[1])
    image_np = image_np[int_y_min: int_y_max, int_x_min: int_x_max]
    return image_np