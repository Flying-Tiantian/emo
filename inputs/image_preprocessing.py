import tensorflow as tf


def _sample_bbox(image, bbox):
    if bbox is None:
        return None
    
    return tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=[0.75, 1.33],
            area_range=[0.05, 1.0],
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)


def _bbox_crop(image, bbox):
    if bbox is None:
        return image

    bbox_begin, bbox_size, _ = bbox

    # Reassemble the bounding box in the format the crop op requires.
    offset_height, offset_width, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)

    # Use the fused decode and crop op here, which is faster than each in series.
    cropped = tf.image.crop_to_bounding_box(image, offset_height, offset_width, target_height, target_width)
    
    return cropped


def _central_crop(image, crop_height, crop_width):
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(
            image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def _image_standardization(image):
    return tf.image.per_image_standardization(image)

    max_value = tf.reduce_max(image)
    min_value = tf.reduce_min(image)

    image = (image - min_value) / (max_value - min_value)
    # image = image - tf.reduce_mean(image)
    
    return image


def _resize_image(image, height, width):
    return tf.image.resize_images(
            image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)


def _aspect_preserving_resize(image, resize_min):
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return _resize_image(image, new_height, new_width)


def preprocess_image(inputs, output_height, output_width, square=True, 
                     num_channels=3, padding=0, rand=True, bbox=None, is_training=False):
    if inputs.dtype == tf.string:
        image = tf.image.decode_image(inputs, channels=num_channels)
    else:
        image = inputs

    image.set_shape([None, None, None])
    
    if not square:
        image = _aspect_preserving_resize(image, min(output_height, output_width))
        image = _central_crop(image, output_height, output_width)

    if is_training:
        if padding > 0:
            padding *= 2
            shape = tf.shape(image)
            height, width = shape[0], shape[1]
            image = tf.image.resize_image_with_crop_or_pad(
                image, height + padding, width + padding)

            image = tf.image.random_crop(image, [height, width, num_channels])

        if rand:
            image = _bbox_crop(image, _sample_bbox(image, bbox))
            image = tf.image.random_flip_left_right(image)
    else:
        image = _bbox_crop(image, bbox)

    image = _resize_image(image, output_height, output_width)
    image.set_shape([output_height, output_width, num_channels])

    return _image_standardization(image)
