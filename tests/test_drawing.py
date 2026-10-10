import numpy as np

from gestureflow.ui.drawing import draw_hands, to_pixels


def test_to_pixels_scales_and_clips():
    points = np.array([[0.5, 0.5, 0], [1.2, -0.1, 0]] + [[0, 0, 0]] * 19, dtype=np.float32)
    px = to_pixels(points, 640, 480)
    assert tuple(px[0]) == (320, 240)
    assert tuple(px[1]) == (639, 0)  # off-screen points are clamped to the edge


def test_draws_on_every_landmark(open_hand):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hands(image, [open_hand])
    for x, y in to_pixels(open_hand.points, 640, 480):
        assert image[y, x].any(), f"nothing drawn at landmark ({x}, {y})"


def test_no_hands_leaves_image_untouched():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    draw_hands(image, [])
    assert not image.any()
