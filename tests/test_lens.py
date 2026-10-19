import numpy as np

from vh.lens import blur_diameter, disc_blur, disc_kernel, image_distance


def test_lens_formula():
    f, z = 0.05, 2.0
    assert np.isclose(1 / z + 1 / image_distance(f, z), 1 / f)
    assert np.isclose(image_distance(f, 1e12), f)  # focused at infinity, the sensor is at f


def test_blur_vanishes_at_the_focus_depth_and_grows_with_aperture():
    f, z_focus = 0.05, 2.0
    s = image_distance(f, z_focus)
    assert np.isclose(blur_diameter(0.025, s, f, z_focus), 0.0)
    assert blur_diameter(0.025, s, f, 4.0) > blur_diameter(0.0125, s, f, 4.0) > 0.0


def test_blur_is_linear_in_inverse_depth():
    f, A = 0.05, 0.025
    s = image_distance(f, 2.0)
    z = np.array([1.0, 1.5, 3.0, 8.0, 1e9])
    assert np.allclose(blur_diameter(A, s, f, z), A * s * np.abs(1 / 2.0 - 1 / z))


def test_disc_blur_keeps_brightness_and_spreads_a_point():
    kernel = disc_kernel(4.0)
    assert np.isclose(kernel.sum(), 1.0)
    image = np.zeros((41, 41))
    image[20, 20] = 1.0
    blurred = disc_blur(image, 4.0)
    assert blurred.shape == image.shape
    assert np.isclose(blurred.sum(), 1.0)
    rows, cols = np.mgrid[:41, :41]
    assert np.isclose((blurred * rows).sum(), 20) and np.isclose((blurred * cols).sum(), 20)  # not shifted
    assert blurred[20, 24] > 0 and np.isclose(blurred[20, 27], 0.0, atol=1e-9)
    flat = np.full((10, 12, 3), 0.7)
    assert np.allclose(disc_blur(flat, 3.0), 0.7)  # edges do not darken
