"""Depth-based image quality checks; these do not assign object identity."""
import numpy as np


def candidate_view_quality(obj, frame, bounds):
    """Measure foreground obstruction along rays through the measured body.

    Use the projected volume, not the detected visible fragment: a small cap
    can be detected even when a robot link hides most of the body. Depth is
    camera-Z, so the unnormalised camera rays use that same parameterisation.
    The margin tolerates imperfect surface bounds; it is not a shape classifier.
    """
    x0, y0, x1, y1 = bounds
    yy, xx = np.mgrid[y0:y1, x0:x1]
    k = frame.intrinsics
    rays = np.stack(((xx-k[0, 2])/k[0, 0], (yy-k[1, 2])/k[1, 1],
                     np.ones_like(xx)), axis=-1) @ frame.camera_to_world[:3, :3].T
    origin = frame.camera_to_world[:3, 3]
    low, high = obj.low-.01, obj.high+.01
    parallel = np.abs(rays) < 1e-10
    safe_rays = np.where(parallel, 1., rays)
    a, b = (low-origin)/safe_rays, (high-origin)/safe_rays
    inside = (origin >= low) & (origin <= high)
    entry = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(a, b))
    exit_ = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(a, b))
    near, far = entry.max(-1), exit_.min(-1)
    footprint = far >= np.maximum(near, .025)
    depth = frame.depth[y0:y1, x0:x1]
    known = footprint & np.isfinite(depth) & (depth > .025) & (depth < 3.)
    occluded = known & (depth < near-.01)
    if frame.robot_mask is not None:
        robot = footprint & frame.robot_mask[y0:y1, x0:x1]
        occluded |= robot
        known |= robot
    pixels = int(footprint.sum())
    known_pixels = int(known.sum())
    coverage = known_pixels/max(pixels, 1)
    obstruction = float(occluded.sum()/max(known_pixels, 1))
    return {'usable': pixels >= 30 and coverage >= .5 and obstruction <= .35,
            'projected_pixels': pixels, 'depth_coverage': coverage,
            'foreground_occlusion_fraction': obstruction}
