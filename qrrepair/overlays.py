"""Image evidence overlays; independent of guesses and corrected module values."""
import cv2
import numpy as np


def confidence_colors(strength):
    t = np.clip(np.asarray(strength), 0, 1)
    rgba = np.empty((*t.shape,4),np.uint8)
    rgba[...,:3] = [235,45,35]
    rgba[...,3] = np.rint(255*(1-t)).astype(np.uint8)
    return rgba


def contour_color(level):
    """Map grayscale black→deep navy and white→white."""
    t = np.clip(np.asarray(level), 0, 255)[..., None]/255
    return np.rint(np.array([12,45,105])*(1-t) + np.array([255,255,255])*t).astype(np.uint8)


def grayscale_contours(crop, interval=16):
    """Filled iso-intensity bands, white at zero and navy at 255."""
    if not 4 <= interval <= 128:
        raise ValueError("等高线灰阶间距应为 4–128")
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    gray = cv2.GaussianBlur(gray, (0,0), .8)
    levels = np.minimum(np.rint(gray.astype(np.float32)/interval)*interval,255)
    out = np.full((*gray.shape,4),255,np.uint8)
    out[...,:3] = contour_color(levels)
    return out


def grid_colors(grid, heatmap=False, preview=None):
    a = grid.values if preview is None else preview
    p = np.asarray(grid.probability, np.float32)
    chosen = np.where(a < 0, p >= .5, a == 1).astype(np.float32)
    light = np.array([255, 255, 255], np.float32)
    deep = np.array([0, 0, 0], np.float32)
    rgb = (light[None,None,:] * (1-chosen[...,None]) + deep[None,None,:] * chosen[...,None]).astype(np.uint8)
    if heatmap:
        strength = np.abs(grid.probability-.5)*2
        machine = (grid.source != 1) & (grid.source != 2)
        heat = confidence_colors(strength[machine])
        alpha = heat[:,3:4].astype(np.float32)/255
        rgb[machine] = np.rint(rgb[machine]*(1-alpha)+heat[:,:3]*alpha).astype(np.uint8)
    if preview is not None:
        changed = a != grid.values
        rgb[changed] = np.where((a[changed] == 1)[:,None], [0,128,105], [150,248,221])
    return rgb


def confidence_legend(width=180, height=16):
    return np.repeat(confidence_colors(np.linspace(0,1,width))[None,:,:],height,axis=0)
