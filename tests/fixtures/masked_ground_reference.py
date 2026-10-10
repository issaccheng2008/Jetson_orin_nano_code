"""Image operations on observed ground only; padding has zero statistical weight."""
import cv2
import numpy as np


def validity(image, mask):
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if image.ndim != 2 or mask.shape != image.shape:
        raise ValueError('ground mask must match the grayscale image')
    return mask


def gaussian(image, valid, size, sigma=0):
    """Normalized convolution: sum(weight * value) / sum(valid weight)."""
    weight = valid.astype(np.float32)
    numerator = cv2.GaussianBlur(np.where(valid, image, 0).astype(np.float32),
                                size, sigma, borderType=cv2.BORDER_REPLICATE)
    denominator = cv2.GaussianBlur(weight, size, sigma, borderType=cv2.BORDER_REPLICATE)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator > 1e-8)


def morphology(image, operation, kernel, valid):
    """Invalid neighbours are excluded from min/max, not treated as black ink."""
    def dilate(x):
        out = cv2.dilate(np.where(valid, x, 0).astype(np.uint8), kernel)
        out[~valid] = 0
        return out
    def erode(x):
        out = cv2.erode(np.where(valid, x, 255).astype(np.uint8), kernel)
        out[~valid] = 0
        return out
    if operation == cv2.MORPH_CLOSE:
        return erode(dilate(image))
    if operation == cv2.MORPH_OPEN:
        return dilate(erode(image))
    if operation == cv2.MORPH_BLACKHAT:
        out = cv2.subtract(erode(dilate(image)), image)
        out[~valid] = 0
        return out
    raise ValueError('unsupported masked morphology operation')


def clahe(image, valid, clip_limit=2., grid=(8, 8)):
    """CLAHE with observed-pixel tile histograms and bilinear LUT interpolation.

    Empty tiles use the global observed histogram; no synthetic padding samples
    enter either histogram. This path is used only when the image has holes.
    """
    h,w = image.shape
    nx,ny = min(grid[0],w),min(grid[1],h)
    tw,th = int(np.ceil(w/nx)),int(np.ceil(h/ny))
    luts = np.empty((ny,nx,256),np.float32)
    fallback = np.bincount(image[valid],minlength=256)
    for iy in range(ny):
        for ix in range(nx):
            tile = image[iy*th:min(h,(iy+1)*th),ix*tw:min(w,(ix+1)*tw)]
            keep = valid[iy*th:min(h,(iy+1)*th),ix*tw:min(w,(ix+1)*tw)]
            hist = np.bincount(tile[keep],minlength=256) if keep.any() else fallback.copy()
            count = int(hist.sum())
            if not count:
                luts[iy,ix] = np.arange(256)
                continue
            limit = max(1,int(clip_limit*count/256))
            excess = int(np.maximum(hist-limit,0).sum())
            hist = np.minimum(hist,limit)
            hist += excess//256
            remainder = excess%256
            if remainder:
                hist[np.arange(0,256,max(1,256//remainder))[:remainder]] += 1
            luts[iy,ix] = np.cumsum(hist)*(255./count)
    x=np.arange(w)/tw-.5; y=np.arange(h)/th-.5
    ix=np.floor(x).astype(int); iy=np.floor(y).astype(int)
    ax=(x-ix)[None,:]; ay=(y-iy)[:,None]
    x0=np.clip(ix,0,nx-1)[None,:]; x1=np.clip(ix+1,0,nx-1)[None,:]
    y0=np.clip(iy,0,ny-1)[:,None]; y1=np.clip(iy+1,0,ny-1)[:,None]
    top=(1-ax)*luts[y0,x0,image]+ax*luts[y0,x1,image]
    bottom=(1-ax)*luts[y1,x0,image]+ax*luts[y1,x1,image]
    out=np.clip(np.rint((1-ay)*top+ay*bottom),0,255).astype(np.uint8)
    out[~valid]=0
    return out
