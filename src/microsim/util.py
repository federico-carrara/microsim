from __future__ import annotations

import itertools
import shutil
import warnings
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast
from urllib import parse, request
from urllib.error import HTTPError

import numpy as np
import numpy.typing as npt
import platformdirs
import tqdm
from scipy import signal

from ._data_array import ArrayProtocol, DataArray, xrDataArray

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from pathlib import Path
    from typing import Literal

    from numpy.typing import DTypeLike, NDArray

    ShapeLike = Sequence[int]


# don't use this directly... it's patched during tests
# use cache_path() instead
_MICROSIM_CACHE = platformdirs.user_cache_path("microsim")


def microsim_cache(subdir: Literal["psf", "ground_truth"] | None = None) -> Path:
    """Return the microsim cache path.

    If `subdir` is provided, return the path to the specified subdirectory.
    (We use literal here to ensure that only the specified values are allowed.)
    """
    if subdir:
        return _MICROSIM_CACHE / subdir
    return _MICROSIM_CACHE


def clear_cache(pattern: str | None = None) -> None:
    """Clear the microsim cache."""
    if pattern:
        for p in microsim_cache().glob(pattern):
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p, ignore_errors=True)
    else:
        shutil.rmtree(microsim_cache(), ignore_errors=True)


def uniformly_spaced_coords(
    shape: tuple[int, ...] = (64, 128, 128),
    scale: tuple[float, ...] = (),
    extent: tuple[float, ...] = (),
    axes: str | Sequence[str] = "ZYX",
) -> dict[str, Sequence[float]]:
    # we now calculate the shape, scale, and extent based on input
    # where shape is the shape of the array, scale is the spacing between points
    # and extent is the total size of the array in each dimension (shape * scale)
    if not shape:
        if not extent:
            raise ValueError("Must provide either 'shape' or 'extent'")
        if not scale:
            raise ValueError("Must provide 'scale' along with 'extent'.")
        # scale = scale or ((1,) * len(extent))
        shape = tuple(int(x / s) for x, s in zip(extent, scale, strict=False))
    elif extent:
        if scale:
            warnings.warn(
                "Overdetermined: all three of shape, scale, and extent provided."
                "Ignoring value for extent.",
                stacklevel=2,
            )
        else:
            scale = tuple(x / s for x, s in zip(extent, shape, strict=False))
    elif not scale:
        scale = (1,) * len(shape)

    if not all(isinstance(i, int) for i in shape):
        raise TypeError(f"Shape must be a tuple of integers. Got {shape!r}")

    ndim = len(shape)
    if len(scale) != ndim:
        raise ValueError(f"length of scale and shape must match ({len(scale)}, {ndim})")
    if len(axes) < ndim:
        raise ValueError(f"Only {len(axes)} axes provided but got {ndim} dims")

    axes = axes[-ndim:]  # pick last ndim axes, in case there are too many provided.
    return {
        ax: np.arange(sh) * sc  # type: ignore
        for ax, sh, sc in zip(axes, shape, scale, strict=False)
    }


def uniformly_spaced_xarray(
    shape: tuple[int, ...] = (64, 128, 128),
    scale: tuple[float, ...] = (),
    extent: tuple[float, ...] = (),
    axes: str | Sequence[str] = "ZYX",
    array_creator: Callable[[ShapeLike], ArrayProtocol] = np.zeros,
    attrs: Mapping | None = None,
) -> xrDataArray:
    coords = uniformly_spaced_coords(shape, scale, extent, axes)
    shape = tuple(len(c) for c in coords.values())
    return DataArray(array_creator(shape), dims=tuple(axes), coords=coords, attrs=attrs)


def get_fftconvolve_shape(
    in1: npt.NDArray,
    in2: npt.NDArray,
    mode: Literal["full", "valid", "same"] = "full",
    axes: int | Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Get output shape of an fftconvolve operation (without performing it).

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution. The default is over all axes.

    Returns
    -------
    tuple
        Tuple of ints, with output shape

    Raises
    ------
    ValueError
        If in1.shape and in2.shape are invalid for the provided mode.
    """
    if mode == "same":
        return in1.shape

    s1 = in1.shape
    s2 = in2.shape
    ndim = in1.ndim
    if axes is None:
        _axes = set(range(ndim))
    else:
        _axes = set(range(axes) if isinstance(axes, int) else axes)

    full_shape = tuple(
        max((s1[i], s2[i])) if i not in _axes else s1[i] + s2[i] - 1 for i in _axes
    )

    if mode == "valid":
        final_shape = tuple(
            full_shape[a] if a not in _axes else s1[a] - s2[a] + 1
            for a in range(len(full_shape))
        )
        if any(i <= 0 for i in final_shape):
            raise ValueError(
                "For 'valid' mode, one must be at least "
                "as large as the other in every dimension"
            )
    elif mode == "full":
        final_shape = full_shape

    else:
        raise ValueError("Acceptable mode flags are 'valid'," " 'same', or 'full'")
    return final_shape


def _centered(arr: NDArray, newshape: ShapeLike) -> NDArray:
    """Return the center `newshape` portion of `arr`."""
    _newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind: NDArray = (currshape - _newshape) // 2
    endind: NDArray = startind + _newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _iter_block_locations(
    chunks: tuple[tuple[int, ...]],
) -> Iterator[tuple[list[tuple[int, int]], tuple[int, ...], ShapeLike]]:
    """Iterate block indices.

    Examples
    --------
    >>> chunks = ((2, 2), (3, 3), (4, 4, 4))
    >>> list(_iter_block_locations(chunks))
    [
        ([(0, 2), (0, 3), (0, 4)], (0, 0, 0), (2, 3, 4)),
        ([(0, 2), (0, 3), (4, 8)], (0, 0, 1), (2, 3, 4)),
        ([(0, 2), (0, 3), (8, 12)], (0, 0, 2), (2, 3, 4)),
        ...
    ]
    """
    starts = [(0, *tuple(np.cumsum(i))) for i in chunks]
    for block_id in itertools.product(*(range(len(c)) for c in chunks)):
        arr_slc = [(starts[ij][j], starts[ij][j + 1]) for ij, j in enumerate(block_id)]
        chunk_shape = tuple(chunks[ij][j] for ij, j in enumerate(block_id))
        yield arr_slc, block_id, chunk_shape


class Convolver(Protocol):
    def __call__(
        self, in1: NDArray, in2: NDArray, mode: Literal["full", "valid", "same"]
    ) -> NDArray: ...


def tiled_convolve(
    in1: npt.NDArray,
    in2: npt.NDArray,
    mode: Literal["full", "valid", "same"] = "full",
    chunks: tuple | None = None,
    func: Convolver = signal.convolve,
    dtype: DTypeLike | None = None,
) -> npt.NDArray:
    from dask.array.core import normalize_chunks

    if chunks is None:
        chunks = getattr(in1, "chunks", None) or (100,) * in1.ndim  # TODO: change 100

    _chunks: tuple[tuple[int, ...]] = normalize_chunks(chunks, in1.shape)

    final_shape = get_fftconvolve_shape(in1, in2, mode="full")

    out = np.zeros(final_shape, dtype=dtype)
    for loc, *_ in tqdm.tqdm(list(_iter_block_locations(_chunks))):
        block = np.asarray(in1[tuple(slice(*i) for i in loc)])
        result = func(block, in2, mode="full")
        if hasattr(result, "get"):
            result = result.get()
        out_idx = tuple(
            slice(i, i + s) for (i, _), s in zip(loc, result.shape, strict=False)
        )
        out[out_idx] += result
        del result

    if mode == "same":
        return _centered(out, in1.shape)
    elif mode == "valid":
        return _centered(out, get_fftconvolve_shape(in1, in2, mode="valid"))
    return out


# convenience function we'll use a couple times
def ortho_plot(
    img: ArrayProtocol,
    gamma: float = 0.5,
    mip: bool = False,
    cmap: str | list[str] | None = None,
    *,
    title: str | None = None,
    show: bool = True,
) -> None:
    """Plot XY and XZ slices of a 3D array."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if isinstance(img, xrDataArray):
        img = img.data
    if hasattr(img, "get"):
        img = img.get()
    img = np.asarray(img).squeeze()
    cmap = [cmap] if isinstance(cmap, str) else cmap
    if img.ndim == 3:
        channels = [img]
        cm_list = cmap if cmap is not None else ["gray"]
    elif img.ndim == 4:
        channels = list(img)
        colors = ["green", "magenta", "cyan", "yellow", "red", "blue"]
        cm_list = cmap if cmap is not None else colors
    else:
        raise ValueError("Input must be a 3D or 4D array")

    # Initialize RGB images for xy and xz
    xy_rgb = np.zeros((channels[0].shape[1], channels[0].shape[2], 3))
    xz_rgb = np.zeros((channels[0].shape[0], channels[0].shape[2], 3))

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    for img, cmap in zip(channels, cm_list, strict=False):
        xy = np.max(img, axis=0) if mip else img[img.shape[0] // 2]
        xz = np.max(img, axis=1) if mip else img[:, img.shape[1] // 2]

        # Normalize the images to the range [0, 1]
        xy = (xy - xy.min()) / (xy.max() - xy.min())
        xz = (xz - xz.min()) / (xz.max() - xz.min())

        # Apply gamma correction
        xy = np.power(xy, 1 / gamma)
        xz = np.power(xz, 1 / gamma)

        # Convert the grayscale images to RGB using the specified colormap
        cm = LinearSegmentedColormap.from_list("_cmap", ["black", cmap])
        xy_rgb += cm(xy)[..., :3]  # Exclude alpha channel
        xz_rgb += cm(xz)[..., :3]  # Exclude alpha channel

    # Clip the values to the range [0, 1]
    xy_rgb = np.clip(xy_rgb, 0, 1)
    xz_rgb = np.clip(xz_rgb, 0, 1)

    ax[0].imshow(xy_rgb)
    ax[1].imshow(xz_rgb)
    ax[0].set_title("XY slice")
    ax[1].set_title("XZ slice")
    try:
        fig.set_layout_engine("tight")
    except AttributeError:
        fig.set_tight_layout(True)

    if title:
        fig.suptitle(title)
    if show:
        plt.show()


def ndview(ary: Any, cmap: Any | None = None) -> None:
    """View any array using ndv.imshow.

    This function is a thin wrapper around `ndv.imshow`.
    """
    try:
        import ndv
        import qtpy
        import vispy.app
    except ImportError as e:
        raise ImportError(
            "Please `pip install 'ndv[pyqt,vispy]' to use this function."
        ) from e

    vispy.use(qtpy.API_NAME)
    ndv.imshow(ary, cmap=cmap)


ArrayType = TypeVar("ArrayType", bound=ArrayProtocol)


def downsample(
    array: ArrayType,
    factor: int | Sequence[int],
    method: Callable[
        [ArrayType, Sequence[int] | int | None, npt.DTypeLike], ArrayType
    ] = np.sum,
    dtype: npt.DTypeLike | None = None,
) -> ArrayType:
    binfactor = (factor,) * array.ndim if isinstance(factor, int) else factor
    new_shape = []
    for s, b in zip(array.shape, binfactor, strict=False):
        new_shape.extend([s // b, b])
    reshaped = cast("ArrayType", np.reshape(array, new_shape))
    for d in range(array.ndim):
        reshaped = method(reshaped, -1 * (d + 1), dtype)
    return reshaped


def bin_window(
    array: npt.NDArray,
    window: int | Sequence[int],
    dtype: npt.DTypeLike | None = None,
    method: str | Callable = "sum",
) -> npt.NDArray:
    """Bin an nd-array by applying `method` over `window`."""
    # TODO: deal with xarray

    binwindow = (window,) * array.ndim if isinstance(window, int) else window
    new_shape = []
    for s, b in zip(array.shape, binwindow, strict=False):
        new_shape.extend([s // b, b])

    sliced = array[
        tuple(slice(0, s * b) for s, b in zip(new_shape[::2], binwindow, strict=True))
    ]
    reshaped = np.reshape(sliced, new_shape)

    if callable(method):
        f = method
    elif method == "mode":
        # round and cast to int before calling bincount
        reshaped = np.round(reshaped).astype(np.int32, casting="unsafe")

        def f(a: npt.NDArray, axis: int) -> npt.NDArray:
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis, a)
    else:
        f = getattr(np, method)
    axes = tuple(range(1, reshaped.ndim, 2))
    result = np.apply_over_axes(f, reshaped, axes).squeeze()
    if dtype is not None:
        result = result.astype(dtype)
    return result


def norm_name(name: str) -> str:
    """Normalize a name to something easily searchable."""
    name = str(name).lower()
    for char in " -/\\()[],;:!?@#$%^&*+=|<>'\"":
        name = name.replace(char, "_")
    return name


def http_get(url: str, params: dict | None = None) -> bytes:
    """API like requests.get but with standard-library urllib."""
    if params:
        url += "?" + parse.urlencode(params)

    with request.urlopen(url) as response:
        if not 200 <= response.getcode() < 300:
            raise HTTPError(
                url, response.getcode(), "HTTP request failed", response.headers, None
            )
        return cast(bytes, response.read())


rec_header_dtd = \
    [
        ("nx", "i4"),  # Number of columns
        ("ny", "i4"),  # Number of rows
        ("nz", "i4"),  # Number of sections

        ("mode", "i4"),  # Types of pixels in the image. Values used by IMOD:
        #  0 = unsigned or signed bytes depending on flag in imodFlags
        #  1 = signed short integers (16 bits)
        #  2 = float (32 bits)
        #  3 = short * 2, (used for complex data)
        #  4 = float * 2, (used for complex data)
        #  6 = unsigned 16-bit integers (non-standard)
        # 16 = unsigned char * 3 (for rgb data, non-standard)

        ("nxstart", "i4"),  # Starting point of sub-image (not used in IMOD)
        ("nystart", "i4"),
        ("nzstart", "i4"),

        ("mx", "i4"),  # Grid size in X, Y and Z
        ("my", "i4"),
        ("mz", "i4"),

        ("xlen", "f4"),  # Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
        ("ylen", "f4"),
        ("zlen", "f4"),

        ("alpha", "f4"),  # Cell angles - ignored by IMOD
        ("beta", "f4"),
        ("gamma", "f4"),

        # These need to be set to 1, 2, and 3 for pixel spacing to be interpreted correctly
        ("mapc", "i4"),  # map column  1=x,2=y,3=z.
        ("mapr", "i4"),  # map row     1=x,2=y,3=z.
        ("maps", "i4"),  # map section 1=x,2=y,3=z.

        # These need to be set for proper scaling of data
        ("amin", "f4"),  # Minimum pixel value
        ("amax", "f4"),  # Maximum pixel value
        ("amean", "f4"),  # Mean pixel value

        ("ispg", "i4"),  # space group number (ignored by IMOD)
        ("next", "i4"),  # number of bytes in extended header (called nsymbt in MRC standard)
        ("creatid", "i2"),  # used to be an ID number, is 0 as of IMOD 4.2.23
        ("extra_data", "V30"),  # (not used, first two bytes should be 0)

        # These two values specify the structure of data in the extended header; their meaning depend on whether the
        # extended header has the Agard format, a series of 4-byte integers then real numbers, or has data
        # produced by SerialEM, a series of short integers. SerialEM stores a float as two shorts, s1 and s2, by:
        # value = (sign of s1)*(|s1|*256 + (|s2| modulo 256)) * 2**((sign of s2) * (|s2|/256))
        ("nint", "i2"),
        # Number of integers per section (Agard format) or number of bytes per section (SerialEM format)
        ("nreal", "i2"),  # Number of reals per section (Agard format) or bit
        # Number of reals per section (Agard format) or bit
        # flags for which types of short data (SerialEM format):
        # 1 = tilt angle * 100  (2 bytes)
        # 2 = piece coordinates for montage  (6 bytes)
        # 4 = Stage position * 25    (4 bytes)
        # 8 = Magnification / 100 (2 bytes)
        # 16 = Intensity * 25000  (2 bytes)
        # 32 = Exposure dose in e-/A2, a float in 4 bytes
        # 128, 512: Reserved for 4-byte items
        # 64, 256, 1024: Reserved for 2-byte items
        # If the number of bytes implied by these flags does
        # not add up to the value in nint, then nint and nreal
        # are interpreted as ints and reals per section

        ("extra_data2", "V20"),  # extra data (not used)
        ("imodStamp", "i4"),  # 1146047817 indicates that file was created by IMOD
        ("imodFlags", "i4"),  # Bit flags: 1 = bytes are stored as signed

        # Explanation of type of data
        ("idtype", "i2"),  # ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
        ("lens", "i2"),
        # ("nd1", "i2"),  # for idtype = 1, nd1 = axis (1, 2, or 3)
        # ("nd2", "i2"),
        ("nphase", "i4"),
        ("vd1", "i2"),  # vd1 = 100. * tilt increment
        ("vd2", "i2"),  # vd2 = 100. * starting angle

        # Current angles are used to rotate a model to match a new rotated image.  The three values in each set are
        # rotations about X, Y, and Z axes, applied in the order Z, Y, X.
        ("triangles", "f4", 6),  # 0,1,2 = original:  3,4,5 = current

        ("xorg", "f4"),  # Origin of image
        ("yorg", "f4"),
        ("zorg", "f4"),

        ("cmap", "S4"),  # Contains "MAP "
        ("stamp", "u1", 4),  # First two bytes have 17 and 17 for big-endian or 68 and 65 for little-endian

        ("rms", "f4"),  # RMS deviation of densities from mean density

        ("nlabl", "i4"),  # Number of labels with useful data
        ("labels", "S80", 10)  # 10 labels of 80 charactors
    ]


def _read_mrc(
    fpath: str, 
    filetype: Literal['image'] = 'image'
):

    fd = open(fpath, 'rb')
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)

    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]

    if header[0][3] == 1:
        data_type = 'int16'
    elif header[0][3] == 2:
        data_type = 'float32'
    elif header[0][3] == 4:
        data_type = 'single'
        nx = nx * 2
    elif header[0][3] == 6:
        data_type = 'uint16'

    data = np.ndarray(shape=(nx, ny, nz))
    imgrawdata = np.fromfile(fd, data_type)
    fd.close()

    if filetype == 'image':
        for iz in range(nz):
            data_2d = imgrawdata[nx*ny*iz:nx*ny*(iz+1)]
            data[:, :, iz] = data_2d.reshape(nx, ny, order='F')
    else:
        data = imgrawdata

    return data

def compute_FP_intensity_difference(imgs: xrDataArray):
    """Compute the relative difference in intensity between emission images from
    different fluorophores.
    
    """
    from .schema.dimensions import Axis
    
    n = imgs.sizes[Axis.F]
    if imgs.coords[Axis.F].values[-1] == "mixed_spectrum":
        slc = slice(n-1)
        imgs = imgs.isel(f=slc)
        n -= 1
    qtiles_lst = []
    for i in range(n):
        img = imgs.isel(f=i)
        img = img.values.flatten()
        qtiles_lst.append(np.quantile(img, (0.5, 0.75, 0.95, 0.99)))
    
    # get the first quantiles as reference; compute the difference w.r.t. to it
    ref_qtiles = qtiles_lst[0]
    diffs = []
    for qtiles in qtiles_lst[1:]:
        diff = np.mean((qtiles - ref_qtiles) / ref_qtiles)
        diffs.append(diff)
        
    return diffs