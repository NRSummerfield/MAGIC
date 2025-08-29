from typing import Union, Any
import numpy as np, torch
import cupy as cp
import scipy.ndimage as ndimage
import monai
import re
from cucim.core.operations.morphology import distance_transform_edt as _dte

from cupy import ndarray as cp_ndarray
has_cp = True
has_ndimage = True

def dtype_torch_to_numpy(dtype: torch.dtype) -> np.dtype:
    """Convert a torch dtype to its numpy equivalent."""
    return torch.empty([], dtype=dtype).numpy().dtype  # type: ignore


def dtype_numpy_to_torch(dtype: np.dtype) -> torch.dtype:
    """Convert a numpy dtype to its torch equivalent."""
    return torch.from_numpy(np.empty([], dtype=dtype)).dtype

def get_equivalent_dtype(dtype, data_type):
    """Convert to the `dtype` that corresponds to `data_type`.

    The input dtype can also be a string. e.g., `"float32"` becomes `torch.float32` or
    `np.float32` as necessary.

    Example::

        im = torch.tensor(1)
        dtype = get_equivalent_dtype(np.float32, type(im))

    """
    if dtype is None:
        return None
    if data_type is torch.Tensor or data_type.__name__ == "MetaTensor":
        if isinstance(dtype, torch.dtype):
            # already a torch dtype and target `data_type` is torch.Tensor
            return dtype
        return dtype_numpy_to_torch(dtype)
    if not isinstance(dtype, torch.dtype):
        # assuming the dtype is ok if it is not a torch dtype and target `data_type` is not torch.Tensor
        return dtype
    return dtype_torch_to_numpy(dtype)

UNSUPPORTED_TYPES = {np.dtype("uint16"): np.int32, np.dtype("uint32"): np.int64, np.dtype("uint64"): np.int64}
def get_dtype_bound_value(dtype) -> tuple[float, float]:
    """
    Get dtype bound value
    Args:
        dtype: dtype to get bound value
    Returns:
        (bound_min_value, bound_max_value)
    """
    if dtype in UNSUPPORTED_TYPES:
        is_floating_point = False
    else:
        is_floating_point = get_equivalent_dtype(dtype, torch.Tensor).is_floating_point
    dtype = get_equivalent_dtype(dtype, np.array)
    if is_floating_point:
        return (np.finfo(dtype).min, np.finfo(dtype).max)  # type: ignore
    else:
        return (np.iinfo(dtype).min, np.iinfo(dtype).max)


def safe_dtype_range(data: Any, dtype = None) -> Any:
    """
    Utility to safely convert the input data to target dtype.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert to target dtype and keep the original type.
            for dictionary, list or tuple, convert every item.
        dtype: target data type to convert.
    """

    def _safe_dtype_range(data, dtype):
        output_dtype = dtype if dtype is not None else data.dtype
        dtype_bound_value = get_dtype_bound_value(output_dtype)
        if data.ndim == 0:
            data_bound = (data, data)
        else:
            if isinstance(data, torch.Tensor):
                data_bound = (torch.min(data), torch.max(data))
            else:
                data_bound = (np.min(data), np.max(data))
        if (data_bound[1] > dtype_bound_value[1]) or (data_bound[0] < dtype_bound_value[0]):
            if isinstance(data, torch.Tensor):
                return torch.clamp(data, dtype_bound_value[0], dtype_bound_value[1])
            elif isinstance(data, np.ndarray):
                return np.clip(data, dtype_bound_value[0], dtype_bound_value[1])
            elif has_cp and isinstance(data, cp_ndarray):
                return cp.clip(data, dtype_bound_value[0], dtype_bound_value[1])
        else:
            return data

    if has_cp and isinstance(data, cp_ndarray):
        return cp.asarray(_safe_dtype_range(data, dtype))
    elif isinstance(data, np.ndarray):
        return np.asarray(_safe_dtype_range(data, dtype))
    elif isinstance(data, torch.Tensor):
        return _safe_dtype_range(data, dtype)
    elif isinstance(data, (float, int, bool)) and dtype is None:
        return data
    elif isinstance(data, (float, int, bool)) and dtype is not None:
        output_dtype = dtype
        dtype_bound_value = get_dtype_bound_value(output_dtype)
        data = dtype_bound_value[1] if data > dtype_bound_value[1] else data
        data = dtype_bound_value[0] if data < dtype_bound_value[0] else data
        return data

    elif isinstance(data, list):
        return [safe_dtype_range(i, dtype=dtype) for i in data]
    elif isinstance(data, tuple):
        return tuple(safe_dtype_range(i, dtype=dtype) for i in data)
    elif isinstance(data, dict):
        return {k: safe_dtype_range(v, dtype=dtype) for k, v in data.items()}
    return data

def convert_to_tensor(
    data: Any,
    dtype = None,
    device = None,
    wrap_sequence: bool = False,
    track_meta: bool = False,
    safe: bool = False,
) -> Any:
    """
    Utility to convert the input data to a PyTorch Tensor, if `track_meta` is True, the output will be a `MetaTensor`,
    otherwise, the output will be a regular torch Tensor.
    If passing a dictionary, list or tuple, recursively check every item and convert it to PyTorch Tensor.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to Tensor, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a Tensor if applicable.
        dtype: target data type to when converting to Tensor.
        device: target device to put the converted Tensor data.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[tensor(1), tensor(2)]`. If `True`, then `[1, 2]` -> `tensor([1, 2])`.
        track_meta: whether to track the meta information, if `True`, will convert to `MetaTensor`.
            default to `False`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[tensor(0), tensor(244)]`.
            If `True`, then `[256, -12]` -> `[tensor(255), tensor(0)]`.

    """

    def _convert_tensor(tensor: Any, **kwargs: Any) -> Any:
        if not isinstance(tensor, torch.Tensor):
            # certain numpy types are not supported as being directly convertible to Pytorch tensors
            if isinstance(tensor, np.ndarray) and tensor.dtype in UNSUPPORTED_TYPES:
                tensor = tensor.astype(UNSUPPORTED_TYPES[tensor.dtype])

            # if input data is not Tensor, convert it to Tensor first
            tensor = torch.as_tensor(tensor, **kwargs)
        if track_meta and not isinstance(tensor, monai.data.MetaTensor):
            return monai.data.MetaTensor(tensor)
        if not track_meta and isinstance(tensor, monai.data.MetaTensor):
            return tensor.as_tensor()
        return tensor

    if safe:
        data = safe_dtype_range(data, dtype)
    dtype = get_equivalent_dtype(dtype, torch.Tensor)
    if isinstance(data, torch.Tensor):
        return _convert_tensor(data).to(dtype=dtype, device=device, memory_format=torch.contiguous_format)
    if isinstance(data, np.ndarray):
        # skip array of string classes and object, refer to:
        # https://github.com/pytorch/pytorch/blob/v1.9.0/torch/utils/data/_utils/collate.py#L13
        if re.search(r"[SaUO]", data.dtype.str) is None:
            # numpy array with 0 dims is also sequence iterable,
            # `ascontiguousarray` will add 1 dim if img has no dim, so we only apply on data with dims
            if data.ndim > 0:
                data = np.ascontiguousarray(data)
            return _convert_tensor(data, dtype=dtype, device=device)
    elif (has_cp and isinstance(data, cp_ndarray)) or isinstance(data, (float, int, bool)):
        return _convert_tensor(data, dtype=dtype, device=device)
    elif isinstance(data, list):
        list_ret = [convert_to_tensor(i, dtype=dtype, device=device, track_meta=track_meta) for i in data]
        return _convert_tensor(list_ret, dtype=dtype, device=device) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_tensor(i, dtype=dtype, device=device, track_meta=track_meta) for i in data)
        return _convert_tensor(tuple_ret, dtype=dtype, device=device) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_tensor(v, dtype=dtype, device=device, track_meta=track_meta) for k, v in data.items()}

    return data


def convert_data_type(
    data: Any,
    output_type = None,
    device = None,
    dtype = None,
    wrap_sequence: bool = False,
    safe: bool = False,
):
    """
    Convert to `MetaTensor`, `torch.Tensor` or `np.ndarray` from `MetaTensor`, `torch.Tensor`,
    `np.ndarray`, `float`, `int`, etc.

    Args:
        data: data to be converted
        output_type: `monai.data.MetaTensor`, `torch.Tensor`, or `np.ndarray` (if `None`, unchanged)
        device: if output is `MetaTensor` or `torch.Tensor`, select device (if `None`, unchanged)
        dtype: dtype of output data. Converted to correct library type (e.g.,
            `np.float32` is converted to `torch.float32` if output type is `torch.Tensor`).
            If left blank, it remains unchanged.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.

    Returns:
        modified data, orig_type, orig_device

    Note:
        When both `output_type` and `dtype` are specified with different backend
        (e.g., `torch.Tensor` and `np.float32`), the `output_type` will be used as the primary type,
        for example::

            >>> convert_data_type(1, torch.Tensor, dtype=np.float32)
            (1.0, <class 'torch.Tensor'>, None)

    """
    orig_type: type
    if isinstance(data, monai.data.MetaTensor):
        orig_type = monai.data.MetaTensor
    elif isinstance(data, torch.Tensor):
        orig_type = torch.Tensor
    elif isinstance(data, np.ndarray):
        orig_type = np.ndarray
    elif has_cp and isinstance(data, cp.ndarray):
        orig_type = cp.ndarray
    else:
        orig_type = type(data)

    orig_device = data.device if isinstance(data, torch.Tensor) else None

    output_type = output_type or orig_type
    dtype_ = get_equivalent_dtype(dtype, output_type)

    # data_: s
    if issubclass(output_type, torch.Tensor):
        track_meta = issubclass(output_type, monai.data.MetaTensor)
        data_ = convert_to_tensor(
            data, dtype=dtype_, device=device, wrap_sequence=wrap_sequence, track_meta=track_meta, safe=safe
        )
        return data_, orig_type, orig_device
    if issubclass(output_type, np.ndarray):
        data_ = convert_to_numpy(data, dtype=dtype_, wrap_sequence=wrap_sequence, safe=safe)
        return data_, orig_type, orig_device
    elif has_cp and issubclass(output_type, cp.ndarray):
        data_ = convert_to_cupy(data, dtype=dtype_, wrap_sequence=wrap_sequence, safe=safe)
        return data_, orig_type, orig_device
    raise ValueError(f"Unsupported output type: {output_type}")


def convert_to_numpy(data: Any, dtype = None, wrap_sequence: bool = False, safe: bool = False) -> Any:
    """
    Utility to convert the input data to a numpy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to numpy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, list, dictionary, int, float, bool, str, etc.
            will convert Tensor, Numpy array, float, int, bool to numpy arrays, strings and objects keep the original.
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to numpy array.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.
    """
    if safe:
        data = safe_dtype_range(data, dtype)
    if isinstance(data, torch.Tensor):
        data = np.asarray(data.detach().to(device="cpu").numpy(), dtype=get_equivalent_dtype(dtype, np.ndarray))
    elif has_cp and isinstance(data, cp_ndarray):
        data = cp.asnumpy(data).astype(dtype, copy=False)
    elif isinstance(data, (np.ndarray, float, int, bool)):
        # Convert into a contiguous array first if the current dtype's size is smaller than the target dtype's size.
        # This help improve the performance because (convert to contiguous array) -> (convert dtype) is faster
        # than (convert dtype) -> (convert to contiguous array) when src dtype (e.g., uint8) is smaller than
        # target dtype(e.g., float32) and we are going to convert it to contiguous array anyway later in this
        # method.
        if isinstance(data, np.ndarray) and data.ndim > 0 and data.dtype.itemsize < np.dtype(dtype).itemsize:
            data = np.ascontiguousarray(data)
        data = np.asarray(data, dtype=dtype)
    elif isinstance(data, list):
        list_ret = [convert_to_numpy(i, dtype=dtype) for i in data]
        return np.asarray(list_ret) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_numpy(i, dtype=dtype) for i in data)
        return np.asarray(tuple_ret) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_numpy(v, dtype=dtype) for k, v in data.items()}

    if isinstance(data, np.ndarray) and data.ndim > 0:
        data = np.ascontiguousarray(data)

    return data

def convert_to_cupy(data: Any, dtype = None, wrap_sequence: bool = False, safe: bool = False) -> Any:
    """
    Utility to convert the input data to a cupy array. If passing a dictionary, list or tuple,
    recursively check every item and convert it to cupy array.

    Args:
        data: input data can be PyTorch Tensor, numpy array, cupy array, list, dictionary, int, float, bool, str, etc.
            Tensor, numpy array, cupy array, float, int, bool are converted to cupy arrays,
            for dictionary, list or tuple, convert every item to a numpy array if applicable.
        dtype: target data type when converting to Cupy array, tt must be an argument of `numpy.dtype`,
            for more details: https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html.
        wrap_sequence: if `False`, then lists will recursively call this function.
            E.g., `[1, 2]` -> `[array(1), array(2)]`. If `True`, then `[1, 2]` -> `array([1, 2])`.
        safe: if `True`, then do safe dtype convert when intensity overflow. default to `False`.
            E.g., `[256, -12]` -> `[array(0), array(244)]`. If `True`, then `[256, -12]` -> `[array(255), array(0)]`.
    """
    if safe:
        data = safe_dtype_range(data, dtype)
    # direct calls
    if isinstance(data, torch.Tensor) and data.device.type == "cuda":
        # This is needed because of https://github.com/cupy/cupy/issues/7874#issuecomment-1727511030
        if data.dtype == torch.bool:
            data = data.detach().to(torch.uint8)
            if dtype is None:
                dtype = bool  # type: ignore
        data = cp.asarray(data, dtype)
    elif isinstance(data, (cp_ndarray, np.ndarray, torch.Tensor, float, int, bool)):
        print(f'  Found dtype({type(data)})')
        data = cp.asarray(data.contiguous(), dtype)
    elif isinstance(data, list):
        list_ret = [convert_to_cupy(i, dtype) for i in data]
        return cp.asarray(list_ret) if wrap_sequence else list_ret
    elif isinstance(data, tuple):
        tuple_ret = tuple(convert_to_cupy(i, dtype) for i in data)
        return cp.asarray(tuple_ret) if wrap_sequence else tuple_ret
    elif isinstance(data, dict):
        return {k: convert_to_cupy(v, dtype) for k, v in data.items()}
    # make it contiguous
    if not isinstance(data, cp.ndarray):
        raise ValueError(f"The input data type [{type(data)}] cannot be converted into cupy arrays!")

    if data.ndim > 0:
        data = cp.ascontiguousarray(data)
    return data

def distance_transform_edt(
    img: Union[np.ndarray, torch.Tensor],
    sampling = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances = None,
    indices = None,
    *,
    block_params = None,
    float64_distances: bool = False,
):
    
    distance_transform_edt = _dte

    use_cp = isinstance(img, torch.Tensor) and img.device.type == "cuda"
    if not return_distances and not return_indices:
        raise RuntimeError("Neither return_distances nor return_indices True")

    if not (img.ndim >= 3 and img.ndim <= 4):
        raise RuntimeError("Wrong input dimensionality. Use (num_channels, H, W [,D])")

    distances_original, indices_original = distances, indices
    distances, indices = None, None
    if use_cp:
        distances_, indices_ = None, None
        if return_distances:
            dtype = torch.float64 if float64_distances else torch.float32
            if distances is None:
                distances = torch.zeros_like(img, memory_format=torch.contiguous_format, dtype=dtype)  # type: ignore
            else:
                if not isinstance(distances, torch.Tensor) and distances.device != img.device:
                    raise TypeError("distances must be a torch.Tensor on the same device as img")
                if not distances.dtype == dtype:
                    raise TypeError("distances must be a torch.Tensor of dtype float32 or float64")
            distances_ = convert_to_cupy(distances)
        if return_indices:
            dtype = torch.int32
            if indices is None:
                indices = torch.zeros((img.dim(),) + img.shape, dtype=dtype)  # type: ignore
            else:
                if not isinstance(indices, torch.Tensor) and indices.device != img.device:
                    raise TypeError("indices must be a torch.Tensor on the same device as img")
                if not indices.dtype == dtype:
                    raise TypeError("indices must be a torch.Tensor of dtype int32")
            indices_ = convert_to_cupy(indices)
        img_ = convert_to_cupy(img)
        for channel_idx in range(img_.shape[0]):
            distance_transform_edt(
                img_[channel_idx],
                sampling=sampling,
                return_distances=return_distances,
                return_indices=return_indices,
                distances=distances_[channel_idx] if distances_ is not None else None,
                indices=indices_[channel_idx] if indices_ is not None else None,
                block_params=block_params,
                float64_distances=float64_distances,
            )
        torch.cuda.synchronize()
    else:
        if not has_ndimage:
            raise RuntimeError("scipy.ndimage required if cupy is not available")
        img_ = convert_to_numpy(img)
        if return_distances:
            if distances is None:
                distances = np.zeros_like(img_, dtype=np.float64)
            else:
                if not isinstance(distances, np.ndarray):
                    raise TypeError("distances must be a numpy.ndarray")
                if not distances.dtype == np.float64:
                    raise TypeError("distances must be a numpy.ndarray of dtype float64")
        if return_indices:
            if indices is None:
                indices = np.zeros((img_.ndim,) + img_.shape, dtype=np.int32)
            else:
                if not isinstance(indices, np.ndarray):
                    raise TypeError("indices must be a numpy.ndarray")
                if not indices.dtype == np.int32:
                    raise TypeError("indices must be a numpy.ndarray of dtype int32")

        for channel_idx in range(img_.shape[0]):
            ndimage.distance_transform_edt(
                img_[channel_idx],
                sampling=sampling,
                return_distances=return_distances,
                return_indices=return_indices,
                distances=distances[channel_idx] if distances is not None else None,
                indices=indices[channel_idx] if indices is not None else None,
            )

    r_vals = []
    if return_distances and distances_original is None:
        r_vals.append(distances_ if use_cp else distances)
    if return_indices and indices_original is None:
        r_vals.append(indices)
    if not r_vals:
        return None
    device = img.device if isinstance(img, torch.Tensor) else None
    return convert_data_type(r_vals[0] if len(r_vals) == 1 else r_vals, output_type=type(img), device=device)[0]
