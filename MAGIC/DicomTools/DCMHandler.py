import SimpleITK as sitk, numpy as np, scipy.ndimage as ndimg, pydicom as pdcm, nibabel as nib, skimage.draw as draw

import glob, os, math as m, json
from datetime import datetime as dt
from typing import Union, Optional, Any

# --------------------------------
# Basic Functions
# --------------------------------

def get_image(*dicom_paths: str, return_numpy: bool = False):
    if len(dicom_paths) == 1 and isinstance(dicom_paths[0], (list, tuple)): dicom_paths = dicom_paths[0]
    return_first = True if len(dicom_paths) == 1 else False

    reader = sitk.ImageSeriesReader()
    imgs = [sitk.ReadImage(reader.GetGDCMSeriesFileNames(str(path))) for path in dicom_paths]
    if return_numpy: imgs = [sitk.GetArrayFromImage(img) for img in imgs]
    if return_first: imgs = imgs[0]
    return imgs

def file_from_dir(
    path: str,
    file_identity: str = '*.dcm',
    return_1st: bool = False,
    ) -> Union[str, list[str]]:
    if not os.path.isdir(path): return path 
    assert '*' in file_identity
    file = glob.glob(os.path.join(path, file_identity))

    if len(file) == 1 or return_1st: return file[0]
    else: return 
    
def return_if_one(paths: list[str], forced_index: Optional[int] = None):
    if not isinstance(paths, (list, tuple)): paths = [paths]
    if forced_index: return paths[forced_index]
    if (nPaths := len(paths)) == 1: return paths[0]
    else: raise IndexError('`paths` has {} elements -> Pass `forced_index` to continue...\nPaths found: {}'.format(nPaths, "\n  ".join(paths)))

def get_StudyInstanceUID(item: Union[str, pdcm.FileDataset]):
    """
    The dicom key [0x0020, 0x000d] Study Instance UID matches between the RTStruct and parent volume
    """
    if isinstance(item, str):
        item = file_from_dir(item, return_1st=True)
        item: pdcm.FileDataset = pdcm.read_file(item)
    return item[0x0020, 0x000d].value

def get_SeriesDescription(item: Union[str, pdcm.FileDataset]):
    if isinstance(item, str):
        item = file_from_dir(item, return_1st=True)
        item: pdcm.FileDataset = pdcm.read_file(item)
    return item[0x0008, 0x103e].value

def get_ZXY_resolution(item: Union[str, pdcm.FileDataset]):
    if isinstance(item, str):
        item = file_from_dir(item, return_1st=True)
        item: pdcm.FileDataset = pdcm.read_file(item)
    xy_res = [float(val) for val in item[0x0028, 0x0030].value]
    z_res = [float(item[0x0018, 0x0050].value)]
    return z_res + xy_res

# --------------------------------
# Searchable Classes
# --------------------------------

class SearchDict(dict):
    def __init__(self, base: dict[str, Any] = None, **kwargs):
        if isinstance(base, dict):
            for key, item in base.items(): self.__setitem__(key.upper(), item)
        self.find_conflicts()

    def find_conflicts(self):
        """
        Method to see if any keys contain each other to avoid matching condlicts.
        Example: key1 = "LA", key2 = "LADA"
            key1 also exists in key2 so when searching for "LA", this will flag the key to also consider other matches
        """
        keys = list(self.keys())
        self.conf = {k: [] for k in keys}

        # For each key, see if any other key has a competing match
        for i in range(len(keys)):
            k = keys[i]
            _list = keys.copy()
            _list.pop(i)

            for _k in _list:
                if k in _k: self.conf[k].append(_k)
    
    def __getitem__(self, k:str) -> Any:
        k = k.upper()

        # For all possible keys
        for kk in self.keys():
            # Check if kk is a candidate
            if not kk in k: continue
            conflicts = self.conf[kk]
            # if kk has a known conflic, check other options
            if conflicts:
                matched = True
                for c in conflicts:
                    if c in k: matched = False
                if matched: return super().__getitem__(kk)

            else: return super().__getitem__(kk)
        raise KeyError(f'Key {k} not found in search dictionary')
    
class SearchList:
    def __init__(self, search_list: list[str]):
        self.list = [k.upper() for k in search_list]
        self.find_conflicts()
    
    def find_conflicts(self):
        keys = self.list
        self.conf = {k: [] for k in keys}

        # For each key, see if any other key has a competing match
        for i in range(len(keys)):
            k = keys[i]
            _list = keys.copy()
            _list.pop(i)

            for _k in _list:
                if k in _k: self.conf[k].append(_k)

    def __call__(self, k: str):
        k = k.upper()

        for kk in self.list:
            # Check if kk is a candidate
            if not kk in k: continue
            conflicts = self.conf[kk]
            # if kk has a known conflic, check other options
            if conflicts:
                matched = True
                for c in conflicts:
                    if c in k: matched = False
                if matched: return True

            else: return True
        return False


# --------------------------------
# Getting the structures from the rtstruct files
# --------------------------------

def get_masks_from_dicom(dcm_img: sitk.Image, dcm_rts: pdcm.FileDataset, required_names: Optional[list[str]] = None, verbose: bool = False) -> dict[str, np.ndarray]:
    struct_point_sequence = {cs.ReferencedROINumber: cs for cs in dcm_rts.ROIContourSequence}
    struct_list = []
    name_list = []

    if required_names is None: name_search = lambda name: True
    else: name_search = SearchList(required_names)
    
    for struct_ds in dcm_rts.StructureSetROISequence:
        img_blank = np.zeros(dcm_img.GetSize()[::-1], dtype=np.uint8)

        # optional- only include required names
        struct_name = "_".join(struct_ds.ROIName.split())
        if not name_search(struct_name): 
            if verbose: print(f'{struct_name} not found in name search, skipping...')
            continue

        struct_index = struct_ds.ROINumber

        if not struct_index in struct_point_sequence: continue
        if not hasattr(struct_point_sequence[struct_index], 'ContourSequence'): continue
        if not struct_point_sequence[struct_index].ContourSequence[0].ContourGeometricType == "CLOSED_PLANAR": continue

        # Each structure is stored as a list of 1D arrays. 1 Array per slice and each array holds x, y, z coordinates in the form [X0, Y0, Z0, X1, Y1, Z1, ...]
        for sl in range(len(struct_point_sequence[struct_index].ContourSequence)):

            # Pull the point cloud for this slice [x0, y0, z0, x1, y1, z1, ...]
            data = struct_point_sequence[struct_index].ContourSequence[sl].ContourData
            struct_slice_contour_data = np.array(data, dtype=np.double)

            # Reshape it into [[x0, y0, z0], ...]
            vertex_arr_physical = struct_slice_contour_data.reshape(struct_slice_contour_data.shape[0] // 3, 3)

            # Transform points into array indicies
            point_arr = np.array([dcm_img.TransformPhysicalPointToIndex(i) for i in vertex_arr_physical]).T
            
            # Pull out ([x0, x1, ...], [y0, y1, ...], z)
            [x_vertex_arr_image, y_vertex_arr_image] = point_arr[[0, 1]]
            z_index = point_arr[2][0]

            # Convert to a filled_region and fill in slice
            slice_arr = np.zeros(img_blank.shape[-2:], dtype=np.uint8)
            filled_indices_x, filled_indices_y = draw.polygon(x_vertex_arr_image, y_vertex_arr_image, shape=slice_arr.shape)
            slice_arr[filled_indices_y, filled_indices_x] = 1
            img_blank[z_index] += slice_arr

        struct_image = sitk.GetImageFromArray(1 * (img_blank > 0))
        struct_image.CopyInformation(dcm_img)

        struct_list.append(sitk.Cast(struct_image, sitk.sitkUInt8))
        name_list.append(struct_name)

    return struct_list, name_list

def match_order_rtstruct(structures: list, names: list[str], matching_dict: Union[SearchDict, dict], name_map: Optional[dict[int, Any]] = None):
    matching_dict = SearchDict(matching_dict)
    if name_map is None: name_map = {i: None for i in range(len(structures))}
    
    struture_set = {i: None for i in name_map.keys()}
    for name, structure in zip(names, structures):
        id = matching_dict[name]
        struture_set[id] = structure
    return struture_set

def get_rts(dicom_img: str, dicom_rtstruct: str, search_dict: Union[dict, SearchDict], name_map: Optional[dict[int, Any ]] = None, verbose: bool = False):
    dicom_img: sitk.Image = get_image(dicom_img)
    dicom_rtstruct = pdcm.read_file(file_from_dir(dicom_rtstruct), force=True)

    struct_list, name_list = get_masks_from_dicom(dicom_img, dicom_rtstruct, required_names=search_dict.keys(), verbose=verbose)
    struct_set = match_order_rtstruct(struct_list, name_list, matching_dict=search_dict, name_map=name_map)

    empty_class = np.zeros(dicom_img.GetSize()[::-1])
    structs = [sitk.GetArrayFromImage(_s) if (_s := struct_set[i]) else empty_class for i in sorted(struct_set.keys())]
    
    return np.stack(structs)

# --------------------------------
# Processing Function
# --------------------------------
# When altering the image, return the amoun that it was changed for tracing

# crop to center of mass with pre-defined image size:
def CenterOfMassCropToSize(src_arr: np.ndarray, size: Union[int, list[int]], channeled_src: bool = True, aux_arr: tuple[np.ndarray] = tuple([])):

    # If channeled, collapse dimension
    csrc = src_arr.sum(0) if channeled_src else src_arr

    # Determine the size to crop to
    if isinstance(size, int): size = [size for _ in csrc.shape]
    assert len(size) == len(csrc.shape)

    # Finding center of mass of the src
    csrc = (csrc > 0).astype(int)
    com = [int(c) for c in ndimg.center_of_mass(csrc)]

    # Determine the size extents of the ROI
    size_low = [m.floor(s / 2) for s in size]
    size_high = [m.ceil(s / 2) for s in size]
    low_extent = [c - sl for c, sl in zip(com, size_low)]
    high_extent = [c + sh for c, sh in zip(com, size_high)]

    # Determining the low and high padding / slicing needed
    low_pad = [0 if c > 0 else abs(c) for c in low_extent]
    low_slice = [c if c > 0 else 0 for c in low_extent]

    high_lim = csrc.shape
    high_pad = [0 if c < hl else c - hl for c, hl in zip(high_extent, high_lim)]
    high_slice = [hl - c if c < hl else hl for c, hl in zip(high_extent, high_lim)]

    # Calculating the amount of padding or slicing needed
    padding = [[l, h] for l, h in zip(low_pad, high_pad)]
    slices = tuple([slice(l, -h) for l, h in zip(low_slice, high_slice)])
    slice_indexs = [[l, -h] for l, h in zip(low_slice, high_slice)]

    # Recording the slicing and padding amounts
    info = {'slices': slice_indexs, 'padding': padding}

    # Applying the slicing and padding to all the volumes
    s_pad = [[0, 0]] + padding if channeled_src else padding
    s_slice = tuple([slice(None)]) + slices if channeled_src else slices

    src_arr = src_arr[s_slice]
    src_arr = np.pad(src_arr, pad_width=s_pad)

    aux_arr = list(aux_arr)
    for i in range(len(aux_arr)):
        aux_arr[i] = aux_arr[i][slices]
        aux_arr[i] = np.pad(aux_arr[i], pad_width=padding)
    aux_arr = tuple(aux_arr)
    return tuple([src_arr]) + aux_arr + tuple([info])


def BoundingBoxCropToSize(src_arr: np.ndarray, channeled_src: bool = True, pad_amount: int = 0, aux_arr : tuple[np.ndarray] = tuple([]), minimum_size: Optional[Union[int, list[int]]] = None):
    # If channeled, collapse dimension
    csrc = src_arr.sum(0) if channeled_src else src_arr
    shape = csrc.shape
    bsrc = csrc > 0

    # Calculate the extents around the src and pad as necessary
    dims: list[np.ndarray] = np.meshgrid(*[np.arange(0, s, 1) for s in shape], indexing='ij')
    if isinstance(pad_amount, int): pad_amount = [pad_amount for _ in csrc.shape]
    pad_amount_low = [m.floor(p/2) for p in pad_amount]
    pad_amount_high = [m.ceil(p/2) for p in pad_amount]
    low_extent = [dim[bsrc].min() - pl for dim, pl in zip(dims, pad_amount_low)]
    high_extent = [dim[bsrc].max() + ph for dim, ph in zip(dims, pad_amount_high)]

    if minimum_size is not None:
        if isinstance(minimum_size, int): minimum_size = [minimum_size for _ in csrc.shape]
        sizes = [h - l for h, l in zip(high_extent, low_extent)]
        differences = [ms - s for s, ms in zip(sizes, minimum_size)]
        padding_needed = [d if d > 0 else 0 for d in differences]
        pad_amount_low = [m.floor(d/2) for d in padding_needed]
        pad_amount_high = [m.ceil(d/2) for d in padding_needed]

        low_extent = [le - pal for le, pal in zip(low_extent, pad_amount_low)]
        high_extent = [he + pah for he, pah in zip(high_extent, pad_amount_high)]

    # Determining the low and high padding / slicing needed
    low_pad = [0 if c > 0 else abs(c) for c in low_extent]
    low_slice = [c if c > 0 else 0 for c in low_extent]

    high_lim = csrc.shape
    high_pad = [0 if c < hl else c - hl for c, hl in zip(high_extent, high_lim)]
    high_slice = [hl - c if c < hl else 0 for c, hl in zip(high_extent, high_lim)]

    # high_extent = 174, high_lim = 160
    # (hl := 160) - (c := 174) = 1

    # Calculating the amount of padding or slicing needed
    padding = [[int(l), int(h)] for l, h in zip(low_pad, high_pad)]
    slices = tuple([slice(l, (-h if h != 0 else None)) for l, h in zip(low_slice, high_slice)])
    slice_indexs = [[int(l), int(-h)] for l, h in zip(low_slice, high_slice)]

    # Recording the slicing and padding amounts
    info = {'slicing_performed': slice_indexs, 'padding_performed': padding}

    # Applying the slicing and padding to all the volumes
    s_pad = [[0, 0]] + padding if channeled_src else padding
    s_slice = tuple([slice(None)]) + slices if channeled_src else slices

    src_arr = src_arr[s_slice]
    src_arr = np.pad(src_arr, pad_width=s_pad)

    if isinstance(aux_arr, np.ndarray): aux_arr = [aux_arr]
    aux_arr = list(aux_arr)
    for i in range(len(aux_arr)):
        aux_arr[i] = aux_arr[i][slices]
        aux_arr[i] = np.pad(aux_arr[i], pad_width=padding)
    aux_arr = tuple(aux_arr)
    return tuple([src_arr]) + aux_arr + tuple([info])

def resample(arr: np.ndarray, src_res: list[float], dst_res: list[float], order:int=0, **kwargs) -> np.ndarray:
    if len(arr.shape) != len(src_res) != len(dst_res): raise ValueError(f'Arr shape must match len of src_res and dst_res')
    return ndimg.zoom(arr, [x/y for x, y in zip(src_res, dst_res)], order=order, **kwargs)
    
def KScoreNormalize(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.mean()) / arr.std()
# --------------------------------
# Structure Definitions
# --------------------------------

Substructures: dict[str, int] = {
    'WholeHeart': 0,
    'WH': 0,
    'Heart': 0,

    'RA': 1, 
    'RT_Atrium': 1,
    'RT Atrium': 1,

    'LA': 2, 
    'LT_Atrium':2,
    'LT Atrium':2,

    'RV': 3, 
    'RT_Ventricle': 3,
    'RT Ventricle': 3,

    'LV': 4, 
    'LT_Ventricle': 4,
    'LT Ventricle': 4,

    'AA': 5, 
    'Asc_Aorta': 5,
    'Asc Aorta': 5,

    'SVC': 6, 
    'Sup_VC': 6,
    'Sup VC': 6,

    'IVC': 7, 
    'Inf_VC': 7,
    'Inf VC': 7,

    'PA': 8, 
    'Pulmonary_Artery': 8,
    'Pulmonary Artery': 8,

    'PVs': 9,
    'PV': 9, 
    'Pulmonary_Vein':9,
    'Pulmonary Vein':9,

    'LMCA': 10, 
    'LT_Main_Cor_Artery': 10,
    'LT Main Cor Artery': 10,
    'LCMA_ST10': 10,

    'LADA': 11,
    'LAD_Artery':11,
    'LAD Artery':11,

    'RCA': 12,
    'RT_Cor_Artery': 12,
    'RT Cor Artery': 12,

    'LCFX': 13,
    'CFX': 13,
    'LCX': 13,
    'CLFX': 13, # Misspelling
    'CFLX': 13, # Misspelling

    'V-AV': 14,
    'V-AA': 14, # Misspelling
    'V_AV': 14, # Misspelling

    'V-PV': 15,
    'V-PA': 15, # Misspelling

    'V-TV': 16,

    'V-MV': 17,

    'N-SA': 18,
    'N-SV': 18, # Misspelling
    
    'N-AV': 19,
    }

name_map = {
    0: 'WH',
    1: 'RA',
    2: 'LA',
    3: 'RV',
    4: 'LV',
    5: 'AA',
    6: 'SVC',
    7: 'IVC',
    8: 'PA',
    9: 'PV',
    10: 'LMCA',
    11: 'LADA',
    12: 'RCA',
    13: 'LCFX',
    14: 'V-AV',
    15: 'V-PV',
    16: 'V-TV',
    17: 'V-MV',
    18: 'N-SA',
    19: 'N-AV'
}