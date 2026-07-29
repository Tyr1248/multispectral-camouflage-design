import numpy as np

from color_calculate.colors import color_matching_functions
from color_calculate.colors import illuminants


def get_cmfs(bx, str_color_space="cie1931"):
    assert isinstance(bx, np.ndarray)

    if str_color_space == "cie1931":
        cmfs = color_matching_functions.cmfs_cie1931
    else:
        raise ValueError

    cmfs = np.array(cmfs)
    assert cmfs.shape[1] == 4
    assert np.min(cmfs[:, 0]) <= np.min(bx)
    assert np.max(bx) <= np.max(cmfs[:, 0])

    cmfs_interpolated = np.concatenate(
        [
            bx[..., np.newaxis],
            np.interp(bx, cmfs[:, 0], cmfs[:, 1])[..., np.newaxis],
            np.interp(bx, cmfs[:, 0], cmfs[:, 2])[..., np.newaxis],
            np.interp(bx, cmfs[:, 0], cmfs[:, 3])[..., np.newaxis],
        ],
        axis=1,
    )

    return cmfs_interpolated


def get_illuminant(bx, str_illuminant="d65"):
    assert isinstance(bx, np.ndarray)

    if str_illuminant == "d65":
        illuminant = illuminants.illuminant_d65
    else:
        raise ValueError

    illuminant = np.array(illuminant)
    assert illuminant.shape[1] == 2
    assert np.min(illuminant[:, 0]) <= np.min(bx)
    assert np.max(bx) <= np.max(illuminant[:, 0])

    illuminant_interpolated = np.concatenate(
        [
            bx[..., np.newaxis],
            np.interp(bx, illuminant[:, 0], illuminant[:, 1])[..., np.newaxis],
        ],
        axis=1,
    )

    return illuminant_interpolated
