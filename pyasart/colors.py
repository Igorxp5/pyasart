"""
The source code of this file is based on https://github.com/colour-science/colour project. 

# LICENSE

Copyright 2013 Colour Developers

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
"""

import numpy as np

from typing import Union

from numpy.typing import ArrayLike, NDArray

DTypeFloat = Union[np.float16, np.float32, np.float64]
NDArrayFloat = NDArray[DTypeFloat]

MATRIX_SRGB_to_XYZ = np.array(
    [[ 0.4124,  0.3576,  0.1805],
     [ 0.2126,  0.7152,  0.0722],
     [ 0.0193,  0.1192,  0.9505]]
)

MATRIX_XYZ_to_SRGB = np.array(
    [[ 3.2406, -1.5372, -0.4986],
     [-0.9689,  1.8758,  0.0415],
     [ 0.0557, -0.204 ,  1.057 ]]
)

MATRIX_SRGB_XYZ_M_CAT = np.array(
    [[  1.00000000e+00,  -3.29597460e-17,  -2.77555756e-17],
     [  4.05762663e-17,   1.00000000e+00,   0.00000000e+00],
     [ -1.30104261e-18,   0.00000000e+00,   1.00000000e+00]]
)



def sRGB_to_XYZ(SRGB: ArrayLike) -> NDArrayFloat:
    XYZ = np.where(
        0.040449935999999999 >= SRGB,
        SRGB / 12.92,
        (np.sign((SRGB + 0.055) / 1.055) * np.abs((SRGB + 0.055) / 1.055) ** 2.4),
    )
    XYZ = np.einsum("...ij,...j->...i", MATRIX_SRGB_to_XYZ, XYZ)
    return np.einsum("...ij,...j->...i", MATRIX_SRGB_XYZ_M_CAT, XYZ)


def XYZ_to_sRGB(XYZ: ArrayLike) -> NDArrayFloat:
    XYZ = np.einsum("...ij,...j->...i", MATRIX_SRGB_XYZ_M_CAT, XYZ)
    SRGB = np.einsum("...ij,...j->...i", MATRIX_XYZ_to_SRGB, XYZ)
    return np.where(SRGB <= 0.0031308, SRGB * 12.92, 1.055 * (np.sign(SRGB) * np.abs(SRGB) ** (1 / 2.4)) - 0.055)


def XYZ_to_Lab(XYZ: ArrayLike) -> NDArrayFloat:
    XYZ_n = 0.95045592705167159, 1.0, 1.0890577507598784
    f_XYZ_n = [0, 0, 0]

    for i in range(3):
        W = XYZ[:,i]
        W_W_n = W / XYZ_n[i]

        f_XYZ_n[i] = np.where(
            W_W_n > (24 / 116) ** 3,
            W_W_n ** (1 / 3),
            (841 / 108) * W_W_n + 16 / 116,
        )

    L = 116 * f_XYZ_n[1] - 16
    a = 500 * (f_XYZ_n[0] - f_XYZ_n[1])
    b = 200 * (f_XYZ_n[1] - f_XYZ_n[2])

    return np.column_stack([L, a, b])


def Lab_to_XYZ(Lab: ArrayLike) -> NDArrayFloat:
    L, a, b = np.hsplit(Lab, 3)

    f_Y_Y_n = (L + 16) / 116
    f_X_X_n = a / 500 + f_Y_Y_n
    f_Z_Z_n = f_Y_Y_n - b / 200

    XYZ = [0, 0, 0]
    XYZ_n = 0.95045592705167159, 1.0, 1.0890577507598784

    for i, f_W_W_n in enumerate((f_X_X_n, f_Y_Y_n, f_Z_Z_n)):
        XYZ[i] = np.where(
            f_W_W_n > 24 / 116,
            XYZ_n[i] * f_W_W_n**3,
            XYZ_n[i] * (f_W_W_n - 16 / 116) * (108 / 841),
        )
    return np.column_stack(XYZ)


def RGB_to_Lab(rgb) -> NDArray[np.float32]:
    srgb = rgb / 255.0
    xyz = sRGB_to_XYZ(srgb)
    return XYZ_to_Lab(xyz).astype(np.float32)


def Lab_to_RGB(lab) -> NDArray[np.uint8]:
    xyz = Lab_to_XYZ(lab)
    srgb = np.clip(XYZ_to_sRGB(xyz), a_min=0, a_max=1)
    return np.round(srgb * 255).astype(np.uint8)


def delta_E_CIE2000(
    Lab_1: ArrayLike, Lab_2: ArrayLike, textiles: bool = False
) -> NDArrayFloat:
    L_1, a_1, b_1 = Lab_1[:,0], Lab_1[:,1], Lab_1[:,2]
    L_2, a_2, b_2 = Lab_2[:,0], Lab_2[:,1], Lab_2[:,2]

    k_L = 2 if textiles else 1
    k_C = 1
    k_H = 1

    C_1_ab = np.hypot(a_1, b_1)
    C_2_ab = np.hypot(a_2, b_2)

    C_bar_ab = (C_1_ab + C_2_ab) / 2
    C_bar_ab_7 = C_bar_ab**7

    G = 0.5 * (1 - np.sqrt(C_bar_ab_7 / (C_bar_ab_7 + 25**7)))

    a_p_1 = (1 + G) * a_1
    a_p_2 = (1 + G) * a_2

    C_p_1 = np.hypot(a_p_1, b_1)
    C_p_2 = np.hypot(a_p_2, b_2)

    h_p_1 = np.where(
        np.logical_and(b_1 == 0, a_p_1 == 0),
        0,
        np.degrees(np.arctan2(b_1, a_p_1)) % 360,
    )
    h_p_2 = np.where(
        np.logical_and(b_2 == 0, a_p_2 == 0),
        0,
        np.degrees(np.arctan2(b_2, a_p_2)) % 360,
    )

    delta_L_p = L_2 - L_1

    delta_C_p = C_p_2 - C_p_1

    h_p_2_s_1 = h_p_2 - h_p_1
    C_p_1_m_2 = C_p_1 * C_p_2
    delta_h_p = np.select(
        [
            C_p_1_m_2 == 0,
            np.fabs(h_p_2_s_1) <= 180,
            h_p_2_s_1 > 180,
            h_p_2_s_1 < -180,
        ],
        [
            0,
            h_p_2_s_1,
            h_p_2_s_1 - 360,
            h_p_2_s_1 + 360,
        ],
    )

    delta_H_p = 2 * np.sqrt(C_p_1_m_2) * np.sin(np.deg2rad(delta_h_p / 2))

    L_bar_p = (L_1 + L_2) / 2

    C_bar_p = (C_p_1 + C_p_2) / 2

    a_h_p_1_s_2 = np.fabs(h_p_1 - h_p_2)
    h_p_1_a_2 = h_p_1 + h_p_2
    h_bar_p = np.select(
        [
            C_p_1_m_2 == 0,
            a_h_p_1_s_2 <= 180,
            np.logical_and(a_h_p_1_s_2 > 180, h_p_1_a_2 < 360),
            np.logical_and(a_h_p_1_s_2 > 180, h_p_1_a_2 >= 360),
        ],
        [
            h_p_1_a_2,
            h_p_1_a_2 / 2,
            (h_p_1_a_2 + 360) / 2,
            (h_p_1_a_2 - 360) / 2,
        ],
    )

    T = (
        1
        - 0.17 * np.cos(np.deg2rad(h_bar_p - 30))
        + 0.24 * np.cos(np.deg2rad(2 * h_bar_p))
        + 0.32 * np.cos(np.deg2rad(3 * h_bar_p + 6))
        - 0.20 * np.cos(np.deg2rad(4 * h_bar_p - 63))
    )

    delta_theta = 30 * np.exp(-(((h_bar_p - 275) / 25) ** 2))

    C_bar_p_7 = C_bar_p**7
    R_C = 2 * np.sqrt(C_bar_p_7 / (C_bar_p_7 + 25**7))

    L_bar_p_2 = (L_bar_p - 50) ** 2
    S_L = 1 + ((0.015 * L_bar_p_2) / np.sqrt(20 + L_bar_p_2))

    S_C = 1 + 0.045 * C_bar_p

    S_H = 1 + 0.015 * C_bar_p * T

    R_T = -np.sin(np.deg2rad(2 * delta_theta)) * R_C

    d_E = np.sqrt(
        (delta_L_p / (k_L * S_L)) ** 2
        + (delta_C_p / (k_C * S_C)) ** 2
        + (delta_H_p / (k_H * S_H)) ** 2
        + R_T * (delta_C_p / (k_C * S_C)) * (delta_H_p / (k_H * S_H))
    )

    return d_E
