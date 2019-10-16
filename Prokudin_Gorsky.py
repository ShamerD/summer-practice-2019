import numpy as np
from math import sqrt


def align(img, g_coord):
    third = len(img) // 3
    b, g, r = img[:third, :], img[third:(2 * third), :], img[(2 * third):(3 * third), :]

    def cut(img):
        y = round(img.shape[0] * 0.05)
        x = round(img.shape[1] * 0.05)
        return img[y:-y, x:-x]

    red = cut(r)
    green = cut(g)
    blue = cut(b)

    def MSE(img1, img2, shift):
        img1_cross = img1
        img2_cross = img2
        height = img1.shape[0] - abs(2 * shift[0])
        width = img1.shape[1] - abs(2 * shift[1])
        if shift[0] < 0:
            img1_cross = img1[:shift[0], :]
            img2_cross = img2[-shift[0]:, :]
        elif shift[0] > 0:
            img1_cross = img1[shift[0]:, :]
            img2_cross = img2[:-shift[0]:, :]
        if shift[1] < 0:
            img1_cross = img1_cross[:, :shift[1]]
            img2_cross = img2_cross[:, -shift[1]:]
        elif shift[1] > 0:
            img1_cross = img1_cross[:, shift[1]:]
            img2_cross = img2_cross[:, :-shift[1]]
        return (1 / (height * width)) * np.sum((img1_cross - img2_cross) ** 2)

    def cross_corr(img1, img2, shift):
        img1_cross = img1
        img2_cross = img2
        if shift[0] < 0:
            img1_cross = img1[:shift[0], :]
            img2_cross = img2[-shift[0]:, :]
        elif shift[0] > 0:
            img1_cross = img1[shift[0]:, :]
            img2_cross = img2[:-shift[0]:, :]
        if shift[1] < 0:
            img1_cross = img1_cross[:, :shift[1]]
            img2_cross = img2_cross[:, -shift[1]:]
        elif shift[1] > 0:
            img1_cross = img1_cross[:, shift[1]:]
            img2_cross = img2_cross[:, :-shift[1]]
        return (np.sum(img1_cross * img2_cross)) / (sqrt(np.sum(img1_cross ** 2)) * sqrt(np.sum(img2_cross ** 2)))

    def find_shift(color_fixed, color_unfixed):
        if color_fixed.shape[0] > 500 or color_fixed.shape[1] > 500:
            cur_shift = 2 * find_shift(color_fixed[::2, ::2], color_unfixed[::2, ::2])
            cur_m = MSE(color_fixed, color_unfixed, (cur_shift[0], cur_shift[1]))
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    mse = MSE(color_fixed, color_unfixed, (i + cur_shift[0], j + cur_shift[1]))
                    if mse < cur_m:
                        cur_m = mse
                        cur_shift = (i + cur_shift[0], j + cur_shift[1])
        else:
            cur_m = MSE(color_fixed, color_unfixed, (0, 0))
            cur_shift = (0, 0)
            for i in range(-15, 16):
                for j in range(-15, 16):
                    mse = MSE(color_fixed, color_unfixed, (i, j))
                    if mse < cur_m:
                        cur_m = mse
                        cur_shift = (i, j)
        shift = np.asarray(cur_shift)
        return shift

    def find_shift_cross(color_fixed, color_unfixed):
        cur_m = cross_corr(color_fixed, color_unfixed, (0, 0))
        cur_shift = (0, 0)
        for i in range(-15, 16):
            for j in range(-15, 16):
                cr_corr = cross_corr(color_fixed, color_unfixed, (i, j))
                if cr_corr > cur_m:
                    cur_m = cr_corr
                    cur_shift = (i, j)
        shift = np.asarray(cur_shift)
        return shift

    red_green_shift = find_shift(green, red)
    blue_green_shift = find_shift(green, blue)

    def find_borders(pic, to_first, to_second):
        res = pic
        if min(0, to_first[0], to_second[0]) == 0:
            res = res[max(0, to_first[0], to_second[0]):, :]
        else:
            res = res[max(0, to_first[0], to_second[0]):min(to_first[0], to_second[0]), :]
        if min(0, to_first[1], to_second[1]) == 0:
            res = res[:, max(0, to_first[1], to_second[1]):]
        else:
            res = res[:, max(0, to_first[1], to_second[1]):min(0, to_first[1], to_second[1])]
        return res

    def combine(r, g, b, r2g, b2g):
        g2r = -r2g
        g2b = -b2g
        b2r = b2g + g2r
        r2b = -b2r
        new_r = find_borders(r, g2r, b2r)
        new_g = find_borders(g, r2g, b2g)
        new_b = find_borders(b, r2b, g2b)
        return np.concatenate((new_r[:, :, np.newaxis], new_g[:, :, np.newaxis], new_b[:, :, np.newaxis]), axis=2)

    img_final = combine(red, green, blue, red_green_shift, blue_green_shift)

    b_coord = np.array(g_coord) - np.array([third, 0]) - blue_green_shift
    r_coord = np.array(g_coord) + np.array([third, 0]) - red_green_shift
    return img_final, tuple(b_coord), tuple(r_coord)

