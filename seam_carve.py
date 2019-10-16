import numpy as np


def find_cum_energy(energy):
    cum_energy = np.zeros_like(energy)
    cum_energy[0, :] = energy[0, :]
    for i in range(1, cum_energy.shape[0]):
        '''
        for j in range(cum_energy.shape[1]):
            if j == 0:
                cum_energy[i, 0] += np.min(cum_energy[i - 1, 0:2])
            else:
                cum_energy[i, j] += np.min(cum_energy[i - 1, j - 1:j + 2])
        '''
        prev = cum_energy[i - 1, :]
        cum_energy[i, :] = prev
        cur = cum_energy[i, :]
        cur[1:] = np.minimum(cur[1:], np.roll(prev, 1)[1:])
        cur[:-1] = np.minimum(cur[:-1], np.roll(prev, -1)[:-1])
        cur += energy[i, :]
    return cum_energy


def seam_carve(img, mode, mask=None):
    if mode == 'horizontal shrink':
        Y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        grad_y, grad_x = np.gradient(Y, 0.5)
        grad_y[0, :] /= 2
        grad_y[-1, :] /= 2
        grad_x[:, 0] /= 2
        grad_x[:, -1] /= 2

        grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
        if mask is None:
            mask = np.zeros((img.shape[0], img.shape[1]))
        coef = img.shape[0] * img.shape[1] * 256
        grad = grad + coef * mask
        cum_energy = find_cum_energy(grad)
        #  carve_mask = np.ones_like(cum_energy, dtype=bool)
        carve_mask = np.full_like(cum_energy, True, dtype=bool)
        cur_arg_min = np.argmin(cum_energy[-1, :])
        carve_mask[-1, cur_arg_min] = False
        for i in range(carve_mask.shape[0] - 2, -1, -1):
            if cur_arg_min == 0:
                cur_arg_min = np.argmin(cum_energy[i, 0:2])
            else:
                cur_arg_min = np.argmin(cum_energy[i, cur_arg_min - 1:cur_arg_min + 2]) + cur_arg_min - 1
            carve_mask[i, cur_arg_min] = False
        carve_mask_3d = np.concatenate((carve_mask[:, :, np.newaxis],
                                        carve_mask[:, :, np.newaxis],
                                        carve_mask[:, :, np.newaxis]), axis=2)
        result = img[carve_mask_3d].reshape((img.shape[0], -1, 3))
        seam_mask = np.logical_not(carve_mask)
        new_mask = np.copy(mask)
        new_mask = new_mask[carve_mask].reshape((img.shape[0], -1))
    elif mode == 'vertical shrink':
        if mask is None:
            result, new_mask, seam_mask = seam_carve(np.transpose(img, (1, 0, 2)), 'horizontal shrink')
        else:
            result, new_mask, seam_mask = seam_carve(np.transpose(img, (1, 0, 2)), 'horizontal shrink', mask.T)
        result = np.transpose(result, (1, 0, 2))
        new_mask = new_mask.transpose()
        seam_mask = seam_mask.transpose()
    elif mode == 'horizontal expand':
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        Y = 0.299 * r + 0.587 * g + 0.114 * b

        grad_y, grad_x = np.gradient(Y, 0.5)
        grad_y[0, :] /= 2
        grad_y[-1, :] /= 2
        grad_x[:, 0] /= 2
        grad_x[:, -1] /= 2

        grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
        if mask is None:
            mask = np.zeros((img.shape[0], img.shape[1]))
        coef = img.shape[0] * img.shape[1] * 256
        grad = grad + coef * mask
        cum_energy = find_cum_energy(grad)
        #  carve_mask = np.ones_like(cum_energy, dtype=bool)
        seam_mask = np.full_like(cum_energy, False, dtype=bool)
        idx_to_add = []
        values_to_add = [[], [], []]
        cur_arg_min = np.argmin(cum_energy[-1, :])
        seam_mask[-1, cur_arg_min] = True
        idx_to_add.append(img.shape[1] * (img.shape[0] - 1) + cur_arg_min + 1)
        if cur_arg_min != img.shape[1] - 1:
            values_to_add[0].append(r[-1, cur_arg_min] // 2 + r[-1, cur_arg_min + 1] // 2)
            values_to_add[1].append(g[-1, cur_arg_min] // 2 + g[-1, cur_arg_min + 1] // 2)
            values_to_add[2].append(b[-1, cur_arg_min] // 2 + b[-1, cur_arg_min + 1] // 2)
        else:
            values_to_add[0].append(r[-1, cur_arg_min])
            values_to_add[1].append(g[-1, cur_arg_min])
            values_to_add[2].append(b[-1, cur_arg_min])
        for i in range(seam_mask.shape[0] - 2, -1, -1):
            if cur_arg_min == 0:
                cur_arg_min = np.argmin(cum_energy[i, 0:2])
            else:
                cur_arg_min = np.argmin(cum_energy[i, cur_arg_min - 1:cur_arg_min + 2]) + cur_arg_min - 1
            seam_mask[i, cur_arg_min] = True
            idx_to_add.append(img.shape[1] * i + cur_arg_min + 1)
            if cur_arg_min != img.shape[1] - 1:
                values_to_add[0].append(r[i, cur_arg_min] // 2 + r[i, cur_arg_min + 1] // 2)
                values_to_add[1].append(g[i, cur_arg_min] // 2 + g[i, cur_arg_min + 1] // 2)
                values_to_add[2].append(b[i, cur_arg_min] // 2 + b[i, cur_arg_min + 1] // 2)
            else:
                values_to_add[0].append(r[i, cur_arg_min])
                values_to_add[1].append(g[i, cur_arg_min])
                values_to_add[2].append(b[i, cur_arg_min])

        new_r = np.insert(r, idx_to_add, values_to_add[0]).reshape(img.shape[0], -1)
        new_g = np.insert(g, idx_to_add, values_to_add[1]).reshape(img.shape[0], -1)
        new_b = np.insert(b, idx_to_add, values_to_add[2]).reshape(img.shape[0], -1)
        result = np.concatenate((new_r[:, :, np.newaxis], new_g[:, :, np.newaxis], new_b[:, :, np.newaxis]), axis=2)
        new_mask = np.copy(mask)
        new_mask[seam_mask] += 1
        new_mask = np.insert(new_mask, idx_to_add, 0).reshape(img.shape[0], -1)
    elif mode == 'vertical expand':
        if mask is None:
            result, new_mask, seam_mask = seam_carve(np.transpose(img, (1, 0, 2)), 'horizontal expand')
        else:
            result, new_mask, seam_mask = seam_carve(np.transpose(img, (1, 0, 2)), 'horizontal expand', mask.transpose())
        result = np.transpose(result, (1, 0, 2))
        new_mask = new_mask.transpose()
        seam_mask = seam_mask.transpose()
    return result, new_mask, seam_mask

