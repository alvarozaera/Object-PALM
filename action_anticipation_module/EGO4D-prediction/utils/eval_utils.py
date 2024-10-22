import editdistance

import numpy as np 

def edit_distance(preds, labels):
    """
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
    Lowest among K predictions
    """
    N, Z, K = preds.shape
    dists = []
    for n in range(N):
        dist = min([editdistance.eval(preds[n, :, k], labels[n])/Z for k in range(K)])
        dists.append(dist)
    return np.mean(dists)

def AUED(preds, labels):
    N, Z, K = preds.shape
    ED = np.vstack(
        [edit_distance(preds[:, :z], labels[:, :z]) for z in range(1, Z + 1)]
    )
    AUED = np.trapz(y=ED, axis=0) / (Z - 1)

    output = {"AUED": AUED}
    output.update({f"ED_{z}": ED[z] for z in range(Z)})
    return output

def edit_distance_w_arg(preds, labels):
    """
    Damerau–Levenshtein edit distance from: https://github.com/gfairchild/pyxDamerauLevenshtein
    Lowest among K predictions
    """
    N, Z, K = preds.shape
    dists = []
    for n in range(N):
        ed_list = [editdistance.eval(preds[n, :, k], labels[n])/Z for k in range(K)]
        dist = min(ed_list)
        dists.append(dist)
    return np.mean(dists), np.argmin(ed_list)


def AUED_w_arg(preds, labels):
    N, Z, K = preds.shape
    #print(preds.shape, labels.shape)

    ele_list = []
    argmin_list = []
    for z in range(1, Z + 1):
        ele, argmin = edit_distance_w_arg(preds[:, :z], labels[:, :z]) 
        ele_list.append(ele)
        argmin_list.append(argmin)

    ED = np.vstack(ele_list)
    AUED = np.trapz(y=ED, axis=0) / (Z - 1)

    output = {"AUED": AUED}
    output.update({f"ED_{z}": ED[z] for z in range(Z)})
    #print(output, argmin_list)
    return output, argmin_list[-1]