from skimage.measure import label
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
from scipy.spatial.distance import directed_hausdorff as hausdorff
from scipy.ndimage.measurements import center_of_mass
from os.path import join
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_metrics(pred, gt, names):
    """
    Computes metrics specified by names between predicted label and groundtruth label.
    """

    gt_labeled = label(gt)
    pred_labeled = label(pred)

    gt_binary = gt_labeled.copy()
    pred_binary = pred_labeled.copy()
    gt_binary[gt_binary > 0] = 1
    pred_binary[pred_binary > 0] = 1
    gt_binary, pred_binary = gt_binary.flatten(), pred_binary.flatten()

    results = {}

    # pixel-level metrics
    if 'acc' in names:
        results['acc'] = accuracy_score(gt_binary, pred_binary)
    if 'roc' in names:
        results['roc'] = roc_auc_score(gt_binary, pred_binary)
    if 'F1' in names:  # pixel-level F1
        results['F1'] = f1_score(gt_binary, pred_binary)
    if 'p_recall' in names:  # pixel-level F1
        results['p_recall'] = recall_score(gt_binary, pred_binary)
    if 'p_precision' in names:  # pixel-level F1
        results['p_precision'] = precision_score(gt_binary, pred_binary)

    # object-level metrics
    if 'aji' in names:
        results['aji'] = get_fast_aji(gt_labeled, pred_labeled)
    if 'pq' in names:
        r, _ = get_fast_pq(gt_labeled, pred_labeled)
        results['dq'], results['sq'], results['pq'] = r[0], r[1], r[2]
    if 'haus' in names:
        results['dice'], results['iou'], results['haus'] = accuracy_object_level(pred_labeled, gt_labeled, True)
    results['Adice'] = instance_Dice(pred_labeled, gt_labeled)
    return results

def instance_Dice(true, pred):
    """
    ADice metric from CNSeg.
    """
    true = np.copy(true)  # ? do we need this
    mask = true
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    if len(pred_id_list)==1:
        aji_score=0.
        return aji_score

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    for pred_id in pred_id_list[1:]:
        p_mask=pred_masks[pred_id]
        pred_true_overlap=mask[p_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for true_id in pred_true_overlap_id :
            if true_id == 0:  # ignore
                continue
            t_mask = true_masks[true_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total
    pairwise_iou = 2*pairwise_inter / (pairwise_union + 1.0e-6)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()
    # return overall_inter,overall_union

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    dice_score = 2*overall_inter / overall_union

    return dice_score

def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
    over-penalisation similar to DICE2.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.

    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    if len(pred_id_list)==1:
        aji_score=0.
        # print("111111")
        return aji_score

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        if true_id>=len(true_masks):
            continue
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    # print(pairwise_iou)
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        if true_id>=len(true_masks):
            continue
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        if pred_id>=len(pred_masks):
            continue
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    return aji_score

def accuracy_object_level(pred, gt, hausdorff_flag=True):
    """ Compute the object-level metrics between predicted and
    groundtruth: dice, iou, hausdorff """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = label(pred, connectivity=2)
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = label(gt, connectivity=2)
    Ng = len(np.unique(gt_labeled)) - 1

    # --- compute dice, iou, hausdorff --- #
    pred_objs_area = np.sum(pred_labeled>0)  # total area of objects in image
    gt_objs_area = np.sum(gt_labeled>0)  # total area of objects in groundtruth gt

    # compute how well groundtruth object overlaps its segmented object
    dice_g = 0.0
    iou_g = 0.0
    hausdorff_g = 0.0
    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_parts = gt_i * pred_labeled

        # get intersection objects numbers in image
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        gamma_i = float(np.sum(gt_i)) / gt_objs_area

        # show_figures((pred_labeled, gt_i, overlap_parts))

        if obj_no.size == 0:   # no intersection object
            dice_i = 0
            iou_i = 0

            # find nearest segmented object in hausdorff distance
            if hausdorff_flag:
                min_haus = 1e3

                # find overlap object in a window [-50, 50]
                pred_cand_indices = find_candidates(gt_i, pred_labeled)

                for j in pred_cand_indices:
                    pred_j = np.where(pred_labeled == j, 1, 0)
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_i)
                    haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                    if haus_tmp < min_haus:
                        min_haus = haus_tmp
                haus_i = min_haus
        else:
            # find max overlap object
            obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
            seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
            pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object

            overlap_area = np.max(obj_areas)  # overlap area

            dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
            iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

            # compute hausdorff distance
            if hausdorff_flag:
                seg_ind = np.argwhere(pred_i)
                gt_ind = np.argwhere(gt_i)
                haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_g += gamma_i * dice_i
        iou_g += gamma_i * iou_i
        if hausdorff_flag:
            hausdorff_g += gamma_i * haus_i

    # compute how well segmented object overlaps its groundtruth object
    dice_s = 0.0
    iou_s = 0.0
    hausdorff_s = 0.0
    for j in range(1, Ns + 1):
        pred_j = np.where(pred_labeled == j, 1, 0)
        overlap_parts = pred_j * gt_labeled

        # get intersection objects number in gt
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((pred_j, gt_labeled, overlap_parts))

        sigma_j = float(np.sum(pred_j)) / pred_objs_area
        # no intersection object
        if obj_no.size == 0:
            dice_j = 0
            iou_j = 0

            # find nearest groundtruth object in hausdorff distance
            if hausdorff_flag:
                min_haus = 1e3

                # find overlap object in a window [-50, 50]
                gt_cand_indices = find_candidates(pred_j, gt_labeled)

                for i in gt_cand_indices:
                    gt_i = np.where(gt_labeled == i, 1, 0)
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_i)
                    haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                    if haus_tmp < min_haus:
                        min_haus = haus_tmp
                haus_j = min_haus
        else:
            # find max overlap gt
            gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
            gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
            gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

            overlap_area = np.max(gt_areas)  # overlap area

            dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
            iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

            # compute hausdorff distance
            if hausdorff_flag:
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_j)
                haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_s += sigma_j * dice_j
        iou_s += sigma_j * iou_j
        if hausdorff_flag:
            hausdorff_s += sigma_j * haus_j

    return (dice_g + dice_s) / 2, (iou_g + iou_s) / 2, (hausdorff_g + hausdorff_s) / 2

def find_candidates(obj_i, objects_labeled, radius=50):
    """
    find object indices in objects_labeled in a window centered at obj_i
    when computing object-level hausdorff distance

    """
    if radius > 400:
        return np.array([])

    h, w = objects_labeled.shape
    x, y = center_of_mass(obj_i)
    x, y = int(x), int(y)
    r1 = x-radius if x-radius >= 0 else 0
    r2 = x+radius if x+radius <= h else h
    c1 = y-radius if y-radius >= 0 else 0
    c2 = y+radius if y+radius < w else w
    indices = np.unique(objects_labeled[r1:r2, c1:c2])
    indices = indices[indices != 0]

    if indices.size == 0:
        indices = find_candidates(obj_i, objects_labeled, 2*radius)

    return indices

def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = (tp + 1.0e-6) / (tp + 0.5 * fp + 0.5 * fn + 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = (paired_iou.sum() + 1.0e-6) / (tp + 1.0e-6)
    # print(len(true_id_list) - 1, len(pred_id_list) - 1, [dq, sq, dq * sq])
    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]
