"""遮罩版的接觸與在動判定。"""
import cv2
import numpy as np


def mask_motion_iou(o_t, o_p, hand):
    """同一個物件框在 t 與 t−Δ 的遮罩,扣掉手像素後的 IoU;兩邊都空 → 1(視為沒動)。"""
    a, b = o_t & ~hand, o_p & ~hand
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


def mask_contact(obj, hand, r_px):
    """手遮罩膨脹 r_px 像素後與物件遮罩重疊的像素數。"""
    if r_px > 0:
        k = 2 * int(round(r_px)) + 1
        hand = cv2.dilate(hand.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return int((obj & hand).sum())
