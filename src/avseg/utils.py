import cv2
import numpy as np


def reconstruct_disc_circle(disc_mask):
    """
    Input the segmented image and output the fitted circular mask image (of uint8 type, with values of 0 or 255)
    """
    if disc_mask.max() <= 1:
        disc_mask = (disc_mask * 255).astype(np.uint8)
    else:
        disc_mask = disc_mask.astype(np.uint8)

    contours, _ = cv2.findContours(
        disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        raise ValueError("The outline was not found. Skip this picture")

    largest_contour = max(contours, key=cv2.contourArea)

    # Fit the minimum circumscribed circle
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius = int(radius)

    circle_mask = np.zeros_like(disc_mask, dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, 1)

    return circle_mask


def get_avr(artery, vein, disk):
    artery = artery.astype(np.uint8)
    vein = vein.astype(np.uint8)
    disk = reconstruct_disc_circle(disk)
    final_artery_img = cv2.bitwise_and(artery, artery, mask=disk)
    final_vein_img = cv2.bitwise_and(vein, vein, mask=disk)

    num_labels_red, _, stats_red, _ = cv2.connectedComponentsWithStats(
        final_artery_img, 8, cv2.CV_32S
    )
    num_labels_blue, _, stats_blue, _ = cv2.connectedComponentsWithStats(
        final_vein_img, 8, cv2.CV_32S
    )
    red_arcs = []
    red_components = []
    for i in range(1, num_labels_red):
        area = stats_red[i, cv2.CC_STAT_AREA]
        red_arcs.append(int(area))
        red_components.append(stats_red[i])

    num_labels_blue, _, stats_blue, _ = cv2.connectedComponentsWithStats(
        final_vein_img, 8, cv2.CV_32S
    )
    blue_arcs = []
    blue_components = []
    for i in range(1, num_labels_blue):
        area = stats_blue[i, cv2.CC_STAT_AREA]
        blue_arcs.append(int(area))
        blue_components.append(stats_blue[i])

    red_arcs_sorted = sorted(red_arcs, reverse=True)[:4]
    blue_arcs_sorted = sorted(blue_arcs, reverse=True)[:4]

    red_top4_avg = np.mean(red_arcs_sorted) if red_arcs_sorted else 0
    blue_top4_avg = np.mean(blue_arcs_sorted) if blue_arcs_sorted else 0

    ratio = red_top4_avg / blue_top4_avg if blue_top4_avg > 0 else float("inf")
    return ratio
