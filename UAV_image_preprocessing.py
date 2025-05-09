import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def hsv_segmentation(input_path, output_path, filename):
    img = cv2.imread(os.path.join(input_path, filename))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([26, 21, 9])
    upper = np.array([78, 246, 254])
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imwrite(os.path.join(output_path, filename), mask)

def morph_open(input_path, output_path, filename):
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.imread(os.path.join(input_path, filename))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(output_path, filename), opened)

def morph_close(input_path, output_path, filename):
    kernel = np.ones((9, 9), np.uint8)
    img = cv2.imread(os.path.join(input_path, filename))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(output_path, filename), closed)

def get_large_mask_contours(image_path, bins=20, top_bins=17):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    counts, bin_edges = np.histogram(areas, bins=bins)
    threshold_area = bin_edges[-top_bins-1]
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= threshold_area]
    return img_gray, large_contours

def compute_centroids(contours):
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    return centroids

def draw_circle_for_largest_mask(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    largest = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return 0
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    points = largest.reshape(-1, 2)
    dists = np.sqrt((points[:, 0] - cx)**2 + (points[:, 1] - cy)**2)
    return int(np.max(dists))

def generate_circle_masks(image_path, centroids, radius, output_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for (cx, cy) in centroids:
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    cv2.imwrite(output_path, mask)

def crop_circular_regions(rgb_path, centroids, radius, prefix, output_folder):
    img = cv2.imread(rgb_path)
    h, w, _ = img.shape
    for i, (cx, cy) in enumerate(centroids, 1):
        x0, y0 = max(cx - radius, 0), max(cy - radius, 0)
        x1, y1 = min(cx + radius, w), min(cy + radius, h)
        crop = img[y0:y1, x0:x1]
        center = (cx - x0, cy - y0)
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        circular_crop = cv2.bitwise_and(crop, crop, mask=mask)
        out_path = os.path.join(output_folder, f"{prefix}_crop_{i}.jpg")
        cv2.imwrite(out_path, circular_crop)

# ---------------- Main Pipeline ----------------
if __name__ == "__main__":
    root = '.'
    raw_path = os.path.join(root, 'raw')
    r1 = os.path.join(root, 'r1')
    r2 = os.path.join(root, 'r2')
    r3 = os.path.join(root, 'r3')
    r6 = os.path.join(root, 'r6')
    r7 = os.path.join(root, 'r7')

    # Ensure all folders exist
    for path in [r1, r2, r3, r6, r7]:
        ensure_dir(path)

    # Process each image in raw/
    for filename in os.listdir(raw_path):
        if not filename.lower().endswith('.jpg'):
            continue

        print(f"\nProcessing: {filename}")

        # Step 1–3: Mask processing
        hsv_segmentation(raw_path, r1, filename)
        morph_open(r1, r2, filename)
        morph_close(r2, r3, filename)

        # Step 4–5: Centroid and radius
        mask_path = os.path.join(r3, filename)
        rgb_path = os.path.join(raw_path, filename)
        gray, large_contours = get_large_mask_contours(mask_path)
        centroids = compute_centroids(large_contours)
        radius = draw_circle_for_largest_mask(mask_path)
        radius_final = int(radius * 1.1)

        if not centroids or radius_final <= 0:
            print(f"Skipping {filename} due to no valid masks.")
            continue

        # Step 6: Generate circle mask
        circle_mask_path = os.path.join(r6, filename)
        generate_circle_masks(mask_path, centroids, radius_final, circle_mask_path)

        # Step 7: Crop and save RGB patches
        crop_circular_regions(rgb_path, centroids, radius_final, filename.split('.')[0], r7)

    print("\nBatch processing complete.")
