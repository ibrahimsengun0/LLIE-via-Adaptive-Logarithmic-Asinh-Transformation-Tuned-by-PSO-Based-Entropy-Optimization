import cv2
import os
import numpy as np
import torch
import pyiqa
from skimage.measure import shannon_entropy


def load_paired_images(eval_path):
    high_path = os.path.join(eval_path, "Normal")
    low_path = os.path.join(eval_path, "Low")

    if not os.path.exists(high_path) or not os.path.exists(low_path):
        raise FileNotFoundError("Normal or low folder not found in the specified evaluation path.")

    paired_images = []

    high_images = sorted(os.listdir(high_path))
    low_images = sorted(os.listdir(low_path))

    if len(high_images) != len(low_images):
        raise ValueError("The number of normal and low images does not match.")

    # Create a dictionary to map filenames for easier pairing by name
    high_image_dict = {os.path.splitext(img)[0]: img for img in high_images}
    low_image_dict = {os.path.splitext(img)[0]: img for img in low_images}

    # Ensure that both dictionaries have the same set of image names
    common_names = set(high_image_dict.keys()).intersection(low_image_dict.keys())

    if not common_names:
        raise ValueError("No matching image names found between high and low folders.")

    for name in common_names:
        high_img_name = high_image_dict[name]
        low_img_name = low_image_dict[name]

        high_img_path = os.path.join(high_path, high_img_name)
        low_img_path = os.path.join(low_path, low_img_name)

        high_img = cv2.imread(high_img_path)
        low_img = cv2.imread(low_img_path)

        if high_img is None or low_img is None:
            raise ValueError(f"Error loading images: {high_img_name}, {low_img_name}")

        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)

        paired_images.append(((low_img_name, high_img_name), low_img, high_img))

    return paired_images


def load_single_paired_image(eval_path, image_name):
    high_path = os.path.join(eval_path, "Normal")
    low_path = os.path.join(eval_path, "Low")

    paired_images = []

    if not os.path.exists(high_path) or not os.path.exists(low_path):
        raise FileNotFoundError("Normal or Low folder not found in the specified evaluation path.")

    high_img_path = os.path.join(high_path, f"{image_name}.png")
    low_img_path = os.path.join(low_path, f"{image_name}.png")

    if not os.path.isfile(high_img_path) or not os.path.isfile(low_img_path):
        raise FileNotFoundError(f"Image {image_name}.png not found in both Normal and Low folders.")

    high_img = cv2.imread(high_img_path)
    low_img = cv2.imread(low_img_path)

    if high_img is None or low_img is None:
        raise ValueError(f"Error loading images: {image_name}.png")

    high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)
    low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
    paired_images.append(((f"{image_name}.png", f"{image_name}.png"), low_img, high_img))
    return paired_images


def ensure_directory_exists(directory):
    """
    Ensure the specified directory exists. If not, create it.
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Error creating directory {directory}: {e}")


def save_image_with_annotation(image, image_name, annotation, save_folder):
    ensure_directory_exists(save_folder)

    base_name, ext = os.path.splitext(image_name)
    new_image_name = f"{base_name}_{annotation}{ext}"
    save_path = os.path.join(save_folder, new_image_name)
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(save_path, image_bgr)


def psnr(high_light_image, enhanced_image):
    psnr_metric = pyiqa.create_metric("psnr")

    # Normalize the RGB image to the range [0, 1]
    gt_img_normalized = high_light_image.astype(np.float32) / 255.0
    en_img_normalized = enhanced_image.astype(np.float32) / 255.0

    # Shape becomes (1, 3, H, W)
    gt_img_tensor = torch.tensor(gt_img_normalized).permute(2, 0, 1).unsqueeze(0)
    en_img_tensor = torch.tensor(en_img_normalized).permute(2, 0, 1).unsqueeze(0)

    # Calculate psnr score
    psnr_score = psnr_metric(en_img_tensor, gt_img_tensor)

    return psnr_score.item()


def ssim(high_light_image, enhanced_image):
    ssim_metric = pyiqa.create_metric("ssim")

    # Normalize the RGB image to the range [0, 1]
    gt_img_normalized = high_light_image.astype(np.float32) / 255.0
    en_img_normalized = enhanced_image.astype(np.float32) / 255.0

    # Shape becomes (1, 3, H, W)
    gt_img_tensor = torch.tensor(gt_img_normalized).permute(2, 0, 1).unsqueeze(0)
    en_img_tensor = torch.tensor(en_img_normalized).permute(2, 0, 1).unsqueeze(0)

    # Calculate ssim score
    ssim_score = ssim_metric(en_img_tensor, gt_img_tensor)

    return ssim_score.item()


lpips_metric = pyiqa.create_metric("lpips")


def lpips(high_light_image, enhanced_image):
    # Normalize the RGB image to the range [0, 1]
    gt_img_normalized = high_light_image.astype(np.float32) / 255.0
    en_img_normalized = enhanced_image.astype(np.float32) / 255.0

    # Shape becomes (1, 3, H, W)
    gt_img_tensor = torch.tensor(gt_img_normalized).permute(2, 0, 1).unsqueeze(0)
    en_img_tensor = torch.tensor(en_img_normalized).permute(2, 0, 1).unsqueeze(0)

    # Calculate lpips score
    lpips_score = lpips_metric(en_img_tensor, gt_img_tensor)

    return lpips_score.item()


def asinh_transform(x, a):
    x = x.astype(np.float32)
    return np.arcsinh(x * a) / (np.arcsinh(x + 1e-6))


def enhance_image(image, alpha):
    # Split channels
    R, G, B = cv2.split(image)

    # Apply asinh transformation per channel
    R_new = np.log1p(asinh_transform(R, alpha))
    G_new = np.log1p(asinh_transform(G, alpha))
    B_new = np.log1p(asinh_transform(B, alpha))

    # Merge channels back to RGB
    transformed = cv2.merge([R_new, G_new, B_new])
    # Normalize the RGB image
    transformed = (transformed / (np.max(transformed) + 1e-6)) * 255
    transformed = transformed.astype(np.uint8)

    return transformed


def clahe(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split the LAB image into channels
    l, a, b = cv2.split(lab_image)

    # Create a CLAHE object with parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply CLAHE to the L channel
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L channel back with the original A and B channels
    clahe_image = cv2.merge((cl, a, b))

    # Convert the image back to RGB
    rgb_clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_LAB2RGB)

    return rgb_clahe_image


def hist_eq(image):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split the LAB image into channels
    l, a, b = cv2.split(lab_image)

    # Apply histogram equalization to the L channel
    l_eq = cv2.equalizeHist(l)

    # Merge the equalized L channel back with the original A and B channels
    hist_eq_image = cv2.merge((l_eq, a, b))

    # Convert the image back to RGB
    rgb_hist_eq_image = cv2.cvtColor(hist_eq_image, cv2.COLOR_LAB2RGB)

    return rgb_hist_eq_image


def norm_log_transformation(image, c=1.0):
    # Normalize image to [0,1] range
    normalized = image / 255.0

    # Apply logarithmic transformation
    log_transformed = c * np.log1p(normalized)  # C . log1p(x) = 1 . log(1 + x)

    # Normalize to [0,1] and convert back to 0-255 uint8
    log_transformed = (log_transformed / np.max(log_transformed)) * 255
    log_transformed_image = log_transformed.astype(np.uint8)

    return log_transformed_image


def raw_log_transformation(image, c=1.0):
    image = np.float32(image)  # Convert to float to avoid overflow
    log_image = c * np.log1p(image)  # Apply log transformation
    log_image = (log_image / np.max(log_image)) * 255  # Normalize to 0-255
    return np.uint8(log_image)


def raw_asinh(x, a, Q):
    return np.arcsinh(a * Q * x) / Q


def lupton_asinh_method(image, a=8, Q=0.02):
    img_float = image.astype(np.float32) / 255.0
    r, g, b = img_float[..., 0], img_float[..., 1], img_float[..., 2]

    I = (r + g + b) / 3.0
    stretched_I = raw_asinh(I, a, Q)
    R = r * stretched_I / (I + 1e-8)
    G = g * stretched_I / (I + 1e-8)
    B = b * stretched_I / (I + 1e-8)

    RGB = np.stack([R, G, B], axis=-1)
    max_RGB = np.maximum(np.max(RGB, axis=-1, keepdims=True), 1.0)
    RGB = RGB / max_RGB
    return np.uint8(RGB * 255)


def pso_optimize_enhance_image(image, alpha):
    # Split channels
    R, G, B = cv2.split(image)

    # Apply asinh transformation per channel
    R_new = np.log1p(asinh_transform(R, alpha))
    G_new = np.log1p(asinh_transform(G, alpha))
    B_new = np.log1p(asinh_transform(B, alpha))

    transformed = cv2.merge([R_new, G_new, B_new])
    transformed = (transformed / (np.max(transformed) + 1e-6)) * 255
    transformed = transformed.astype(np.uint8)

    return objective_func_max_entropy(transformed)


def objective_func_max_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    entropy = shannon_entropy(gray)
    return entropy


def pso_optimize(low_light_image, num_particles=40, max_iters=20, w=0.9, c1=2, c2=2):
    # Initialize particle positions (alpha values)
    alpha_vals = torch.rand(num_particles)

    # Initialize velocities
    velocity_alpha = torch.zeros(num_particles)

    # Initialize personal best positions and scores
    personal_best_alpha = alpha_vals.clone()
    personal_best_scores = torch.full((num_particles,), float('-inf'))

    # Initialize global best
    global_best_score = float('-inf')
    global_best_alpha = alpha_vals[0]  # Initialize with first value

    for _ in range(max_iters):
        # Evaluate all particles
        scores = torch.tensor([
            pso_optimize_enhance_image(low_light_image, alpha.item()) for alpha in alpha_vals
        ], dtype=torch.float32)

        # Update personal bests
        better_mask = scores > personal_best_scores
        personal_best_alpha[better_mask] = alpha_vals[better_mask]
        personal_best_scores[better_mask] = scores[better_mask]

        # Update global best
        best_idx = torch.argmax(personal_best_scores)
        if personal_best_scores[best_idx] > global_best_score:
            global_best_score = personal_best_scores[best_idx]
            global_best_alpha = personal_best_alpha[best_idx]

        # Generate random factors
        r1 = torch.rand(num_particles)
        r2 = torch.rand(num_particles)

        # Update velocities
        velocity_alpha = (
                w * velocity_alpha +
                c1 * r1 * (personal_best_alpha - alpha_vals) +
                c2 * r2 * (global_best_alpha - alpha_vals)
        )

        # Update positions
        alpha_vals += velocity_alpha
        alpha_vals = torch.clamp(alpha_vals, 0.0, 1.0)  # Keep within (0,1)

    return global_best_alpha.item(), global_best_score.item()


def execute_pso(paired_images, num_particles=40, max_iters=20, w=0.9, c1=2, c2=2):
    images = []

    for (low_img_name, high_img_name), low_light_image, high_light_image in paired_images:
        alpha_opt, best_score = pso_optimize(low_light_image, num_particles, max_iters, w, c1, c2)
        images.append(((low_img_name, high_img_name), low_light_image, high_light_image, alpha_opt))

    return images


def calculate_metrics_for_enhanced_images(paired_images, save_output_folder):
    metrics = []

    for image_names, low_light, high_light, alpha in paired_images:
        image_folder = os.path.join(save_output_folder, os.path.splitext(image_names[0])[0])
        ensure_directory_exists(image_folder)

        proposed_image = enhance_image(low_light, alpha)
        proposed_psnr = psnr(high_light, proposed_image)
        proposed_ssim = ssim(high_light, proposed_image)
        proposed_lpips = lpips(high_light, proposed_image)

        save_image_with_annotation(proposed_image, image_names[0], "enhanced_image", image_folder)

        clahe_image = clahe(low_light)
        clahe_psnr = psnr(high_light, clahe_image)
        clahe_ssim = ssim(high_light, clahe_image)
        clahe_lpips = lpips(high_light, clahe_image)

        save_image_with_annotation(clahe_image, image_names[0], "clahe", image_folder)

        hist_eq_image = hist_eq(low_light)
        hist_eq_psnr = psnr(high_light, hist_eq_image)
        hist_eq_ssim = ssim(high_light, hist_eq_image)
        hist_eq_lpips = lpips(high_light, hist_eq_image)

        save_image_with_annotation(hist_eq_image, image_names[0], "he", image_folder)

        raw_log_image = raw_log_transformation(low_light)
        raw_log_psnr = psnr(high_light, raw_log_image)
        raw_log_ssim = ssim(high_light, raw_log_image)
        raw_log_lpips = lpips(high_light, raw_log_image)

        save_image_with_annotation(raw_log_image, image_names[0], "raw_log", image_folder)

        norm_log_image = norm_log_transformation(low_light)
        norm_log_image_psnr = psnr(high_light, norm_log_image)
        norm_log_image_ssim = ssim(high_light, norm_log_image)
        norm_log_image_lpips = lpips(high_light, norm_log_image)

        save_image_with_annotation(norm_log_image, image_names[0], "norm_log", image_folder)

        raw_asinh_image = lupton_asinh_method(low_light)
        raw_asinh_image_psnr = psnr(high_light, raw_asinh_image)
        raw_asinh_image_ssim = ssim(high_light, raw_asinh_image)
        raw_asinh_image_lpips = lpips(high_light, raw_asinh_image)

        save_image_with_annotation(raw_asinh_image, image_names[0], "raw_asinh", image_folder)

        save_image_with_annotation(low_light, image_names[0], "low", image_folder)
        save_image_with_annotation(high_light, image_names[0], "normal", image_folder)

        metrics.append({
            "Image name": image_names[0],
            "Alpha": alpha,

            "CLAHE PSNR": clahe_psnr,
            "HE PSNR": hist_eq_psnr,
            "Raw Asinh PSNR": raw_asinh_image_psnr,
            "Norm Log PSNR": norm_log_image_psnr,
            "Raw Log PSNR": raw_log_psnr,
            "Proposed PSNR": proposed_psnr,

            "CLAHE SSIM": clahe_ssim,
            "HE SSIM": hist_eq_ssim,
            "Raw Asinh SSIM": raw_asinh_image_ssim,
            "Norm Log SSIM": norm_log_image_ssim,
            "Raw Log SSIM": raw_log_ssim,
            "Proposed SSIM": proposed_ssim,

            "CLAHE LPIPS": clahe_lpips,
            "HE LPIPS": hist_eq_lpips,
            "Raw Asinh LPIPS": raw_asinh_image_lpips,
            "Norm Log LPIPS": norm_log_image_lpips,
            "Raw Log LPIPS": raw_log_lpips,
            "Proposed LPIPS": proposed_lpips,

        })
    return metrics


def read_and_calculate_mean_metrics(metrics):
    psnr_keys = [
        "CLAHE PSNR", "HE PSNR", "Raw Asinh PSNR",
        "Norm Log PSNR", "Raw Log PSNR", "Proposed PSNR"
    ]
    ssim_keys = [
        "CLAHE SSIM", "HE SSIM", "Raw Asinh SSIM",
        "Norm Log SSIM", "Raw Log SSIM", "Proposed SSIM"
    ]
    lpips_keys = [
        "CLAHE LPIPS", "HE LPIPS", "Raw Asinh LPIPS",
        "Norm Log LPIPS", "Raw Log LPIPS", "Proposed LPIPS"
    ]

    all_keys = psnr_keys + ssim_keys + lpips_keys
    all_metrics = {key: [] for key in all_keys}

    # Aggregate all values
    for metric in metrics:
        for key in all_keys:
            all_metrics[key].append(metric[key])

    # Calculate mean for each metric
    mean_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    # Print the results grouped by metric type
    print("\n=== Mean Metrics Across All Images ===\n")

    print("🔹 PSNR:")
    for key in psnr_keys:
        print(f"  {key:<20}: {mean_metrics[key]:.4f}")

    print("\n🔹 SSIM:")
    for key in ssim_keys:
        print(f"  {key:<20}: {mean_metrics[key]:.4f}")

    print("\n🔹 LPIPS:")
    for key in lpips_keys:
        print(f"  {key:<20}: {mean_metrics[key]:.4f}")


def enhance_images(paired_images, output_image_path):
    metric_result = calculate_metrics_for_enhanced_images(paired_images, output_image_path)
    read_and_calculate_mean_metrics(metric_result)

