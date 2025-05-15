import argparse
from image_enhancement import (
    load_single_paired_image,
    load_paired_images,
    execute_pso,
    enhance_images,
)


def execute_for_dataset(dataset_path, output_path):
    """
    Processes a dataset of image pairs from Normal and Low folders.
    """
    paired_images = load_paired_images(dataset_path)
    images = execute_pso(paired_images)
    enhance_images(images, output_path)


def execute_for_single_image_pair(image_pair_path, image_name, output_path):
    """
    Processes a single image pair given its name and folder.
    """
    paired_images = load_single_paired_image(image_pair_path, image_name)
    images = execute_pso(paired_images)
    enhance_images(images, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhance images.\n\n"
                    "Folder structure requirements:\n"
                    "- Dataset or image_pair_path must contain two folders:\n"
                    "  - Normal/: contains normal-light images\n"
                    "  - Low/: contains low-light images\n"
                    "- In dataset mode, both folders must contain matched image filenames.\n"
                    "- In single image mode, provide the image name (without extension).\n",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-dataset_path",
        type=str,
        help="Path to dataset folder. Must contain 'Normal/' and 'Low/' subfolders with matching images."
    )
    parser.add_argument(
        "-image_pair_path",
        type=str,
        help="Path to folder with a single image pair. Must contain 'Normal/' and 'Low/' subfolders."
    )
    parser.add_argument(
        "image_name",
        nargs="?",
        type=str,
        help="Name of the image pair (without extension) to process when using -image_pair_path."
    )
    parser.add_argument(
        "-output_path",
        required=True,
        type=str,
        help="Path to save enhanced image(s)."
    )

    args = parser.parse_args()

    if args.dataset_path:
        execute_for_dataset(args.dataset_path, args.output_path)
    elif args.image_pair_path and args.image_name:
        execute_for_single_image_pair(args.image_pair_path, args.image_name, args.output_path)
    else:
        print("Invalid arguments.\n"
              "Use either:\n"
              "  python main.py -dataset_path <path> -output_path <path>\n"
              "OR\n"
              "  python main.py -image_pair_path <path> <image_name> -output_path <path>")