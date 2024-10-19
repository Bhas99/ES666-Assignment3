import os
from src.stitcher import PanaromaStitcher

def main():
    # Parent directory containing the subdirectories of images (e.g., I2, I6, I4)
    parent_image_directory = './Images'

    # Get all subdirectories in the parent image directory
    subdirs = [d for d in os.listdir(parent_image_directory) if os.path.isdir(os.path.join(parent_image_directory, d))]

    # Initialize the panorama stitcher
    stitcher = PanaromaStitcher()

    # Process each subdirectory
    for subdir in subdirs:
        image_directory = os.path.join(parent_image_directory, subdir)
        print(f"\n************ Processing images in {image_directory} ************")

        try:
            # Perform panorama stitching for the images in this subdirectory
            stitched_image, homography_matrices = stitcher.make_panaroma_for_images_in(image_directory)

            # Save the result for this set of images
            result_path = f'./results/panorama_{subdir}.png'
            cv2.imwrite(result_path, stitched_image)

            print(f"Panorama saved at {result_path}")

            # Optionally, print homography matrices for each stitching step
            for i, H in enumerate(homography_matrices):
                print(f"Homography matrix {i+1} for {subdir}:\n{H}")

        except Exception as e:
            print(f"Error processing {image_directory}: {e}")

if __name__ == "__main__":
    main()


