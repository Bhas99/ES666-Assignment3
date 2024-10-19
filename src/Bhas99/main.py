
import cv2
from Bhas99.stitcher import PanaromaStitcher  # Import your class
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

if __name__ == "__main__":
    # Load images from the Images folder
    images = load_images_from_folder('./Users/bhaskarhazarika/Desktop/ES666-Assignment3/Images')

    # Create an instance of PanaromaStitcher
    stitcher = PanaromaStitcher()
    
    # Call the make_panaroma_for_images_in method with the loaded images
    stitched_image, homographies = stitcher.make_panaroma_for_images_in(images)

    # Save the final stitched image
    cv2.imwrite('./results/stitched_image.jpg', stitched_image)

    # Print out the homography matrices for verification
    for i, H in enumerate(homographies):
        print(f"Homography Matrix {i+1}:\n", H)
