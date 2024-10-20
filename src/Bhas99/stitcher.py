import cv2
import os
from src.Bhas99.stitcher import PanaromaStitcher

def load_images_from_folder(folder_path):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images

def main():
    # List of folder names to process
    folders = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6']
    
    # Initialize the stitcher
    stitcher = PanaromaStitcher()
    
    # Loop through each folder to create panoramas
    for folder in folders:
        folder_path = f'Images/{folder}'
        
        # Load images from the current folder
        images = load_images_from_folder(folder_path)
        if len(images) < 2:
            print(f"Not enough images in {folder} to create a panorama. Skipping.")
            continue
        
        # Create panorama
        try:
            panorama, homographies = stitcher.make_panaroma_for_images_in(images)
            
            # Save the results
            output_path = f'./results/{folder}_panorama.jpg'
            if not os.path.exists('results'):
                os.makedirs('results')
            cv2.imwrite(output_path, panorama)
            print(f"Panorama created and saved for {folder}!")
        except Exception as e:
            print(f"Failed to create panorama for {folder}. Error: {e}")

if __name__ == "__main__":
    main()




