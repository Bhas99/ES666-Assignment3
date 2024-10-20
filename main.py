from src.YourName.stitcher import PanaromaStitcher

def main():
    # Load your images
    images = [cv2.imread('Images/img1.jpg'), cv2.imread('Images/img2.jpg')]
    
    # Initialize the stitcher
    stitcher = PanaromaStitcher()
    
    # Create panorama
    panorama, homographies = stitcher.make_panaroma_for_images_in(images)
    
    # Save the results
    cv2.imwrite('./results/panorama.jpg', panorama)
    print("Panorama created and saved!")

if __name__ == "__main__":
    main()
