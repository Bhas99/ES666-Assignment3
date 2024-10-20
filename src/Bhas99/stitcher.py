import cv2
import numpy as np

class PanaromaStitcher:
    def make_panaroma_for_images_in(self, image_list):
        # Convert images to the correct format (OpenCV expects BGR, not RGB)
        image_list_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in image_list]

        # Create the OpenCV Stitcher instance
        stitcher = cv2.Stitcher_create()
        status, stitched_image = stitcher.stitch(image_list_bgr)
        
        if status == cv2.Stitcher_OK:
            # Apply Gaussian blur to smooth out the panorama
            blurred_image = self.apply_gaussian_blur(stitched_image)

            # Convert back to RGB for consistent display
            stitched_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
            return stitched_image_rgb, None  # No manual homography matrices needed
        else:
            print("Error: Unable to stitch images.")
            return None, None

    def apply_gaussian_blur(self, image):
        """Apply Gaussian blur to smooth out edges and transitions."""
        # Apply Gaussian blur with a kernel size (you can adjust the size as needed)
        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        return blurred
