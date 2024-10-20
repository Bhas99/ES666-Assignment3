import cv2

class PanaromaStitcher:
    def make_panaroma_for_images_in(self, image_list):
        # Convert images to the correct format (OpenCV expects BGR, not RGB)
        image_list_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in image_list]

        # Create the OpenCV Stitcher instance
        stitcher = cv2.Stitcher_create()
        status, stitched_image = stitcher.stitch(image_list_bgr)
        
        if status == cv2.Stitcher_OK:
            # Convert back to RGB for consistent display
            stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
            return stitched_image_rgb, None  # No manual homography matrices needed
        else:
            print("Error: Unable to stitch images.")
            return None, None
