import cv2
import numpy as np

class PanaromaStitcher:
    def make_panaroma_for_images_in(self, image_list):
        # Step 1: Detect and extract features
        keypoints, descriptors = self.detect_and_extract_features(image_list)

        # Step 2: Match the features between the images
        matches = self.match_features(descriptors)

        # Step 3: Estimate homography matrices using the matches
        homographies = self.estimate_homographies(matches, keypoints)

        # Step 4: Stitch the images using homography matrices
        stitched_image = self.stitch_images(image_list, homographies)

        # Return the stitched panorama and the homography matrices
        return stitched_image, homographies

    def detect_and_extract_features(self, image_list):
        orb = cv2.ORB_create()
        keypoints = []
        descriptors = []
        for img in image_list:
            kp, desc = orb.detectAndCompute(img, None)
            keypoints.append(kp)
            descriptors.append(desc)
        return keypoints, descriptors

    def match_features(self, descriptors):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = []
        for i in range(len(descriptors) - 1):
            match = matcher.match(descriptors[i], descriptors[i + 1])
            matches.append(match)
        return matches

    def estimate_homographies(self, matches, keypoints):
        homographies = []
        for i, match_set in enumerate(matches):
            if len(match_set) < 4:
                print(f"Not enough matches to compute homography for image pair {i}. Skipping.")
                continue

            src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in match_set]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in match_set]).reshape(-1, 1, 2)
            
            # Compute homography using RANSAC
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            homographies.append(H)
        return homographies

    def stitch_images(self, images, homographies):
    # Start with the first image as the base
    stitched_image = images[0]
    
    # Create a larger canvas to fit the stitched result
    height, width = images[0].shape[:2]
    result_canvas = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)
    result_canvas[:height, :width] = stitched_image
    
    # Process each image with its homography matrix
    for i in range(1, len(images)):
        if i - 1 < len(homographies) and homographies[i - 1] is not None:
            H = homographies[i - 1]

            # Warp the next image onto the canvas
            warped_image = cv2.warpPerspective(images[i], H, (result_canvas.shape[1], result_canvas.shape[0]))
            
            # Use a mask to blend the stitched image smoothly
            mask = (warped_image > 0).astype(np.uint8)
            result_canvas = cv2.add(result_canvas, warped_image, mask=mask)
        else:
            print(f"Skipping image {i} due to missing homography.")
    
    # Crop out the black areas
    result = self.crop_black_edges(result_canvas)
    
    return result

def crop_black_edges(self, image):
    # Convert to grayscale and find all non-black pixels
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find the bounding box of all non-black areas
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop to the bounding box
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image
