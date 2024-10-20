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
        sift = cv2.SIFT_create(nfeatures=800)  # Increase the number of keypoints
        keypoints = []
        descriptors = []
        for img in image_list:
            kp, desc = sift.detectAndCompute(img, None)
            keypoints.append(kp)
            descriptors.append(desc)
        return keypoints, descriptors

    def match_features(self, descriptors):
        # Use FLANN-based matcher for faster and more accurate matching
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        matches = []
        for i in range(len(descriptors) - 1):
            if descriptors[i] is not None and descriptors[i + 1] is not None:
                match = matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2)
                
                # Apply Lowe's ratio test to retain good matches
                good_matches = []
                for m, n in match:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                # Ensure enough good matches are found
                if len(good_matches) > 10:  # Minimum of 10 matches
                    matches.append(good_matches)
                else:
                    print(f"Not enough good matches between image {i} and {i + 1}. Skipping.")
        return matches

    def estimate_homographies(self, matches, keypoints):
        homographies = []
        for i, match_set in enumerate(matches):
            if len(match_set) < 4:
                print(f"Not enough matches to compute homography for image pair {i}. Skipping.")
                continue

            src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in match_set]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in match_set]).reshape(-1, 1, 2)
            
            # Compute homography using RANSAC for outlier rejection
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            homographies.append(H)
        return homographies

    def stitch_images(self, images, homographies):
        # Start with the first image as the base
        base_img = images[0]
        height, width = base_img.shape[:2]
        
        # Initialize canvas with a larger size
        result_canvas = np.zeros((height * 2, width * len(images), 3), dtype=np.uint8)
        result_canvas[:height, :width] = base_img
        
        current_transform = np.eye(3)  # Start with an identity matrix
        
        # Blend each image with the canvas
        for i in range(1, len(images)):
            if i - 1 < len(homographies) and homographies[i - 1] is not None:
                current_transform = current_transform @ homographies[i - 1]
                
                # Warp the next image using the cumulative transformation
                warped_image = cv2.warpPerspective(images[i], current_transform, 
                                                   (result_canvas.shape[1], result_canvas.shape[0]))
                
                # Create masks to blend the stitched image smoothly
                mask1 = cv2.cvtColor(result_canvas, cv2.COLOR_BGR2GRAY)
                _, mask1 = cv2.threshold(mask1, 1, 255, cv2.THRESH_BINARY)
                mask2 = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
                _, mask2 = cv2.threshold(mask2, 1, 255, cv2.THRESH_BINARY)
                
                # Create a combined mask to determine the overlap region
                overlap_mask = cv2.bitwise_and(mask1, mask2)
                
                # Blend using a weighted addition over the overlap area
                result_canvas = self.blend_images(result_canvas, warped_image, overlap_mask)
            else:
                print(f"Skipping image {i} due to missing homography.")
        
        result = self.crop_black_edges(result_canvas)
        return result

    def blend_images(self, img1, img2, overlap_mask):
        # Blending the overlapping areas
        mask1 = cv2.bitwise_not(overlap_mask)
        result1 = cv2.bitwise_and(img1, img1, mask=mask1)
        result2 = cv2.bitwise_and(img2, img2, mask=overlap_mask)
        
        # Weighted addition to smoothen the overlapping area
        alpha = 0.5
        blended = cv2.addWeighted(result1, alpha, result2, 1 - alpha, 0)
        
        # Combining the non-overlapping parts with the blended result
        result = cv2.add(result1, blended)
        return result

    def crop_black_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
