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
        sift = cv2.SIFT_create(nfeatures=800)
        keypoints = []
        descriptors = []
        for img in image_list:
            kp, desc = sift.detectAndCompute(img, None)
            keypoints.append(kp)
            descriptors.append(desc)
        return keypoints, descriptors

    def match_features(self, descriptors):
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
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
                
                if len(good_matches) > 10:
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
            
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            homographies.append(H)
        return homographies

    def stitch_images(self, images, homographies):
        result = images[0]
        height, width = images[0].shape[:2]
        result_canvas = np.zeros((height * 2, width * len(images), 3), dtype=np.uint8)
        result_canvas[:height, :width] = result
        
        current_transform = np.eye(3)
        
        for i in range(1, len(images)):
            if i - 1 < len(homographies) and homographies[i - 1] is not None:
                current_transform = current_transform @ homographies[i - 1]
                
                warped_image = cv2.warpPerspective(images[i], current_transform, 
                                                   (result_canvas.shape[1], result_canvas.shape[0]))
                
                result_canvas = self.alpha_blend(result_canvas, warped_image)
            else:
                print(f"Skipping image {i} due to missing homography.")
        
        return self.crop_black_edges(result_canvas)

    def alpha_blend(self, img1, img2):
        """Alpha blending for smoother transitions."""
        alpha = 0.5
        blend = np.where(img2 == 0, img1, cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0))
        return blend

    def crop_black_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
