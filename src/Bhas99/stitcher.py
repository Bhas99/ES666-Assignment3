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

    def cylindrical_projection(self, img, f):
        """Apply cylindrical projection to an image."""
        h, w = img.shape[:2]
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])  # Intrinsic camera matrix
        
        # Create the cylindrical map
        cyl = np.zeros_like(img)
        for y in range(h):
            for x in range(w):
                theta = (x - w / 2) / f
                h_ = (y - h / 2) / f
                
                X = np.array([np.sin(theta), h_, np.cos(theta)])
                X = np.dot(K, X)
                x_, y_ = int(X[0] / X[2]), int(X[1] / X[2])
                
                if 0 <= x_ < w and 0 <= y_ < h:
                    cyl[y, x] = img[y_, x_]
        
        return cyl

    def detect_and_extract_features(self, image_list):
        sift = cv2.SIFT_create(nfeatures=500)  # Limit to 500 keypoints per image
        keypoints = []
        descriptors = []
        for img in image_list:
            # Apply cylindrical projection to images
            projected_img = self.cylindrical_projection(img, f=500)
            kp, desc = sift.detectAndCompute(projected_img, None)
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
                matches.append(good_matches)
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
        stitched_image = images[0]
        
        height, width = images[0].shape[:2]
        result_canvas = np.zeros((height * 2, width * 3, 3), dtype=np.uint8)
        result_canvas[:height, :width] = stitched_image
        
        for i in range(1, len(images)):
            if i - 1 < len(homographies) and homographies[i - 1] is not None:
                H = homographies[i - 1]

                warped_image = cv2.warpPerspective(images[i], H, (result_canvas.shape[1], result_canvas.shape[0]))
                
                mask = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                
                result_canvas_bg = cv2.bitwise_and(result_canvas, result_canvas, mask=mask_inv)
                warped_fg = cv2.bitwise_and(warped_image, warped_image, mask=mask)
                
                result_canvas = cv2.add(result_canvas_bg, warped_fg)
            else:
                print(f"Skipping image {i} due to missing homography.")
        
        result = self.crop_black_edges(result_canvas)
        
        return result

    def crop_black_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
