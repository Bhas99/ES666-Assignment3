import cv2
import numpy as np
import os

class PanaromaStitcher:
    def make_panaroma_for_images_in(self, image_list):
        # Apply Gaussian blur to each image to smoothen and reduce noise
        image_list = [cv2.GaussianBlur(img, (5, 5), 0) for img in image_list]
        
        # Detect and extract features
        keypoints, descriptors = self.detect_and_extract_features(image_list)
        
        # Match features across images
        matches = self.match_features(descriptors)
        
        # Estimate homographies
        homographies = self.estimate_homographies(matches, keypoints)
        
        # Stitch images using the estimated homographies
        stitched_image = self.stitch_images(image_list, homographies)
        
        # Save the stitched image if it exists
        if stitched_image is not None:
            if not os.path.exists('./results'):
                os.makedirs('./results')
            cv2.imwrite('./results/panorama_result.jpg', cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))
        
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
                
                # Apply ratio test
                good_matches = [m for m, n in match if m.distance < 0.75 * n.distance]
                
                if len(good_matches) > 10:
                    matches.append(good_matches)
                else:
                    print(f"Not enough matches between image {i} and {i + 1}. Skipping.")
        return matches

    def estimate_homographies(self, matches, keypoints):
        homographies = []
        for i, match_set in enumerate(matches):
            if len(match_set) < 4:
                print(f"Not enough matches to compute homography for image pair {i}. Skipping.")
                continue

            src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in match_set]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in match_set]).reshape(-1, 1, 2)

            # Use RANSAC to compute homography
            H = self.direct_linear_transform(src_pts, dst_pts)
            if H is not None:
                homographies.append(H)
        return homographies

    def direct_linear_transform(self, src_pts, dst_pts):
        # Assuming src_pts and dst_pts are Nx1x2
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0]
            xp, yp = dst_pts[i][0]
            A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
        
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        return H / H[2, 2] if H[2, 2] != 0 else None

    def stitch_images(self, images, homographies):
        result = images[0]
        height, width = images[0].shape[:2]
        canvas = np.zeros((height * 2, width * len(images), 3), dtype=np.uint8)
        canvas[:height, :width] = result
        
        transform = np.eye(3)
        
        for i in range(1, len(images)):
            if i - 1 < len(homographies) and homographies[i - 1] is not None:
                transform = transform @ homographies[i - 1]
                warped = cv2.warpPerspective(images[i], transform, (canvas.shape[1], canvas.shape[0]))
                canvas = self.blend_images(canvas, warped)
            else:
                print(f"Skipping image {i} due to missing homography.")
        
        return self.crop_black_edges(canvas)

    def blend_images(self, img1, img2):
        alpha = 0.5
        mask = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        img1_bg = cv2.bitwise_and(img1, img1, mask=mask_inv)
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
        
        return cv2.addWeighted(img1_bg, alpha, img2_fg, 1 - alpha, 0)

    def crop_black_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]
