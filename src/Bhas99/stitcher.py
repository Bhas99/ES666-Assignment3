import cv2
import numpy as np
import os

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

        # Save the stitched image
        if stitched_image is not None:
            if not os.path.exists('./results'):
                os.makedirs('./results')
            cv2.imwrite('./results/panorama_result.jpg', cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR))

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
                good_matches = []
                for m, n in match:
                    if m.distance < 0.75 * n.distance:
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

            src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in match_set])
            dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in match_set])

            # Manually compute homography using normalized DLT
            H = self.compute_homography(src_pts, dst_pts)
            homographies.append(H)
        return homographies

    def compute_homography(self, src_pts, dst_pts):
        """ Manually compute the homography matrix using Direct Linear Transformation (DLT) with normalization. """
        src_pts, T_src = self.normalize_points(src_pts)
        dst_pts, T_dst = self.normalize_points(dst_pts)

        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0], src_pts[i][1]
            xp, yp = dst_pts[i][0], dst_pts[i][1]
            A.append([-x, -y, -1, 0, 0, 0, xp*x, xp*y, xp])
            A.append([0, 0, 0, -x, -y, -1, yp*x, yp*y, yp])
        
        A = np.array(A)
        U, S, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)

        # Denormalize the homography matrix
        H = np.linalg.inv(T_dst) @ H @ T_src
        H = H / H[2, 2]  # Normalize so that H[2,2] = 1
        return H

    def normalize_points(self, pts):
        """ Normalize points to improve homography computation. """
        mean = np.mean(pts, axis=0)
        std = np.std(pts)

        T = np.array([
            [1/std, 0, -mean[0]/std],
            [0, 1/std, -mean[1]/std],
            [0, 0, 1]
        ])
        normalized_pts = np.dot(T, np.concatenate((pts.T, np.ones((1, pts.shape[0]))), axis=0))
        return normalized_pts[:2].T, T

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
                result_canvas = self.blend_images(result_canvas, warped_image)
            else:
                print(f"Skipping image {i} due to missing homography.")
        
        return self.crop_black_edges(result_canvas)

    def blend_images(self, img1, img2):
        """Basic blending for overlapping regions."""
        mask = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        img1_bg = cv2.bitwise_and(img1, img1, mask=mask_inv)
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
        
        blended = cv2.add(img1_bg, img2_fg)
        return blended

    def crop_black_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
