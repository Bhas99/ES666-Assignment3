import cv2
import numpy as np
import glob
import os

class PanaromaStitcher():
    def __init__(self):
        pass

    def detect_and_match_features(self, img1, img2):
        # Feature detection
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

        # Feature matching
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        return keypoints1, keypoints2, matches

    def compute_homography(self, kp1, kp2, matches):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    def stitch_images(self, images):
        stitched_image = images[0]
        homography_matrices = []

        for i in range(1, len(images)):
            kp1, kp2, matches = self.detect_and_match_features(stitched_image, images[i])
            H = self.compute_homography(kp1, kp2, matches)
            homography_matrices.append(H)
            
            # Warp the next image onto the current panorama
            stitched_image = cv2.warpPerspective(stitched_image, H, (stitched_image.shape[1] + images[i].shape[1], stitched_image.shape[0]))
            stitched_image[0:images[i].shape[0], 0:images[i].shape[1]] = images[i]

        return stitched_image, homography_matrices

    def make_panaroma_for_images_in(self, path):
        # Load images
        image_files = sorted(glob.glob(os.path.join(path, '*')))
        images = [cv2.imread(img) for img in image_files]
        
        print(f'Found {len(images)} images for stitching')

        # Stitch images
        stitched_image, homography_matrix_list = self.stitch_images(images)

        # Save final stitched image
        result_path = './results/panorama.png'
        cv2.imwrite(result_path, stitched_image)
        print(f'Panorama saved ... @ {result_path}')
        
        return stitched_image, homography_matrix_list
