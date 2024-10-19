import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        # Get all images from the directory
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(all_images)))

        if len(all_images) < 2:
            raise ValueError("Need at least two images to stitch!")

        # Read the images
        images = [cv2.imread(img) for img in all_images]

        # Initialize the first image
        stitched_image = images[0]
        homography_matrix_list = []

        # Loop through the images and stitch them together
        for i in range(1, len(images)):
            # Detect and match features between current stitched image and next image
            kp1, kp2, matches = self.detect_and_match_features(stitched_image, images[i])

            # Compute homography matrix between the two images
            H = self.compute_homography(kp1, kp2, matches)
            homography_matrix_list.append(H)

            # Warp the next image using the computed homography
            stitched_image = self.warp_and_stitch(stitched_image, images[i], H)

        # Save the final stitched image
        result_path = './results/panorama.png'
        cv2.imwrite(result_path, stitched_image)
        print(f'Panorama saved ... @ {result_path}')

        return stitched_image, homography_matrix_list

    def detect_and_match_features(self, img1, img2):
        # Convert to grayscale for feature detection
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Initialize SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and descriptors
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        # Match features using BFMatcher (Brute Force)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)

        # Sort matches by distance (lower distance = better match)
        matches = sorted(matches, key=lambda x: x.distance)

        return kp1, kp2, matches

    def compute_homography(self, kp1, kp2, matches):
        # Extract the matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography matrix using RANSAC to find the best fit
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return H

    def warp_and_stitch(self, img1, img2, H):
        # Get dimensions of the first image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Get the perspective of the second image (img2) based on homography
        img1_dims = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        img2_dims = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

        # Warp the second image to fit into the panorama
        img2_transformed_dims = cv2.perspectiveTransform(img2_dims, H)

        # Get the combined dimensions of the resulting stitched image
        combined_dims = np.vstack((img1_dims, img2_transformed_dims))

        [xmin, ymin] = np.int32(combined_dims.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(combined_dims.max(axis=0).ravel() + 0.5)

        # Adjust the translation homography to fit both images
        translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

        # Warp the first image and blend it with the warped second image
        result_img = cv2.warpPerspective(img1, translation @ H, (xmax - xmin, ymax - ymin))

        # Create a mask to avoid overwriting the first image
        mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.uint8)
        mask[-ymin:h2 - ymin, -xmin:w2 - xmin] = 255

        # Use bitwise OR to combine the images based on non-zero mask region
        result_img[-ymin:h2 - ymin, -xmin:w2 - xmin] = cv2.bitwise_or(
            result_img[-ymin:h2 - ymin, -xmin:w2 - xmin], img2)

        return result_img


    def do_something_more(self):
        return None

