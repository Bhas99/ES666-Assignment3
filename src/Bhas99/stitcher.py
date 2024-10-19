import glob
import cv2
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        # Collect all images from the specified path
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} images for stitching')

        # Read the images
        images = [cv2.imread(im) for im in all_images]

        # Initialize the first image as the base
        stitched_image = images[0]

        # List to store homography matrices
        homography_matrix_list = []

        # Iterate over the consecutive images and stitch them together
        for i in range(1, len(images)):
            kp1, des1 = self.detect_features(stitched_image)
            kp2, des2 = self.detect_features(images[i])

            # Match features between the current and next image
            matches = self.match_features(des1, des2)

            # Extract points from the matched keypoints
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

            # Estimate homography matrix
            H = self.estimate_homography(src_pts, dst_pts)
            homography_matrix_list.append(H)

            # Warp the next image based on the homography matrix and stitch it
            stitched_image = self.stitch_images(stitched_image, images[i], H)

        # Return final stitched image and homography matrices
        return stitched_image, homography_matrix_list

    def detect_features(self, image):
        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use SIFT for feature detection
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, des1, des2):
        # Brute-force matcher with L2 norm for SIFT
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        # Sort matches by distance (best matches first)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def estimate_homography(self, src_pts, dst_pts):
        # Create the A matrix for Direct Linear Transformation (DLT)
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0], src_pts[i][1]
            xp, yp = dst_pts[i][0], dst_pts[i][1]
            A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
        A = np.array(A)

        # Perform SVD to solve the system of equations
        U, S, Vh = np.linalg.svd(A)
        H = Vh[-1].reshape(3, 3)
        return H / H[-1, -1]  # Normalize so that H[2,2] = 1

    def stitch_images(self, image1, image2, H):
        # Get size of the first image
        h1, w1 = image1.shape[:2]
        # Warp the second image using the homography matrix H
        warped_image = cv2.warpPerspective(image2, H, (w1 + image2.shape[1], h1))

        # Place the first image in the warped image space
        warped_image[0:h1, 0:w1] = image1
        return warped_image


    def match_features(self, desc1, desc2):
        # Create a Brute-Force matcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # Match descriptors
        matches = bf.match(desc1, desc2)
        # Sort matches by distance (best matches come first)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    def estimate_homography(self, src_pts, dst_pts):
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i][0], src_pts[i][1]
            xp, yp = dst_pts[i][0], dst_pts[i][1]
            A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])
        
        A = np.array(A)
        # SVD decomposition
        U, S, Vh = np.linalg.svd(A)
        # The homography matrix H is the last row of V (or Vh[-1])
        H = Vh[-1].reshape(3, 3)
        # Normalize so that H[2,2] == 1
        return H / H[-1, -1]

    def stitch_images(self, image1, image2, H):
        # Warp image2 to the perspective of image1
        result = cv2.warpPerspective(image2, H, (image1.shape[1] + image2.shape[1], image1.shape[0]))
        # Place image1 in the result (on the left)
        result[0:image1.shape[0], 0:image1.shape[1]] = image1
        return result
