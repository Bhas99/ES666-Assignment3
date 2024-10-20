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
            matches.append(matcher.match(descriptors[i], descriptors[i+1]))
        return matches

    def estimate_homographies(self, matches, keypoints):
        homographies = []
        for match_set in matches:
            src_pts = np.float32([keypoints[0][m.queryIdx].pt for m in match_set]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[1][m.trainIdx].pt for m in match_set]).reshape(-1, 1, 2)
            
            # Compute homography using RANSAC
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            homographies.append(H)
        return homographies

    def stitch_images(self, images, homographies):
        # Assume the first image as the base
        stitched_image = images[0]
        for i in range(1, len(images)):
            # Warp image using the homography matrix
            h, w, _ = images[i].shape
            warped_image = cv2.warpPerspective(images[i], homographies[i-1], (w * 2, h * 2))
            # Combine with the stitched image
            stitched_image = cv2.addWeighted(stitched_image, 0.5, warped_image, 0.5, 0)
        return stitched_image



