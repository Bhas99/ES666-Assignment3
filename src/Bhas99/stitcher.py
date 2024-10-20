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
        for i in range(1, len(images)):
            # Get the homography matrix for the current pair
            if i - 1 < len(homographies) and homographies[i - 1] is not None:
                H = homographies[i - 1]

                # Determine the size of the stitched image canvas
                height, width = images[i].shape[:2]
                stitched_image = cv2.warpPerspective(stitched_image, H, (width * 2, height))

                # Add the next image on top
                stitched_image[0:height, 0:width] = images[i]
            else:
                print(f"Skipping image {i} due to missing homography.")

        return stitched_image
