import cv2
import numpy as np

class PanaromaStitcher:
    def make_panaroma_for_images_in(self, image_list):
        image_list_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in image_list]

        keypoints, descriptors = self.detect_and_extract_features(image_list_bgr)
        
        matches = self.match_features(descriptors)

        homographies = self.estimate_homographies(matches, keypoints)

        # Stitching images using OpenCV Stitcher for comparison
        stitcher = cv2.Stitcher_create()
        status, stitched_image = stitcher.stitch(image_list_bgr)

        if status == cv2.Stitcher_OK:
            # Converting back to RGB for consistent display
            stitched_image_rgb = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
            return stitched_image_rgb, homographies  
        else:
            print("Error: Unable to stitch images.")
            return None, homographies

    def detect_and_extract_features(self, image_list):
        # Using SIFT for feature detection and extraction
        sift = cv2.SIFT_create()
        keypoints = []
        descriptors = []
        for img in image_list:
            kp, desc = sift.detectAndCompute(img, None)
            keypoints.append(kp)
            descriptors.append(desc)
        return keypoints, descriptors

    def match_features(self, descriptors):
        # Using FLANN-based matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

        matches = []
        for i in range(len(descriptors) - 1):
            if descriptors[i] is not None and descriptors[i + 1] is not None:
                match = matcher.knnMatch(descriptors[i], descriptors[i + 1], k=2)

                # Applying Lowe's ratio test to keep good matches
                good_matches = [m for m, n in match if m.distance < 0.75 * n.distance]

                if len(good_matches) > 10:
                    matches.append(good_matches)
                else:
                    print(f"Not enough matches between images {i} and {i + 1}. Skipping.")
        return matches

    def estimate_homographies(self, matches, keypoints):
        homographies = []
        for i, match_set in enumerate(matches):
            if len(match_set) < 4:
                print(f"Not enough matches to compute homography for image pair {i}. Skipping.")
                continue

            src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in match_set]).reshape(-1, 2)
            dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in match_set]).reshape(-1, 2)

            # Manually computing the homography using Direct Linear Transform 
            H = self.compute_homography(src_pts, dst_pts)
            if H is not None:
                homographies.append(H)

        return homographies

    def compute_homography(self, src_pts, dst_pts):
        # Implementing Direct Linear Transform for homography estimation
        A = []
        for i in range(len(src_pts)):
            x, y = src_pts[i]
            xp, yp = dst_pts[i]
            A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x*yp, y*yp, yp])

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2] if H[2, 2] != 0 else None
