import cv2
import numpy as np
import matplotlib.pyplot as plt

class LaneDetection():
    """docstring for LaneDetection"""

    def __init__(self):
        pass

    """docstring for image bgr to gray scale"""
    def channel_conversion(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray_img
    """docstring for image noise reduction using gaussian blur"""

    def feature_extraction(self, img):
        # step 1, parameter need to tune
        blur_img = cv2.GaussianBlur(img, (5, 5), 0)

        # Step 2, canny edge detection
        canny_img = cv2.Canny(blur_img, 50, 150)
        return canny_img

    """docstring for finding roi"""

    def roi(self, img):
        height = img.shape[0]
        roi = np.array([[(50, height), (275, height), (150, 140)]])
        mask_img = self.mask_img(roi, img)
        roi = cv2.bitwise_and(img, mask_img)
        return roi

    """docstring for image masking with roi"""

    def mask_img(self, roi, img):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, roi, 255)
        return mask

    """docstring perform detection based on HoughTransformP"""

    def hough_p_transform(self, img):

        lines = cv2.HoughLinesP(img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        return lines

    """docstring for lane detection"""

    def lane_detection(self, img):
        test_image = np.copy(img)
        # step 1
        gray_img = self.channel_conversion(test_image)
        # step 2
        canny_img = self.feature_extraction(gray_img)

        # step 3
        crop_img = self.roi(canny_img)
        # Step 4
        lines = lane_detection.hough_p_transform(crop_img)
        # step 5
        lane_lines = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(lane_lines, (x1, y1), (x2, y2), (255, 0, 0), 10)
        combo_img = cv2.addWeighted(img, 0.8, lane_lines, 1, 1)
        return combo_img


if __name__ == "__main__":
    lane_detection = LaneDetection()

    cap = cv2.VideoCapture("road.mp4")

    while cap.isOpened():
        #image = "D:/Github Repo/ML_Learning/LaneDetection/image/test_image.jpg"
        #lane_image = cv2.imread(image)
        _, lane_image = cap.read()
        lane_image = cv2.resize(lane_image, (320, 320))
        lane_Detected_img = lane_detection.lane_detection(lane_image)
        # plt.imshow(lane, interpolation='bilinear')
        # plt.show()

        cv2.imshow("Lane Detection", lane_Detected_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
