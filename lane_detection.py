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
        width = img.shape[1]
        # roi = np.array([[(50, height), (275, height), (150, 140)]])
        region_of_interest_vertices = [
            (0, height), (width / 2, height / 2), (width, height)
        ]
        triangle = np.array([region_of_interest_vertices], np.int32)
        mask_img = self.mask_img(triangle, img)
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
        lines = self.line_average(img, lines)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(lane_lines, (x1, y1), (x2, y2), (255, 0, 0), 5)

        lane = cv2.addWeighted(img, 0.8, lane_lines, 1, 1)
        return lane

    def make_coordinate(self, img, line_parameter):
        # print(line_parameter[0],line_parameter[1],line_parameter)
        if line_parameter is not None:
            slope = line_parameter[0]
            intercept = line_parameter[1]
            y1 = img.shape[0]
            y2 = int(y1 * (3 / 5))
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return np.array([x1, y1, x2, y2])

    def line_average(self, img, lines):
        left_fit = []
        right_fit = []
        # if lines is None:
        #   return None
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        if len(left_fit) and len(right_fit):
            left_fit_average = np.average(left_fit, axis=0)
            right_fit_average = np.average(right_fit, axis=0)

            left_line = self.make_coordinate(img, left_fit_average)
            right_line = self.make_coordinate(img, right_fit_average)
            return np.array([left_line, right_line])


def feed_from_video(lane_detection):
    cap = cv2.VideoCapture("road.mp4")

    frame_count = 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count < total:

        ret, lane_image = cap.read()
        lane_Detected_img = lane_detection.lane_detection(lane_image)
        cv2.imshow("Lane Detection", lane_Detected_img)
        frame_count += 1
        if (frame_count == total):
            frame_count=1
            cap.set(cv2.CAP_PROP_POS_FRAMES,1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(frame_count, total)
    # while cap.isOpened():
    # _, lane_image = cap.read()
    # lane_image = cv2.resize(lane_image, (320, 320))
    # lane_Detected_img = lane_detection.lane_detection(lane_image)
    # plt.imshow(lane, interpolation='bilinear')
    # plt.show()

    # cv2.imshow("Lane Detection", lane_Detected_img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    # break

    cap.release()
    #cv2.destroyAllWindows()


def feed_from_image(lane_detection):
    image = "D:/Github Repo/ML_Learning/LaneDetection/image/test_image.jpg"
    lane_image = cv2.imread(image)
    lane_image = cv2.resize(lane_image, (320, 320))
    lane_Detected_img = lane_detection.lane_detection(lane_image)
    cv2.imshow("Lane Detection", lane_Detected_img)
    cv2.waitKey(0)


if __name__ == "__main__":
    lane_detection = LaneDetection()
    # feed_from_image(lane_detection)
    feed_from_video(lane_detection)
# cv2.destroyAllWindows()
