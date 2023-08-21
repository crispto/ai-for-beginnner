import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector

options = vision.ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='../models/efficientdet.tflite'),
    max_results=5, score_threshold=0.5)


# @markdown We implemented some functions to visualize the object detection results. <br/> Run the following cell to activate the functions.

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def cv2_imshow(image):
    cv2.namedWindow('MediaPipe', cv2.WINDOW_GUI_NORMAL)
    cv2.imshow('MediaPipe', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize(
    image,
    detection_result
) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


def main():
    with ObjectDetector.create_from_options(options) as detector:
        image = mp.Image.create_from_file('./cup-test.jpg')
        np_image = image.numpy_view()
        new_shape = np.array(np_image.shape[:2])//2
        print(new_shape)

        smaller = cv2.resize(np_image, (np_image.shape[1]//2, np_image.shape[0]//2))
        print(smaller.shape)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=smaller)
        detection_result = detector.detect(image)
        image_copy = np.copy(image.numpy_view())
        annotated_image = visualize(image_copy, detection_result)
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        cv2_imshow(rgb_annotated_image)

if __name__ == '__main__':
    main()
