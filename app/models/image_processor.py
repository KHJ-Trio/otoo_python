import cv2
import numpy as np
import re
from models.ocr_processor import MyPaddleOCR

class ImageProcessor:
    def __init__(self):
        self.ocr = MyPaddleOCR()

    def adjust_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(9, 9))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def find_contours(self, image):
        kernel = np.ones((2, 2), np.uint8)
        morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
        dilated = cv2.dilate(morphed, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        edged = cv2.Canny(eroded, 30, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def get_bounding_boxes(self, contours, height, width, margin=135):
        contour_coords = []
        min_contour_width = 50
        min_contour_height = 30

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w >= min_contour_width and h >= min_contour_height:
                contour_coords.append((x, y, w, h))
        
        left_boxes = []
        right_boxes = []
        for x, y, w, h in contour_coords:
            contour_center_x = x + w // 2
            distance_to_left = contour_center_x
            distance_to_right = width - contour_center_x

            if distance_to_left < distance_to_right:
                new_w = min(w + margin, width - x)
                left_boxes.append((x, y, x + new_w, y + h))
            else:
                new_x = max(0, x - margin)
                new_w = min(w + margin, width - new_x)
                right_boxes.append((new_x, y, new_x + new_w, y + h))

        return left_boxes, right_boxes

    def draw_bounding_boxes(self, image, boxes, color):
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    def extract_text_from_boxes(self, boxes, ocr_results):
        text_list = []
        for box in boxes:
            result = []
            for points, text in ocr_results:
                x1, y1, x2, y2 = box
                all_inrange = all(x1 <= px <= x2 and y1 <= py <= y2 for px, py in points)
                if all_inrange:
                    result.append(text[0])
            text_list.append(" ".join(result))
        return text_list

    async def process_image(self, img_bytes):
        try:
            ocr_result = await self.ocr.run_ocr(img_bytes)
            if ocr_result is None:
                raise Exception("Failed to perform OCR")

            image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]

            adjusted_image = self.adjust_contrast(image)
            gray = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2GRAY)
            contours = self.find_contours(gray)

            left_boxes, right_boxes = self.get_bounding_boxes(contours, height, width)

            image_with_boxes = image.copy()
            self.draw_bounding_boxes(image_with_boxes, left_boxes, (255, 0, 0))
            self.draw_bounding_boxes(image_with_boxes, right_boxes, (0, 0, 255))

            left_text = self.extract_text_from_boxes(left_boxes, ocr_result)
            right_text = self.extract_text_from_boxes(right_boxes, ocr_result)

            return image_with_boxes, left_text, right_text
        except Exception as e:
            print(f"Error processing image: {e}")
            return None, [], []

    async def process_images(self, image_files):
        all_left_text = []
        all_right_text = []

        for image_file in image_files:
            img_bytes = await image_file.read()
            image_with_boxes, left_text, right_text = await self.process_image(img_bytes)

            if image_with_boxes is not None:
                all_left_text.extend(left_text)
                all_right_text.extend(right_text)

        all_left_text = [re.sub(r'크{2,}', 'ㅋㅋ', text) for text in all_left_text if text.strip()]
        all_right_text = [re.sub(r'크{2,}', 'ㅋㅋ', text) for text in all_right_text if text.strip()]

        return all_left_text, all_right_text