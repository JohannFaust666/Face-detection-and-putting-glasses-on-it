import cv2
import dlib
import numpy as np
import argparse

# Initialize the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to overlay an image with transparency
def overlay_transparent(background, overlay, x, y):
    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay_image

    return background

# Fuction that blurs face and overlay glasses if needed by given arguments
def process_image(image_path, add_sunglasses, blur_face, sunglasses_path='glasses.png'):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        landmarks = predictor(gray, face)

        if blur_face:
            face_mask = np.zeros_like(gray)
            points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in range(0, 68)])
            convexhull = cv2.convexHull(points)
            cv2.fillConvexPoly(face_mask, convexhull, 255)
            inverted_face_mask = cv2.bitwise_not(face_mask)
            face_image = cv2.bitwise_and(image, image, mask=face_mask)
            blurred_face = cv2.GaussianBlur(face_image, (99, 99), 30)
            image = cv2.bitwise_and(blurred_face, blurred_face, mask=face_mask) + cv2.bitwise_and(image, image, mask=inverted_face_mask)

        if add_sunglasses:
            sunglasses_img = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)
            left_eye_x = landmarks.part(36).x
            left_eye_y = landmarks.part(36).y
            right_eye_x = landmarks.part(45).x
            right_eye_y = landmarks.part(45).y
            delta_x = right_eye_x - left_eye_x
            delta_y = right_eye_y - left_eye_y
            angle = np.arctan(delta_y / delta_x)
            angle = (angle * 180) / np.pi
            eye_width = np.sqrt((delta_x ** 2) + (delta_y ** 2))
            sunglass_width = eye_width * 2
            scale_factor = sunglass_width / sunglasses_img.shape[1]
            sunglass_resized = cv2.resize(sunglasses_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
            sunglass_rotated = cv2.getRotationMatrix2D((sunglass_resized.shape[1] / 2, sunglass_resized.shape[0] / 2), angle, 1)
            sunglass_rotated = cv2.warpAffine(sunglass_resized, sunglass_rotated, (sunglass_resized.shape[1], sunglass_resized.shape[0]))
            x = left_eye_x - int(0.25 * sunglass_width)
            y = left_eye_y - int(0.5 * sunglass_resized.shape[0])
            image = overlay_transparent(image, sunglass_rotated, x, y)

    return image



if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Process an image.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--sunglasses', action='store_true', help='Overlay sunglasses on the face')
    parser.add_argument('--blur', action='store_true', help='Blur the face')

    args = parser.parse_args()

    result_image = process_image(args.image_path, args.sunglasses, args.blur)

    # You can change name of the output img if needed
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, result_image)
    print(f"Processed image saved as {output_path}")
