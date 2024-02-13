import cv2
import dlib
import numpy as np

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

# Function to add sunglasses
def add_sunglasses_to_face(image_path, sunglasses_path):
    # Load the images
    image = cv2.imread(image_path)
    sunglasses_img = cv2.imread(sunglasses_path, cv2.IMREAD_UNCHANGED)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Get the position of the left and right eye
        left_eye_x = landmarks.part(36).x
        left_eye_y = landmarks.part(36).y
        right_eye_x = landmarks.part(45).x
        right_eye_y = landmarks.part(45).y

        # Calculate angle between eyes
        delta_x = right_eye_x - left_eye_x
        delta_y = right_eye_y - left_eye_y
        angle = np.arctan(delta_y / delta_x)
        angle = (angle * 180) / np.pi

        # Scale and rotate the sunglasses
        eye_width = np.sqrt((delta_x ** 2) + (delta_y ** 2))
        sunglass_width = eye_width * 2
        scale_factor = sunglass_width / sunglasses_img.shape[1]
        sunglass_resized = cv2.resize(sunglasses_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        sunglass_rotated = cv2.getRotationMatrix2D((sunglass_resized.shape[1] / 2, sunglass_resized.shape[0] / 2), angle, 1)
        sunglass_rotated = cv2.warpAffine(sunglass_resized, sunglass_rotated, (sunglass_resized.shape[1], sunglass_resized.shape[0]))

        # Calculate position and overlay sunglasses
        x = left_eye_x - int(0.25 * sunglass_width)
        y = left_eye_y - int(0.5 * sunglass_resized.shape[0])
        image = overlay_transparent(image, sunglass_rotated, x, y)

    return image

# Add sunglasses to the faces in the image
result_image = add_sunglasses_to_face('3.jpg', 'glasses.png')

# Save or display the result
cv2.imwrite('output_image.jpg', result_image)
# cv2.imshow('Output', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
