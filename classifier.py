import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

class GestureDetection:
    
    def __init__(self) -> None:
        base_options = python.BaseOptions(model_asset_path='./gesture_recognizer_2.task')
        options = vision.GestureRecognizerOptions(base_options=base_options)
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        # print(self.recognizer)

    
    def createImage(self, image):
        # print(image.shape)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data= image
        )
        
        return image
    
    def recognize_gesture(self, image):
        return self.recognizer.recognize(image)
    
    def get_landmarks(self, recognition_result):
        top_gesture = recognition_result.gestures[0][0]
        hand_landmarks = recognition_result.hand_landmarks
        
        return (top_gesture, hand_landmarks)

    def annotate(self, image, results):
        mp_drawing = mp.solutions.drawing_utils
        multi_hand_landmarks_list = results[1]
        annotated_image = image.copy()
        
        for hand_landmarks in multi_hand_landmarks_list:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            pass        
        
        
        return annotated_image