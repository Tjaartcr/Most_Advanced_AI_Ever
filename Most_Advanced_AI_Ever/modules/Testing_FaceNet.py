'''

### Explanation

- **Step 1**: We load the Dlib face detection and Face Recognition models.
- **Step 2**: We read the image from which we want to detect faces.
- **Step 3**: We detect all faces in the image using `detector`.
- **Step 4**: For each detected face, we extract its encoding using `face_encodings`.
- **Step 5**: We use the `compare_faces` function to match the current face encoding with all known encodings. The closest match is used to determine the recognized name.
- Finally, we display the results by matching each face encoding with the known encodings.

This example assumes that you have a CSV file `path_to_known_faces.csv` with columns `name` and `image`, where the `image` column contains the path to the image for each known face. 

'''



import cv2
import dlib
from face_recognition import face_encodings, load_image_file, compare_faces

# Load the Dlib Face Detection Model
detector = dlib.get_frontal_face_detector()

# Load the Face Recognition Model
face_recognizer = face_recognition.face_recognition_model("dlib_face_recognition_resnet50_2017_10_17.svm")

# Read the Image
image = load_image_file("path_to_your_image.jpg")

# Detect Faces
face_locations = detector(image)

# Extract Face Encodings
face_encodings_list = []
for location in face_locations:
    # Crop the face from the image
    face = image[location.top():location.bottom(), location.left():location.right()]
    
    # Extract the face encoding
    encoding = face_recognition.face_encodings(face)[0]
    
    # Append the encoding to the list
    face_encodings_list.append(encoding)

# Assuming you have an existing DataFrame 'known_faces' with columns 'name' and 'encoding'
known_faces = pd.read_csv("path_to_known_faces.csv")

# Convert the DataFrame to a list of known encodings
known_face_encodings = [face_recognition.face_encodings(row['image'])[0] for _, row in known_faces.iterrows()]

# Recognize Faces
matched_names = []
for encoding in face_encodings_list:
    match = None
    min_distance = float('inf')
    
    # Compare the current encoding with all known encodings
    for i, known_encoding in enumerate(known_face_encodings):
        distance = face_recognition.compare_faces([known_encoding], [encoding])[0]
        
        if distance < min_distance:
            min_distance = distance
            match = (i, known_names[i])
    
    # Append the matched name to the list
    matched_names.append(match)

# Display the results
for i, (index, name) in enumerate(matched_names):
    print(f"Face {i+1} is recognized as: {name}")


22:43:35 - At 2025-09-27 :  22:43:35 : qwen2.5-coder:1.5b : To choose the best face recognition model for Python, you should consider several factors such as computational requirements, accuracy, robustness against different lighting conditions, and ease of integration. Here are some popular options:

1. **Dlib**: 
   - Dlib is a powerful library for solving a wide range of machine learning problems, including face detection.
   - It provides a simple API for loading images and detecting faces using the HOG (Histogram of Oriented Gradients) or shape descriptors.
   - Example code:
     ```python
     import dlib

     # Load an image containing faces
     img = dlib.load_image('path_to_image.jpg')

     # Detect faces in the image
     face_detector = dlib.get_frontal_face_detector()
     detected_faces = face_detector(img)

     for face in detected_faces:
         # Draw rectangles around the detected faces
         dlib.draw_rects(img, [face])
         # Display the image with bounding boxes
         cv2.imshow('Detected Faces', img)
         cv2.waitKey(0)
     ```

2. **OpenCV**:
   - OpenCV is a powerful library for computer vision tasks.
   - It includes functions for face detection and landmark extraction, which can be used to recognize faces using machine learning models like LBPH (Local Binary Patterns Histograms) or EigenFaceRecognizer.
   - Example code:
     ```python
     import cv2
     from sklearn.preprocessing import LabelEncoder
     from sklearn.svm import SVC

     # Load an image containing faces
     img = cv2.imread('path_to_image.jpg')

     # Convert the image to grayscale for face detection
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

     # Detect faces in the image
     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
     detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

     for (x, y, w, h) in detected_faces:
         # Draw rectangles around the detected faces
         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

     # Display the image with bounding boxes
     cv2.imshow('Detected Faces', img)
     cv2.waitKey(0)
     ```

3. **Facenet**:
   - Facenet is a pre-trained deep learning model for face recognition.
   - It can be trained on large datasets to recognize faces accurately.
   - Example code (using TensorFlow and Keras):
     ```python
     import tensorflow as tf

     # Load the facenet model
     model = tf.keras.models.load_model('path_to_facenet_model.h5')

     # Load an image containing faces
     img = tf.keras.preprocessing.image.load_img('path_to_image.jpg', target_size=(160, 160))
     img_array = tf.keras.preprocessing.image.img_to_array(img)
     img_array = tf.expand_dims(img_array, axis=0)

     # Make predictions
     predictions = model.predict(img_array)

     # Print the predicted labels
     print(predictions)
     ```

4. **MTCNN**:
   - MTCNN (Multi-Task Cascaded Convolutional Networks) is a state-of-the-art face detector.
   - It can be used for both face detection and landmark extraction.
   - Example code using OpenCV:
     ```python
     import cv2

     # Load an image containing faces
     img = cv2.imread('path_to_image.jpg')

     # Convert the image to grayscale for face detection
     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

     # Detect faces in the image using MTCNN
     detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
     detected_faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

     for (x, y, w, h) in detected_faces:
         # Draw rectangles around the detected faces
         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

     # Display the image with bounding boxes
     cv2.imshow('Detected Faces', img)
     cv2.waitKey(0)
     ```

Choose the model based on your specific requirements and constraints. Dlib is a good choice for simple face detection tasks, while OpenCV provides more comprehensive functionalities and support for machine learning models. Facenet and MTCNN are excellent options for more advanced applications requiring high accuracy. 









