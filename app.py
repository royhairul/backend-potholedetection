from PIL import Image
import io
import pandas as pd
import numpy as np

from typing import Optional

from ultralytics import YOLO
import ultralytics
from ultralytics.utils.plotting import Annotator, colors

import cv2

print(ultralytics.checks())

# Initialize the models
model_sample_model = YOLO("./models/mymodel/jalanjalan.pt")

def get_image_from_bytes(binary_image: bytes) -> Image:
    """Convert image from bytes to PIL RGB format
    
    Args:
        binary_image (bytes): The binary representation of the image
    
    Returns:
        PIL.Image: The image in PIL RGB format
    """
    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    return input_image

def get_bytes_from_image(image: Image) -> bytes:
    """
    Convert PIL image to Bytes
    
    Args:
    image (Image): A PIL image instance
    
    Returns:
    bytes : BytesIO object that contains the image in JPEG format with quality 85
    """
    return_image = io.BytesIO()
    image.save(return_image, format='JPEG', quality=85)  # save the image in JPEG format with quality 85
    return_image.seek(0)  # set the pointer to the beginning of the file
    return return_image

def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

    Args:
        results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
        labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
        
    Returns:
        predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and class labels.
    """
    # Transform the Tensor to numpy array
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Replace the class number with the class name from the labeles_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict(model: YOLO, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
    """
    Get the predictions of a model on an input image.
    
    Args:
        model (YOLO): The trained YOLO model.
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Make predictions
    predictions = model.predict(source=input_image)
    
    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions

################################# BBOX Func #####################################

# def add_bboxs_on_img(image: Image, predict: pd.DataFrame()) -> Image:
#     """
#     add a bounding box on the image

#     Args:
#     image (Image): input image
#     predict (pd.DataFrame): predict from model

#     Returns:
#     Image: image whis bboxs
#     """

#     # Create an annotator object
#     annotator = Annotator(np.array(image))

#     # sort predict by xmin value
#     predict = predict.sort_values(by=['xmin'], ascending=True)

#     # iterate over the rows of predict dataframe
#     for i, row in predict.iterrows():
        
#         # Calculate width and height of the bounding box
#         width = row['xmax'] - row['xmin']
#         height = row['ymax'] - row['ymin']


#         # create the text to be displayed on image
#         label = f"P: {width:.1f}, L: {height:.1f}"

#         # Get the bounding box coordinates
#         start_point = (int(row['xmin']), int(row['ymin']))
#         end_point = (int(row['xmax']), int(row['ymax']))
        
#         # create the text to be displayed on image
#         text = f"{row['name']}: {int(row['confidence']*100)}%"

#         # get the bounding box coordinates
#         bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        
#         # Define the position for the text (slightly above the box)
#         text_position = (int(row['xmin']), int(row['ymin'] - 10))

        

#         # Add the text label with confidence
#         cv2.putText(annotator, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, color, 1, cv2.LINE_AA)

#         # add the bounding box and text on the image
#         annotator.box_label(bbox, text, color=colors(row['class'], True))
#     # convert the annotated image to PIL image
#     return Image.fromarray(annotator.result())

# ============================================================================

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=1,
          font_thickness=1,
          text_color=(0, 0, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h), font, font_scale, text_color, font_thickness)

    return text_size

# ===========================================================================


def add_bboxs_on_img(image: Image, predict: pd.DataFrame) -> Image:
    """
    Add a bounding box on the image, calculate and display the contour area within the bounding box.

    Args:
    image (Image): input PIL image
    predict (pd.DataFrame): predictions from the model

    Returns:
    Image: image with bounding boxes and area of contours
    """
    
    # Convert PIL image to numpy array (OpenCV format)
    image_np = np.array(image)

    # Sort predictions by xmin value
    predict = predict.sort_values(by=['xmin'], ascending=True)

    # Iterate over the rows of the predictions dataframe
    for i, row in predict.iterrows():
        
        # Calculate width and height of the bounding box
        width = int(row['xmax'] - row['xmin'])
        height = int(row['ymax'] - row['ymin'])

        # Create the text to be displayed on image
        text = f"{row['name']}: {int(row['confidence']*100)}%"

        # Get the bounding box coordinates
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Crop the image inside the bounding box
        cropped_img = image_np[ymin:ymax, xmin:xmax]

        # Convert cropped image to grayscale (needed for contour detection)
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to create a binary image (black and white)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the area of each contour
        contour_area = 0
        for contour in contours:
            contour_area += cv2.contourArea(contour)  # Sum areas of all contours

        # Draw the bounding box on the original image
        start_point = (xmin, ymin)
        end_point = (xmax, ymax)
        color = (0, 255, 0)  # Green color in BGR
        thickness = 4
        cv2.rectangle(image_np, start_point, end_point, color, thickness)

        # Define the position for the text (slightly above the box)
        conf_position = (xmin, ymin)
        # label_position = (xmin, ymax)

        # Display the text: confidence and bounding box label
        w, h = draw_text(image_np, text, pos=conf_position, font_scale=0.5, text_color_bg=color)

        scale_factor = 0.01
        # Convert the contour area from pixels² to meters²
        contour_area_meters2 = contour_area * (scale_factor ** 2)
        # # Add contour area text on the image
        area_text = f"Luas Area: {contour_area_meters2:.2f} m2"
        area_position = (xmin, ymin - 20)  # Position the area text above the bounding box
        w, h = draw_text(image_np, area_text, pos=area_position, font_scale=0.5, text_color_bg=color)

    # Convert the numpy array back to PIL image
    return Image.fromarray(image_np)

##############################################################################
# Fungsi untuk menghitung luas kontur
def calculate_contour_area(image_np: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int) -> float:
    # Potong gambar sesuai bounding box
    cropped_img = image_np[int(ymin):int(ymax), int(xmin):int(xmax)]
    # cropped_img = image_np.crop((xmin, ymin, xmax, ymax))

    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Terapkan thresholding untuk mendapatkan gambar biner
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Temukan kontur dalam gambar
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hitung luas total dari semua kontur
    contour_area = sum(cv2.contourArea(contour) for contour in contours)

    # Konversi dari piksel² ke meter²
    scale_factor = 0.01  # Sesuaikan dengan faktor skala yang sesuai
    contour_area_meters2 = contour_area * (scale_factor ** 2)

    return contour_area_meters2

################################# Models #####################################

def detect_sample_model(input_image: Image) -> pd.DataFrame:
    """
    Predict from sample_model.
    Base on YoloV8

    Args:
        input_image (Image): The input image.

    Returns:
        pd.DataFrame: DataFrame containing the object location.
    """
    predict = get_model_predict(
        model=model_sample_model,
        input_image=input_image,
        save=False,
        image_size=640,
        augment=False,
        conf=0.5,
    )
    return predict