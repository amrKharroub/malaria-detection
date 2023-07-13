

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

def model_fn(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=torch.device(device))

    # Set the model to evaluation mode
    model.eval()

    return model

def predict_fn(input_data, model):
    # Define the transformation
    transform = transforms.ToTensor()

    # Apply the transformation
    preprocessed_data = transform(input_data)
    preprocessed_data = preprocessed_data.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessed_data = preprocessed_data.to(device)
    # Perform the prediction
    with torch.no_grad():
        prediction = model(preprocessed_data)

    # Postprocess the prediction as needed
    #postprocessed_prediction = postprocess_output(prediction)

    return prediction

def draw_boxes_on_image(pil_image, predictions):
    decode_label = {1: 'RBC',
                    3: 'trophozoite',
                    4: 'difficult',
                    5: 'ring',
                    6: 'schizont',
                    7: 'gametocyte',
                    2: 'leukocyte'}
    # Create a copy of the original image
    image_with_boxes = pil_image.copy()

    # Create a drawing object
    draw = ImageDraw.Draw(image_with_boxes)

    # Set the font for the labels
    font = ImageFont.truetype("/content/drive/MyDrive/ARIALI 1.TTF", 18)

    # Iterate over the predictions
    for prediction in predictions:
        boxes = prediction["boxes"].tolist()
        labels = prediction["labels"].tolist()
        scores = prediction["scores"].tolist()

        # Iterate over each box
        for box, label, score in zip(boxes, labels, scores):
            # Get the class label text
            class_label = decode_label[label]

            # Draw the bounding box
            draw.rectangle(box, outline="red", width=2)

            # Draw the label text with score
            label_text = f"{class_label}: {score:.2f}"
            draw.text((box[0], box[1]), label_text, fill="red", font=font)

    return image_with_boxes

pil_image = Image.open("/content/drive/MyDrive/00a02700-2ea2-4590-9e15-ffc9160fd3de.png")

model = model_fn('/content/drive/MyDrive/model_f.pt')
predictions = predict_fn(pil_image, model)

out_image = draw_boxes_on_image(pil_image, predictions)

out_image.show()

