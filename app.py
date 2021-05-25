from flask import Flask, render_template, request, redirect
import base64
import numpy as np
import cv2
import time
import pytesseract
from imutils.object_detection import non_max_suppression
app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r"U:/program files/Tesseract-OCR/tesseract.exe"


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]      # file storage object
            npimg = np.fromstring(image.read(), np.uint8)    # array
            # convert numpy array to image
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)  # ndarray

            image = east_detect(img)      # ndarray

            _, buffer = cv2.imencode('.png', image)

            image_string = base64.b64encode(buffer)
            image_string = image_string.decode('utf-8')
            print(image_string)
            return render_template("output.html", filestring=image_string)
    return redirect('/')


def east_detect(image):
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    orig = image.copy()

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height: Should be multiple of 32
    (newW, newH) = (320, 320)

    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))

    (H, W) = image.shape[:2]

    net = cv2.dnn.readNet("models/frozen_east_text_detection.pb")

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)

    start = time.time()

    net.setInput(blob)

    (scores, geometry) = net.forward(layerNames)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            # Set minimum confidence as required
            if scoresData[x] < 0.5:
                continue
                # compute the offset factor as our resulting feature maps will
            #  x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    words = []

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = max(int(startX * rW) - 5, 0)
        startY = max(int(startY * rH) - 5, 0)
        endX = int(endX * rW) + 5
        endY = int(endY * rH) + 5

        print("./././.", startX, startY, endX, endY)
        roi = orig[startY:endY, startX:endX]
        words.append((startX, startY, tesseract(roi)))

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY),
                      (endX, endY), (0, 255, 0), 2)

    # Sort bounding boxes according to position and put numbers on it
    words.sort(key=lambda x: x[1])
    id = 1

    for startX, startY, _ in words:
        orig = cv2.putText(orig, str(id), (startX, startY - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        id += 1

    print(words)
    print("Time taken", time.time() - start)
    return orig


def tesseract(image):
    text = pytesseract.image_to_string(
        image, config=("-l eng --oem 1 --psm 8"))
    text = text.split('\n')[0]
    return text
