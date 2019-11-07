import cv2  # import OpenCV
import numpy as np  # import Numpy

video = cv2.VideoCapture(0)  # Select the image capture device

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Load pre trained model (weights and configuration)

# load file containing object names and correlate with layers in model
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colours = np.random.uniform(0, 255, size=(len(classes), 3))

while True:
    _, img = video.read()  # img is the image frame that we're capturing in this loop
    height, width, channels = img.shape  # get the image dimensions

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    net.setInput(blob)  # Assign the blob object the the input layer of the neural network
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                centre_x = int(detection[0] * width)
                centre_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(centre_x - w / 2)
                y = int(centre_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                # cv2.circle(img, (centre_x, centre_y), 3, (0, 0, 255), 2)  # draw circle at centre

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            object_label = str(classes[class_ids[i]]) + " " + str(round((confidences[0]) * 100, 2)) + "%"

            if (classes[class_ids[i]]) in ["banana", "apple", "orange"]:
            # if 1:
                # print(classes[class_ids[i]])
                # Draw rectangles around object with a object_label
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)
                cv2.putText(img, object_label, (x, y - 30), font, 3, (0, 0, 0), 5)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv2.putText(img, object_label, (x, y - 30), font, 3, (255, 255, 255), 2)
                cv2.circle(img, (centre_x, centre_y), 3, (0, 0, 255), 2)  # draw circle at centre

    img2 = cv2.resize(img, dsize=(1920, 1025))
    cv2.imshow("Display Window", img2)  # Final image is displayed here
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows() 
