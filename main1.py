
import numpy as np
import cv2
from copy import deepcopy
from PIL import Image
import pytesseract as tess
from flask import Flask, render_template, request
import re
from pyrebase import pyrebase

config = {
    # your config here
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/Process', methods=['POST'])
def index():
    filepath = request.form['fname']
    img = cv2.imread(filepath)

    threshold_img = preprocess(img)
    contours = extract_contours(threshold_img)
    ans = cleanAndRead(img, contours)
    finalans = text = re.sub('[^A-Za-z0-9]+', '', ans)
    all = db.child("users").get()

    flag = 0
    for user in all.each():
        b = user.val()
        if finalans == b['vno']:
            flag = 1
            break

    return render_template('index.html', text=finalans, name=b['name'], phone=b['phone'])


@app.route('/register', methods=['POST'])
def register():
    userdata = {"name": request.form['name'],
                "vno": request.form['vno'],
                "phone": request.form['phone'],
                "class": request.form['class']}

    db.child("users").push(userdata)
    return render_template('registersuccess.html')


def preprocess(img):
    cv2.imshow("Input", img)
    cv2.waitKey(0)
    imgBlurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    cv2.imshow("Sobel", sobelx)
    cv2.waitKey(0)
    ret2, threshold_img = cv2.threshold(
        sobelx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("Threshold", threshold_img)
    cv2.waitKey(0)
    return threshold_img


def cleanPlate(plate):
    # print("CLEANING PLATE. . .")
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # thresh= cv2.dilate(gray, kernel, iterations=1)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)

        max_cnt = contours[max_index]
        max_cntArea = areas[max_index]
        x, y, w, h = cv2.boundingRect(max_cnt)

        if not ratioCheck(max_cntArea, w, h):
            return plate, None

        cleaned_final = thresh[y:y+h, x:x+w]
        cv2.imshow("Function Test", cleaned_final)
        return cleaned_final, [x, y, w, h]

    else:
        return plate, None


def extract_contours(threshold_img):
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(17, 3))
    morph_img_threshold = threshold_img.copy()
    cv2.morphologyEx(src=threshold_img, op=cv2.MORPH_CLOSE,
                     kernel=element, dst=morph_img_threshold)
    cv2.imshow("Morphed", morph_img_threshold)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(
        morph_img_threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    return contours


def ratioCheck(area, width, height):
    ratio = float(width) / float(height)
    if ratio < 1:
        ratio = 1 / ratio

    aspect = 4.7272
    min = 15*aspect*15  # minimum area
    max = 125*aspect*125  # maximum area

    rmin = 3
    rmax = 6

    if (area < min or area > max) or (ratio < rmin or ratio > rmax):
        return False
    return True


def isMaxWhite(plate):
    avg = np.mean(plate)
    if(avg >= 115):
        return True
    else:
        return False


def validateRotationAndRatio(rect):
    (x, y), (width, height), rect_angle = rect

    if(width > height):
        angle = -rect_angle
    else:
        angle = 90 + rect_angle

    if angle > 15:
        return False

    if height == 0 or width == 0:
        return False

    area = height*width
    if not ratioCheck(area, width, height):
        return False
    else:
        return True


def cleanAndRead(img, contours):
    # count=0
    for i, cnt in enumerate(contours):
        min_rect = cv2.minAreaRect(cnt)

        if validateRotationAndRatio(min_rect):

            x, y, w, h = cv2.boundingRect(cnt)
            plate_img = img[y:y+h, x:x+w]

            if(isMaxWhite(plate_img)):
                # count+=1
                clean_plate, rect = cleanPlate(plate_img)

                if rect:
                    x1, y1, w1, h1 = rect
                    x, y, w, h = x+x1, y+y1, w1, h1
                    cv2.imshow("Cleaned Plate", clean_plate)
                    cv2.waitKey(0)
                    plate_im = Image.fromarray(clean_plate)
                    text = tess.image_to_string(plate_im, lang='eng')
                    print(text)
                    img = cv2.rectangle(
                        img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imshow("Detected Plate", img)
                    cv2.waitKey(0)
    return text
    # print "No. of final cont : " , count


if __name__ == "__main__":
    # print( "DETECTING PLATE . . .")

    # img = cv2.imread("testData/Final.JPG")

    # if len(contours)!=0:
    # print len(contours) #Test
    # cv2.drawContours(img, contours, -1, (0,255,0), 1)
    # cv2.imshow("Contours",img)
    # cv2.waitKey(0)

    app.run(debug=True)
