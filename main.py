import cv2
import imutils
import random
import numpy as np
from imutils import contours
from PIL import Image, ImageOps  # pip install Pillow instead PIL
import streamlit as st
import pytesseract
import os
import oracledb


# to run streamlit run main.py

# Set the page title
st.title("Scanner System")

# Add a text element to the page
st.write("Hello, User!")

uploaded_files = st.file_uploader("Choose a picture", accept_multiple_files=True)

# to text extraction OCR --
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Path to Tesseract executable
tesseract_dir = r'C:\Program Files\Tesseract-OCR\tessdata'

os.environ['TESSDATA_PREFIX'] = tesseract_dir
#--
def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left

      # Create a placeholder rectangle matrix 4x2
    rect = np.zeros((4, 2), dtype = "float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now, compute the difference between the points (y - x),
    # the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack them
    # individually

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
      # Minus 1 for the first pixel
    dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped

def crop_image(input_image_path, output_image_path, x, y, width, height):
    image = Image.open(input_image_path)
    cropped_image = image.crop((x, y, x + width, y + height))
    cropped_image.save(output_image_path)
def add_black_frame(image_path, frame_width,color):
    original_image = Image.open(image_path)
    new_width = original_image.width + 2 * frame_width
    new_height = original_image.height + 2 * frame_width
    new_image = Image.new("RGB", (new_width, new_height),color)
    new_image.paste(original_image, (frame_width, frame_width))
    return new_image

frame_width = 5

def return_id (input_image_path,output_image_path):
    x = 500  # 25  # X-coordinate of the top-left corner of the cropping area
    y = 300  # Y-coordinate of the top-left corner of the cropping area
    width = 360  # Width of the cropping area
    height = 200  # Height of the cropping area

    crop_image(input_image_path, output_image_path, x, y, width, height)
    image = Image.open(output_image_path)

    # Step 4: Character Whitelisting
    whitelist = '0123456789'  # + '٠١٢٣٤٥٦٧٨٩'

    # Step 5: Post-processing
    def filter_text(text):
        filtered_text = ''.join(filter(lambda char: char in whitelist, text))
        return filtered_text

    extracted_text = pytesseract.image_to_string(image, lang='eng',
                                                 config='--psm 6 {tesseract_dir}')  # Use 'ara' for Arabic

    # Manually replace Arabic numeral look-alike characters with actual Arabic numerals

    # extracted_text = re.sub(r'[٠١٢٣٤٥٦٧٨٩]', lambda match: str(ord(match.group()) - ord('٠')), extracted_text)

    # print(extracted_text)

    # Split the text into lines using the newline character
    # lines = extracted_text.split('\n')

    # in state name and id is english

    id = filter_text(extracted_text)  # id


    return id

# extract id --
id_path='id_path.png'
list_id=[]
no_questions=0
try :
    if uploaded_files:
        for IMAGE in uploaded_files :
            id=return_id(IMAGE,id_path)
            list_id.append(id)

        # check no of questions
        less_than_16=st.radio(("is number of questions is less than 17 : "), ('Yes','No'), horizontal=True)
        # take no of questions
        no_questions = st.number_input('What is no of questions ?')
        different_score =list(st.text_input("Enter number of the question with 2 score :").split(','))

        different_score = [int(i) for i in different_score]
except ValueError :
    st.write("please write number of the question with 2 score")
#--

# choose right anwser--
answer=[]
answer.append(st.radio(r"What\'s right answer : Q{}".format(1),(0,1,2,3),horizontal=True))
for i in range(1,int(no_questions)):
    answer.append(st.radio(("Q{} :".format(i+1)),(0,1,2,3),horizontal=True,key=i))
#--

# process scan and extract right answer
def process(answer,input_image_path):
    def show_images(images, titles, kill_later=True):
        for index, image in enumerate(images):
            print(titles)
            cv2.imshow(image)
        cv2.waitKey(0)
        if kill_later:
            cv2.destroyAllWindows()

    # edge detection
    image = cv2.imread(input_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 40,200)
    #show_images([edged], ["Edged"])


    # find contours in edge detected image
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    allContourImage = image.copy()
    cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)
    #print("Total contours found after edge detection {}".format(len(cnts)))
    #show_images([allContourImage], ["All contours from edge detected image"])

    # finding the document contour
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:
            peri = cv2.arcLength(c, closed=True)
            approx = cv2.approxPolyDP(c, epsilon=peri*0.02, closed=True)

            if len(approx) == 4:
                docCnt = approx
                break

    contourImage = image.copy()
    cv2.drawContours(contourImage, [docCnt], -1, (0, 0, 255), 2)
    #show_images([contourImage], ["Outline"])


    # Getting the bird's eye view, top-view of the document
    paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))
    #show_images([paper, warped], ["Paper", "Gray"])


    # Thresholding the document
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #show_images([thresh], ["Thresh"])


    # Finding contours in threshold image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("Total contours found after threshold {}".format(len(cnts)))
    questionCnts = []

    allContourImage = paper.copy()
    cv2.drawContours(allContourImage, cnts, -1, (0, 0, 255), 3)
    #show_images([allContourImage], ["All contours from threshold image"])

    # Finding the questions contours
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if w >= 30 and h >= 30 and ar >= 0.9 and ar <= 1.2:
            # Extract the region of interest (ROI) from the grayscale image
            roi = thresh[y:y + h, x:x + w]

            # Check if there are any non-zero pixels in the ROI
            if cv2.countNonZero(roi) > 0:
                questionCnts.append(c)

    #print("Total questions contours found: {}".format(len(questionCnts)))
    # if len(questionCnts) < answer*4 :
    #     st.write("please enter right photo with the rules")

    questionsContourImage = paper.copy()
    cv2.drawContours(questionsContourImage, questionCnts, -1, (0, 0, 255), 3)
    # show_images([questionsContourImage], ["All questions contours after filtering questions"])

    # Sorting the contours according to the question
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0
    questionsContourImage = paper.copy()

    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        cnts = contours.sort_contours(questionCnts[i: i+4])[0]
        cv2.drawContours(questionsContourImage, cnts, -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)
    #   show_images([cv2.drawContours(questionsContourImage, cnts, -1, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)], ["-----"])
        bubbled = None

        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)

            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
    #       show_images([mask], ["Mask of question {} for row {}".format(j+1, q+1)])
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        color = (255, 0,0)
        k = answer[q]

        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
        if k == bubbled[1] and (q+1) in different_score:
            color = (0, 255, 0)
            correct += 2


        cv2.drawContours(paper, [cnts[k]], -1, color, 3)

    #show_images([questionsContourImage], ["All questions contours with different colors"])
    print((different_score))
    score = (correct )# / 34) * 100
    print("INFO Score: {:.2f}%".format(score))
    cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return paper,image,score
    #show_images([image, paper], ["Original", "exam"])

# process to first part of image

def process1(answer,image_path_original):
   # x = 25  # X-coordinate of the top-left corner of the cropping area
   # y = 295  # Y-coordinate of the top-left corner of the cropping area
   # width = 300  # 650  # Width of the cropping area
   # height = 680  # Height of the cropping area

    x = 0  # 25  # X-coordinate of the top-left corner of the cropping area
    y = 715  # Y-coordinate of the top-left corner of the cropping area
    width = 690  # Width of the cropping area
    height = 1550  # Height of the cropping areaht of the cropping area

    input_image_path = 'out_process1.png'

    crop_image(image_path_original, input_image_path, x, y, width, height)
    image_with_frame = add_black_frame(input_image_path, frame_width, (253, 242, 2))
    # Save the image with frame 1
    image_with_frame.save(input_image_path)
    image_with_frame2 = add_black_frame(input_image_path, frame_width * 3 + 3, "black")
    # Save the image with frame 1
    image_with_frame2.save(input_image_path)

    paper,image,score = process(answer,input_image_path)

    return score ,paper,image

# process to second part of image

def process2(answer,image_path_original):
    #x = 362  # X-coordinate of the top-left corner of the cropping area
    #y = 295  # Y-coordinate of the top-left corner of the cropping area
    #width = 300  # 650  # Width of the cropping area
    #height = 650  # Height of the cropping area

    x = 800  # 25  # X-coordinate of the top-left corner of the cropping area
    y = 715  # Y-coordinate of the top-left corner of the cropping area
    width = 690  # Width of the cropping area
    height = 1550  # Height of the cropping areaht of the cropping area

    input_image_path='out_process2.png'

    crop_image(image_path_original, input_image_path, x, y, width, height)

    image_with_frame = add_black_frame(input_image_path, frame_width, (253, 242, 2))
    # Save the image with frame 1
    image_with_frame.save(input_image_path)
    image_with_frame2 = add_black_frame(input_image_path, frame_width * 3 + 3, "black")
    # Save the image with frame 1
    image_with_frame2.save(input_image_path)

    paper,image,score = process(answer,input_image_path)

    return score , paper,image





connection = oracledb.connect(
    user="test",
    password="test",
    dsn="75.119.149.252/xepdb1")

print("Successfully connected to Oracle Database")
cursor = connection.cursor()

try :
    if st.button('get score'):
        for num , IMAGE in enumerate( uploaded_files):
            # st.image(process1(answer[:16],IMAGE)[2], caption='Original')
            # st.image(process1(answer[:16],IMAGE)[1], caption='exam')

            # st.image(process2(answer[16:], IMAGE)[2], caption='Original')
            # st.image(process2(answer[16:], IMAGE)[1], caption='exam')


            if less_than_16 =='No' :
                score_part1 = process1(answer[:16], IMAGE)[0]

                score_part2 = process2(answer[16:],IMAGE)[0]
                total=score_part1+score_part2
                st.write("the total for :",str(list_id[num])," : ",total)
                # save data to database
                rows = [(list_id[num],total, 100, 100 - total)]

            else :
                score_part1 = process1(answer, IMAGE)[0]

                st.write("the total for :",str(list_id[num])," : ",score_part1)
                # save data to database
                rows = [(list_id[num], score_part1, 100, 100 - score_part1)]

            cursor.executemany("insert into nada (STUENT_ID, SCORE,TOTAL_True,TOTAL_FALSE) values(:2,:2,:2,:2)", rows)
            connection.commit()
            print(cursor.rowcount, "Rows Inserted")

        print("sheets were scored")

except ValueError :
    st.write("please put right photo")
#
# else :
#     st.write("Please upload files")


