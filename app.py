from flask import Flask, render_template, request, send_from_directory
import cv2
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
OUTPUT_FOLDER = 'outputs/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

NUM_QUESTIONS = 5
NUM_CHOICES   = 5
CORRECT_ANSWERS = [1, 2, 0, 2, 4]  # Index of correct answers
ANSWER_LETTERS  = ["A", "B", "C", "D", "E"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'quizImage' not in request.files:
        return 'No file part', 400
    file = request.files['quizImage']
    if file.filename == '':
        return 'No selected file', 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    processed_path, selected_answers, score, accuracy, error = process_quiz_image(filepath)

    if error:
        return f'<h2>Ошибка: {error}</h2><a href="/">Назад</a>', 400

    return render_template(
        'result.html',
        image_path=processed_path,
        selected_answers=selected_answers,
        correct_answers=CORRECT_ANSWERS,
        score=score,
        accuracy=accuracy,
        answer_letters=ANSWER_LETTERS
    )

def reorder_corners(pts):
    pts = pts.reshape((4, 2))
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  
    ordered[2] = pts[np.argmax(s)] 
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  
    ordered[3] = pts[np.argmax(diff)] 
    return ordered

def process_quiz_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, 0, 0,

    img = cv2.resize(img, (700, 900))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                biggest = approx
                max_area = area

    if biggest is not None:
        corners = reorder_corners(biggest)
        width, height = 700, 900
        dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        matrix = cv2.getPerspectiveTransform(corners, dst)
        warped = cv2.warpPerspective(img, matrix, (width, height))
    else:
        warped = img.copy()

    roi_y1, roi_y2 = 80, 880
    roi_x1, roi_x2 = 40, 660
    
    answer_region = warped[roi_y1:roi_y2, roi_x1:roi_x2]
    region_h, region_w = answer_region.shape[:2]

    gray_region = cv2.cvtColor(answer_region, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray_region, 150, 255, cv2.THRESH_BINARY_INV)[1]
    
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    cell_h = region_h // NUM_QUESTIONS
    cell_w = region_w // NUM_CHOICES

    pixel_values = np.zeros((NUM_QUESTIONS, NUM_CHOICES))

    for q in range(NUM_QUESTIONS):
        for c in range(NUM_CHOICES):
            y1, y2 = q * cell_h, (q + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            cell = thresh[y1:y2, x1:x2]
            pixel_values[q][c] = cv2.countNonZero(cell)
            
            cv2.rectangle(warped, (roi_x1 + x1, roi_y1 + y1), 
                          (roi_x1 + x2, roi_y1 + y2), (0, 255, 0), 1)

    selected_answers = [int(np.argmax(pixel_values[q])) for q in range(NUM_QUESTIONS)]

    total_conf = []
    for q in range(NUM_QUESTIONS):
        max_px = np.max(pixel_values[q])
        conf = min(100, (max_px / 400) * 100)
        total_conf.append(conf)
    
    analysis_accuracy = round(sum(total_conf) / NUM_QUESTIONS, 1)

    grading = [1 if selected_answers[q] == CORRECT_ANSWERS[q] else 0 for q in range(NUM_QUESTIONS)]
    score = round((sum(grading) / NUM_QUESTIONS) * 100, 1)

    out_path = os.path.join(app.config['OUTPUT_FOLDER'], 'processed_image.jpg')
    cv2.imwrite(out_path, warped)

    return out_path, selected_answers, score, analysis_accuracy, None

@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)