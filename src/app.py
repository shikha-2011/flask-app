import torch
from flask_ngrok import run_with_ngrok
from flask import Flask
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for,flash
import string
import os
from CRNN.utils.utils import strLabelConverter, decode_prediction
from inference import dataextraction, ImageProcessing

alphabet = string.digits
label_converter = strLabelConverter(alphabet)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UPLOAD_FOLDER = '/content/static/uploads/'


app = Flask(__name__)
run_with_ngrok(app)   #starts ngrok when the app is run
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "shikhaaa"

@app.route('/', methods=['GET', 'POST'])
def hello_world():
  if request.method == 'GET':
    return render_template('home.html', value='hi')
  if request.method == 'POST':
    print(request.files)
    file = request.files['file']
    filename = file.filename
    img = file.read()
    
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    flash('Image successfully uploaded and displayed')

    img = ImageProcessing(img)
    pred_text = dataextraction(img, device, label_converter)
    print(pred_text)
    return render_template('result.html', reading=pred_text, filename=filename)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
  app.run()
