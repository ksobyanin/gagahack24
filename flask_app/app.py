#!flask/bin/python
from flask import Flask, request, jsonify, render_template
from PIL import Image
import json
from io import BytesIO
import os
import base64
from ocr import OCR
import numpy as np


def inint_model():
    model = OCR('best_text_boxes.pt', 'best_class.pt')
    return model


app = Flask(__name__)
model = inint_model()

@app.route('/detect', methods=['POST'])
def process_image():
    base64_image_data = request.json['image']
    if base64_image_data.startswith('data:image/jpeg;base64,'):
        base64_image_data = base64_image_data[len('data:image/jpeg;base64,'):]

        try:
            image_data = base64.b64decode(base64_image_data)
        
        except Exception as e:
            return jsonify({"error": str(e)})

    # Open the image from the decoded data

    img = Image.open(BytesIO(image_data))
    img.save('image.jpg')


    # print('image, loaded', img.size, 'starting inference','array shape', array.shape)
    
    result = json.loads(model.predict_and_return_json('image.jpg'))
    print(result)
    return jsonify({
        'confidence': result["confidence"],
        'class': result['type'], 
        'page': result['page_number'],
        'series': result['series'],
        'number': result['number'],
    })


@app.route('/')
def index():
    return render_template('index.html') 

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    