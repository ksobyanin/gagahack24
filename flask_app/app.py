#!flask/bin/python
from flask import Flask, request, jsonify, render_template
from PIL import Image
from io import BytesIO
import os
import base64

# conf = OmegaConf.load('/service/model/config.yaml')

def model_init():
    # model = YOLO(conf.obb_model_path)
    # return model
    pass


app = Flask(__name__)

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

    print('image, loaded', img.size, 'starting inference')
    # results = model(img)
    # print(results, file=sys.stderr)

    # print(results[0].obb, file=sys.stderr)
    return jsonify({
        'confidence': 0.9,
        'class': 'passport', 
        'page': 0,
        'series': 0,
        'number': 0,
    })
   



@app.route('/')
def index():
    return render_template('index.html') 

if __name__ == '__main__':
    model = model_init()
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
    