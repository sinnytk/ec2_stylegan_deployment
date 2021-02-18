import json
import boto3
from io import BytesIO
import os
from torch import load as load_model
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_file
from flask_bootstrap import Bootstrap
from cyclegan import model
from PIL import Image
UPLOAD_FOLDER = 'tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
Bootstrap(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

s3 = boto3.client("s3", aws_access_key_id=os.environ.get("ACCESS_ID"),
                  aws_secret_access_key=os.environ.get("ACCESS_KEY"))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template("index.html")


@app.route("/convert", methods=['POST'])
def convert():
    print(request)
    if request.method == 'POST':
        file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        im = model.single_test(os.path.join(
            app.config['UPLOAD_FOLDER'], filename))
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_pil = Image.fromarray(im)
        in_mem_file = BytesIO()
        image_pil.save(in_mem_file, format="png")
        in_mem_file.seek(0)
        s3.upload_fileobj(
            in_mem_file,  # This is what i am trying to upload
            "cycleganapp",
            filename,
            ExtraArgs={
                'ACL': 'public-read'
            }
        )
    return jsonify({"image_link": "https://cycleganapp.s3-ap-southeast-1.amazonaws.com/"+filename})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
