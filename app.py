# Importing required libs
from flask import Flask, render_template, request
from model import preprocess_img, predict_result
from werkzeug.utils import secure_filename

# Instantiating flask app
app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = "static/"

# Home route
@app.route("/")
def main():
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            f = request.files['file']
            filename = secure_filename(f.filename)
            f.save(app.config['UPLOAD_FOLDER'] + filename)
            img_path = app.config['UPLOAD_FOLDER'] + filename
            img = preprocess_img(img_path)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))

    except:
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
