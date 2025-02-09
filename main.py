from flask import Flask, render_template, request # type: ignore
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import PIL
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np	

print(PIL.__version__)
# Paksa penggunaan CPU saja
tf.config.set_visible_devices([], 'GPU')
app = Flask(__name__)

dic = {0 : 'Cabai Normal',  
       1 : 'Cabai Matang',
       2 : 'Cabai Patek1', 
	   }

model = load_model('static/model/Cendekia-Penyakit dan Kematangan Cabai-95.83.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)

	pred = model.predict(i)  # Menggunakan predict()
	p = np.argmax(pred, axis=1)[0]  # Ambil indeks kelas dengan nilai tertinggi
	return dic[p]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("classification.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("classification.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))