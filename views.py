"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template , request
from FlaskWebProject import app
import os
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
@app.route('/' , methods = ["GET","POST"] )
def home():
    if request.method=="POST":
        file = request.files["file"]
        file.save(os.path.join("uploads",file.filename))
        filename = file.filename
        image = Image.open('uploads/'+ filename)
        model = tensorflow.keras.models.load_model('keras_model.h5')
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        print(prediction)
        for x in prediction:
            if x[0] > 0.50:
              print("fake")
              prediction ="Fake"
            else:
                print("real")
                prediction="Real"
        return render_template("WebPage1.html", message=prediction,)
        return render_template("WebPage1.html", message=image)
    """Renders the home page."""
    return render_template("WebPage1.html", message="upload")








