# from inference import get_flower_name
# from commons import get_tensor
from procesing import predict
from io import BytesIO
from PIL import Image
from subprocess import Popen, PIPE
from flask import Flask, request, render_template
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html', value='hi')
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        className, probabil = predict(file)
        return render_template('result.html', image=className, category=probabil)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
