from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
@app.route('/home', methods=['POST', 'GET'])
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST', 'GET'])
def results():
    print('HI')
    if 'file' not in request.files:
        print('No file attached...')
        return 'im finished'
    f = request.files['file']
    print('File received!')
    print(f)

        
    return 'hi'
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')