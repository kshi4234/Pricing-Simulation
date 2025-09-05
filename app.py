from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def results():
    return render_template('results.html')
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')