from flask import Flask, request, make_response

app = Flask(__name__)

@app.route("/")
def home():
    response = make_response('''
    <form action="/login" method="post">
        Username: <input name="username"><br>
        Password: <input name="password" type="password"><br>
        <input type="submit">
    </form>
    ''')
    response.headers['Strict-Tranport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

@app.route("/login", methods=["POST"])
def login():
    return "Secure Login Successful"

app.run(host="0.0.0.0",port=5000,
ssl_context=('cert.pem','key.pem'))
