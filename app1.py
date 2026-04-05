from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def home():
    return '''
    <form action="/login" method="post">
        Username: <input name="username"><br>
        Password: <input name="password" type="password"><br>
        <input type="submit">
    </form>
    '''

@app.route("/login", methods=["POST"])
def login():
    return "Login Successful (HTTP - Insecure)"

app.run(host="0.0.0.0", port=5000)
