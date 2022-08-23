from flask import Flask
from flask import request, redirect
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import json
import os
import secure
from flask_httpauth import HTTPBasicAuth

app = Flask(__name__)
auth = HTTPBasicAuth();

USER_DATA = {
    os.getenv("API_WEBAPP_USERNAME"): os.getenv("API_WEBAPP_PASSWORD")
}

hsts_value = secure.StrictTransportSecurity().include_subdomains().preload().max_age(31536000)
secure_headers = secure.Secure(hsts=hsts_value)

@app.before_request
def redirect_http_requests():
    if not request.url.startswith('https'):
        return redirect(request.url.replace('http', 'https', 1))

@app.after_request
@app.middleware("https")
def set_secure_headers(response):
    secure_headers.framework.flask(response)
    return response

@auth.verify_password
def verify(username, password):
    if not (username and password):
        return False
    return USER_DATA.get(username) == password

@app.route("/")
@auth.login_required
def hello():
    return "Try predicting the performance of a PC build!"

@app.route('/predict', methods = ['POST'])
@auth.login_required
def predict():
    if request.method == 'POST':
        data = request.data

        #data = json.loads(data)
        data = json.loads(json.loads(data)[0])

        # remove parentheses
        data["CPU"] = data["CPU"].replace("(", "")
        data["Motherboard"] = data["Motherboard"].replace("(", "")
        data["GPU"] = data["GPU"].replace("(", "")
        data["CPU"] = data["CPU"].replace(")", "")
        data["Motherboard"] = data["Motherboard"].replace(")", "")
        data["GPU"] = data["GPU"].replace(")", "")

        # remove whitespace
        data['GPU'] = data['GPU'].replace(" ", "")
        data['CPU'] = data['CPU'].replace(" ", "")
        data['Motherboard'] = data['Motherboard'].replace(" ", "")

        #score = preditScore(['CPU Name:Intel Core i7-6950X', 'Motherboard:ASUS RAMPAGE V EDITION 10', 'OS:Windows 10  (build 15063)', 'GPU Name:GeForce GTX 1080 Ti  (GP102)'])
        score = preditScore(['CPUName:' + str(data["CPU"]), 'Motherboard:' + str(data["Motherboard"]), 'OS:Windows 10  (build 15063)', 'GPUName:' + str(data["GPU"])])


        return str(score)

@app.route('/suggest', methods = ['POST'])
@auth.login_required
def suggest():
    if request.method == 'POST':

        data = request.data

        #data = json.loads(data)
        data = json.loads(json.loads(data)[0])

        suggestedCpus = data['suggestedCpus']
        currentCpu = data['currentCpu']
        currentGpu = data['currentGpu']
        suggestedGpus = data['suggestedGpus']
        currentScore = data['currentScore']

        scoresSuggestedCpus = []
        scoresSuggestesGpus = []

        for suggestedCpu in suggestedCpus:
            label = suggestedCpu

            # remove parentheses
            suggestedCpu = suggestedCpu.replace("(", "")
            data["Motherboard"] = data["Motherboard"].replace("(", "")
            currentGpu = currentGpu.replace("(", "")
            suggestedCpu = suggestedCpu.replace(")", "")
            data["Motherboard"] = data["Motherboard"].replace(")", "")
            currentGpu = currentGpu.replace(")", "")

            # remove whitespace
            currentGpu = currentGpu.replace(" ", "")
            suggestedCpu = suggestedCpu.replace(" ", "")
            data['Motherboard'] = data['Motherboard'].replace(" ", "")

            score = preditScore(['CPUName:' + str(suggestedCpu), 'Motherboard:' + str(data["Motherboard"]), 'OS:Windows 10  (build 15063)', 'GPUName:' + str(currentGpu)])

            increase = (score / float(currentScore))*100 - 100

            if increase > 0:
                scoresSuggestedCpus.append([str(label), str(round(increase,2))])

        for suggestedGpu in suggestedGpus:
            label = suggestedGpu

            # remove parentheses
            currentCpu = currentCpu.replace("(", "")
            data["Motherboard"] = data["Motherboard"].replace("(", "")
            suggestedGpu = suggestedGpu.replace("(", "")
            currentCpu = currentCpu.replace(")", "")
            data["Motherboard"] = data["Motherboard"].replace(")", "")
            suggestedGpu = suggestedGpu.replace(")", "")

            # remove whitespace
            suggestedGpu = suggestedGpu.replace(" ", "")
            currentCpu = currentCpu.replace(" ", "")
            data['Motherboard'] = data['Motherboard'].replace(" ", "")

            score = preditScore(['CPUName:' + str(currentCpu), 'Motherboard:' + str(data["Motherboard"]), 'OS:Windows 10  (build 15063)', 'GPUName:' + str(suggestedGpu)])
            
            increase = (score / float(currentScore))*100 - 100

            if increase > 0:
                scoresSuggestesGpus.append([str(label), str(round(increase,2))])

        return json.dumps([scoresSuggestedCpus, scoresSuggestesGpus])
    

@app.route('/csgo', methods = ['POST'])
@auth.login_required
def csgo():
    if request.method == 'POST':

        data = request.data
        #data = json.loads(data)
        data = json.loads(json.loads(data)[0])

        file = os.path.join(os.path.dirname(__file__), "performance.xls")
        df = pd.read_excel(file)

        csgoModel = np.poly1d(np.polyfit(df['Score'], df['CSGO'], 2))

        fps = csgoModel(float(data["Score"]))

        return str(fps)

@app.route('/overwatch', methods = ['POST'])
@auth.login_required
def overwatch():
    if request.method == 'POST':

        data = request.data
        #data = json.loads(data)
        data = json.loads(json.loads(data)[0])

        file = os.path.join(os.path.dirname(__file__), "performance.xls")
        df = pd.read_excel(file)

        overwatchModel = np.poly1d(np.polyfit(df['Score'], df['Overwatch'], 2))

        fps = overwatchModel(float(data["Score"]))

        return str(fps)

@app.route('/pubg', methods = ['POST'])
@auth.login_required
def pubg():
    if request.method == 'POST':

        data = request.data
        #data = json.loads(data)
        data = json.loads(json.loads(data)[0])

        file = os.path.join(os.path.dirname(__file__), "performance.xls")
        df = pd.read_excel(file)
        
        pubgModel = np.poly1d(np.polyfit(df['Score'], df['PUBG'], 2))

        fps = pubgModel(float(data["Score"]))

        return str(fps)

@app.route('/fortnite', methods = ['POST'])
@auth.login_required
def fortnite():
    if request.method == 'POST':

        data = request.data
        #data = json.loads(data)
        data = json.loads(json.loads(data)[0])

        file = os.path.join(os.path.dirname(__file__), "performance.xls")
        df = pd.read_excel(file)
        
        fortniteModel = np.poly1d(np.polyfit(df['Score'], df['Fortnite'], 2))

        fps = fortniteModel(float(data["Score"]))

        return str(fps)

@app.route('/gtav', methods = ['POST'])
@auth.login_required
def gtav():
    if request.method == 'POST':

        data = request.data
        #data = json.loads(data)
        data = json.loads(json.loads(data)[0])

        file = os.path.join(os.path.dirname(__file__), "performance.xls")
        df = pd.read_excel(file)
        df.head()
        gtavModel = np.poly1d(np.polyfit(df['Score'], df['GTAV'], 2))

        fps = gtavModel(float(data["Score"]))

        return str(fps)


def preditScore(x_toPredict):

    file = os.path.join(os.path.dirname(__file__), "data.csv")
    dataset = pd.read_csv(file, error_bad_lines=False, sep=';')

    del dataset["Column 1"]
    del dataset["Column 2"]
    del dataset["Summary"]
    del dataset["Column 5"]
    del dataset["Column 7"]
    del dataset["Column 8"]
    del dataset["Column 11"]
    del dataset["Column 12"]
    del dataset["Column 15"]
    del dataset["GPU Vendor"]

    # remove parentheses
    dataset['GPU'] = dataset['GPU'].str.replace("(", "")
    dataset['CPU'] = dataset['CPU'].str.replace("(", "")
    dataset['Motherboard'] = dataset['Motherboard'].str.replace("(", "")
    dataset['GPU'] = dataset['GPU'].str.replace(")", "")
    dataset['CPU'] = dataset['CPU'].str.replace(")", "")
    dataset['Motherboard'] = dataset['Motherboard'].str.replace(")", "")

    # remove whitespace
    dataset['GPU'] = dataset['GPU'].str.replace(" ", "")
    dataset['CPU'] = dataset['CPU'].str.replace(" ", "")
    dataset['Motherboard'] = dataset['Motherboard'].str.replace(" ", "")

    X = dataset.iloc[:, 1:5].values #variables
    y = dataset.iloc[:, 0].values #labels

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    le = LabelEncoder()

    # Encode CPU
    le.fit(X[:, 0].astype(str))
    X[:, 0] = le.transform(X[:, 0].astype(str))
    x_toPredict[0] = le.transform([x_toPredict[0]])[0]

    le = LabelEncoder()

    #Encode Motherboard
    le.fit(X[:, 1].astype(str))
    X[:, 1] = le.transform(X[:, 1].astype(str))
    x_toPredict[1] = le.transform([x_toPredict[1]])[0]

    le = LabelEncoder()

    #Encode OS
    le.fit(X[:, 2].astype(str))
    X[:, 2] = le.transform(X[:, 2].astype(str))
    x_toPredict[2] = le.transform([x_toPredict[2]])[0]

    le = LabelEncoder()

    #Encode GPU
    le.fit(X[:, 3].astype(str))
    X[:, 3] = le.transform(X[:, 3].astype(str))
    x_toPredict[3] = le.transform([x_toPredict[3]])[0]

    # optimise training data
    scaler = StandardScaler()
    scaler.fit(X)

    X = scaler.transform(X)
    x_toPredict = scaler.transform([x_toPredict])

    clf = RandomForestRegressor(max_depth=16, random_state=0)
    clf.fit(X, y)

    y_pred = clf.predict(x_toPredict)[0]

    return y_pred


if __name__ == '__main__':
    app.run(threaded=True, port=5000)