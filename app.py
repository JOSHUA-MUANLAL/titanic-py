from flask import Flask, request, render_template
import numpy as np
import titanic as tic


app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=["POST"])
def play():
    pclass=int(request.form['pclass'])
    sex=int(request.form["gender"])
    age=int(request.form['age'])
    sibsp=int(request.form['sibsp'])
    parch=float(request.form['parch'])
    fare=float(request.form['fare'])
    embark=int(request.form['embarked'])
    
    data=np.array([[pclass,sex,age,sibsp,parch,fare,embark]])
    t=tic.titanic()
    result=t.entry(data)
    return render_template('index.html',result=result)


if __name__ == '__main__':  
   app.run(host="0.0.0.0",port=5000) 
    
    
    
    
