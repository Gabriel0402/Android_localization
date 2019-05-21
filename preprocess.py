import pandas as pd
import numpy as np

X=[]
y=[]
y_x=[]
y_y=[]

def get_data():
    global X,y,y_x,y_y
    df=pd.read_csv('data.csv',sep=',', lineterminator='\n')
    X = df.values[:,1:8]
    y_x= df.values[:,8]
    y_y= df.values[:,9]
    y=df.values[:,8:10]

def process_data(x,y,m,n):
    x_scale= 16/m
    y_scale= 22/n
    res=np.array([])
    for i in range(x.size):
        res=np.append(res,'L'+ str(int(x[i]/x_scale)) + str(int(y[i]/y_scale))) 
    return res

get_data()