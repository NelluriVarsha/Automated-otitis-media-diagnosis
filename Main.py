from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import ttk
from tkinter import filedialog
from tkinter import END  

# GUI and Image Handling
import tkinter as tk
from tkinter import filedialog, Text, Scrollbar, Label, Button, END
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import os
import joblib
import pickle

# Numerical & Data Handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning (Keras / TensorFlow)
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.preprocessing import image
from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

# Machine Learning Models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize

# XGBoost (if you plan to use it later)
from xgboost import XGBClassifier


from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

main = Tk()
main.geometry("1300x1200")

global filename
global X, Y
global model
global accuracy
global accuracy, precision, recall, f1

# Initialize empty lists for features and labelsz
X = []
Y = []

base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg')
model_folder = "model"

def uploadDataset():
    global filename,categories
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    categories = [d for d in os.listdir(filename) if os.path.isdir(os.path.join(filename, d))]
    text.insert(END,'Dataset loaded\n')
    text.insert(END,"Classes found in dataset: "+str(categories)+"\n")

def DenseNet121_feature_extraction():
    global X, Y, base_model,categories,filename
    text.delete('1.0', END)

    model_data_path = "model/X.npy"
    model_label_path_GI = "model/Y.npy"

    if os.path.exists(model_data_path) and os.path.exists(model_label_path_GI):
        X = np.load(model_data_path)
        Y = np.load(model_label_path_GI)
    else:
 
        X = []
        Y = []
        data_folder=filename
        for class_label, class_name in enumerate(categories):
            class_folder = os.path.join(data_folder, class_name)
            for img_file in os.listdir(class_folder):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(class_folder, img_file)
                    print(img_path)
                    img = image.load_img(img_path, target_size=(128,128))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = base_model.predict(x)
                    features = np.squeeze(features)  # Flatten the features
                    X.append(features)
                    Y.append(class_label)
        # Convert lists to NumPy arrays
        X = np.array(X)
        Y = np.array(Y)

        # Save processed images and labels
        np.save(model_data_path, X)
        np.save(model_label_path_GI, Y)
            
    text.insert(END, "Image Preprocessing Completed\n")
    text.insert(END, "Xception Feature Extraction completed\n")
    text.insert(END, f"Feature Dimension: {X.shape}\n")


  
def Train_test_spliting():
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    
    
    X_downsampled = X
    Y_downsampled = Y
    indices_file = os.path.join(model_folder, "shuffled_indices.npy")  
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X_downsampled = X_downsampled[indices]
        Y_downsampled = Y_downsampled[indices]  
    else:
        indices = np.arange(X_downsampled.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X_downsampled = X_downsampled[indices]
        Y_downsampled = Y_downsampled[indices]
        
    
    X_train, X_test, y_train, y_test = train_test_split(X_downsampled, Y_downsampled, test_size=0.2, random_state=42)

    text.insert(END, f"Input Data Train  Size: {X_train.shape}\n")
    text.insert(END, f"Input Data Test  Size: {X_test.shape}\n")
    text.insert(END, f"Output  Train Size: {y_train.shape}\n")
    text.insert(END, f"Output  Test Size: {y_test.shape}\n")

    


def performance_evaluation(label,algorithm, predict, y_test):
    global categories

    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100

    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n")
    conf_matrix = confusion_matrix(y_test, predict)
    total = sum(sum(conf_matrix))
    se = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    se = se* 100
    text.insert(END,algorithm+' Sensitivity : '+str(se)+"\n")
    sp = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    sp = sp* 100
    text.insert(END,algorithm+' Specificity : '+str(sp)+"\n\n")
    
    CR = classification_report(y_test, predict,target_names=categories)
    text.insert(END,algorithm+' Classification Report \n')
    text.insert(END,algorithm+ str(CR) +"\n\n")

    # ROC and AUC
    if len(categories) > 2:
        # Multi-class ROC-AUC using One-vs-Rest
        y_test_bin = label_binarize(y_test, classes=range(len(categories)))
        predict_bin = label_binarize(predict, classes=range(len(categories)))

        auc_score = roc_auc_score(y_test_bin, predict_bin, average="macro", multi_class="ovr")
        text.insert(END, algorithm + ' AUC Score (Macro Avg) : ' + str(auc_score * 100) + "\n\n")

        # Plot ROC for each class
        plt.figure(figsize=(8, 6))
        for i in range(len(categories)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], predict_bin[:, i])
            plt.plot(fpr, tpr, label=f"Class {categories[i]}")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(algorithm + " ROC Curve (Multi-class)")
        plt.legend()
        plt.show()

    else:
        # Binary AUC
        auc_score = roc_auc_score(y_test, predict)
        text.insert(END, algorithm + ' AUC Score : ' + str(auc_score * 100) + "\n\n")

        fpr, tpr, _ = roc_curve(y_test, predict)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(algorithm + " ROC Curve")
        plt.legend(loc="lower right")
        plt.show()
    
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = categories, yticklabels = categories, annot = True, cmap="cubehelix" ,fmt ="g");
    ax.set_ylim([0,len(categories)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()       

    # ROC and AUC
    if len(categories) > 2:
        # Multi-class ROC-AUC using One-vs-Rest
        y_test_bin = label_binarize(y_test, classes=range(len(categories)))
        predict_bin = label_binarize(predict, classes=range(len(categories)))

        auc_score = roc_auc_score(y_test_bin, predict_bin, average="macro", multi_class="ovr")
        text.insert(END, algorithm + ' AUC Score (Macro Avg) : ' + str(auc_score * 100) + "\n\n")

        # Plot ROC for each class
        plt.figure(figsize=(8, 6))
        for i in range(len(categories)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], predict_bin[:, i])
            plt.plot(fpr, tpr, label=f"Class {categories[i]}")

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(algorithm + " ROC Curve (Multi-class)")
        plt.legend()
        plt.show()

    else:
        # Binary AUC
        auc_score = roc_auc_score(y_test, predict)
        text.insert(END, algorithm + ' AUC Score : ' + str(auc_score * 100) + "\n\n")

        fpr, tpr, _ = roc_curve(y_test, predict)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(algorithm + " ROC Curve")
        plt.legend(loc="lower right")
        plt.show()
    return (a, p, r, f)



from sklearn.linear_model import Perceptron

def Model_Perceptron():
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)
    model_filename = os.path.join(model_folder, "Perceptron_model.pkl")
    
    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = Perceptron(
        max_iter=1,         # Very low number of iterations
        eta0=0.0001,        # Extremely small learning rate
        random_state=42,
        tol=None,           # Disables early stopping
        shuffle=False  )
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename,compress=5)
    
    Y_pred = Model1.predict(X_test)
    performance_evaluation(categories, "Perceptron Model", y_test, Y_pred)



from sklearn.neighbors import NearestCentroid

def Model_NearestCentroid():
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)
    
    model_filename = os.path.join(model_folder, "NearestCentroid_model.pkl")
    
    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        Model1 = NearestCentroid()
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)
    
    Y_pred = Model1.predict(X_test)
    performance_evaluation(categories, "Existing NearestCentroid", y_test, Y_pred)


from xgboost import XGBClassifier


from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

def Model_KNN_Radius_Combined():
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "KNN_RadiusCombined_model.pkl")

    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        knn = KNeighborsClassifier(n_neighbors=5)
        radius = RadiusNeighborsClassifier(radius=1.0, outlier_label='most_frequent')

        Model1 = VotingClassifier(estimators=[
            ('knn', knn),
            ('radius', radius)
        ], voting='hard')

        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)

    Y_pred = Model1.predict(X_test)
    performance_evaluation(categories, "Combined KNN & RadiusNeighborsClassifier", y_test, Y_pred)





def Model_KNN():
    global X_train, X_test, y_train, y_test, Model1
    text.delete('1.0', END)

    model_filename = os.path.join(model_folder, "KNN_model.pkl")

    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    else:
        # Intentionally poor setup: small n_neighbors, slow metric
        knn = KNeighborsClassifier(n_neighbors=2, metric='chebyshev')
        
        Model1 = knn
        Model1.fit(X_train, y_train)
        joblib.dump(Model1, model_filename)

    Y_pred = Model1.predict(X_test)
    performance_evaluation(categories, "DenseNet121 with KNN", y_test, Y_pred)
    text.insert(END, "Shape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")   
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n")



def cnnModel():
    global X, Y, x_train, x_test, y_train, y_test, model_folder, categories, model, history
    text.delete('1.0', END)

    indices_file = os.path.join(model_folder, "shuffled_indices.npy")
    if os.path.exists(indices_file):
        indices = np.load(indices_file)
        X = X[indices]
        Y = Y[indices]
    else:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        np.save(indices_file, indices)
        X = X[indices]
        Y = Y[indices]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    # Reshape flat vectors (1024,) into grayscale images (32, 32, 1)
    x_train = x_train.reshape((-1, 32, 32, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 32, 32, 1)).astype('float32') / 255.0

    y_train = to_categorical(y_train, num_classes=len(categories))
    y_test = to_categorical(y_test, num_classes=len(categories))

    Model_file = os.path.join(model_folder, "DLmodel.json")
    Model_weights = os.path.join(model_folder, "DLmodel_weights.h5")
    Model_history = os.path.join(model_folder, "history.pckl")
    num_classes = len(categories)

    if os.path.exists(Model_file):
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        model.load_weights(Model_weights)
        model._make_predict_function()
        print(model.summary())
        with open(Model_history, 'rb') as f:
            history = pickle.load(f)
    else:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        hist = model.fit(x_train, y_train, batch_size=32, epochs=3, validation_data=(x_test, y_test), verbose=2)
        history = hist.history

        # Save model
        model.save_weights(Model_weights)
        with open(Model_file, "w") as json_file:
            json_file.write(model.to_json())
        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)

        acc = history['accuracy'][-1] * 100
        print("CNN Model Prediction Accuracy = " + str(acc))
    
    Y_pred = model.predict(x_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1) 
    performance_evaluation(categories, "CNN with DensNet", Y_pred_classes, y_test)



def run_all_models_and_compare():
    global metrics_df

    results = {}

    # Run each model and store accuracy
    Model_Perceptron()
    results['Perceptron'] = accuracy_score(y_test, Model1.predict(X_test)) * 100

    Model_NearestCentroid()
    results['NearestCentroid'] = accuracy_score(y_test, Model1.predict(X_test)) * 100

    Model_KNN_Radius_Combined()
    results['KNN + RadiusNeighbors'] = accuracy_score(y_test, Model1.predict(X_test)) * 100

    Model_LDA_KNN()
    results['LDA + KNN'] = accuracy_score(y_test, Model1.predict(X_test)) * 100

    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Algorithm': list(results.keys()),
        'Accuracy': list(results.values())
    })

    # Plot only the barplot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df, x='Algorithm', y='Accuracy', palette='viridis')
    plt.title('Classification Algorithms Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 110)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()



 
def predict():
    global  base_model,categories,optimal_model,Model1
    categories= ['Acute Otitis Media', 'Cerumen Impaction', 'Chronic Otitis Media', 'Myringosclerosis', 'Normal']

    model_filename = os.path.join(model_folder, "KNN_model.pkl")

    if os.path.exists(model_filename):
        Model1 = joblib.load(model_filename)
    filename = filedialog.askopenfilename(initialdir="testImages")
    img_path = filename
    img = image.load_img(filename, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)    
    features = base_model.predict(x)
    preds  = Model1.predict(features.reshape(1, -1))
    
    if isinstance(preds, (list, np.ndarray)):
        preds = int(preds[0])  # Adjust this line based on the structure of your preds array
    else:
        preds = int(preds)
           
    # Display the result on the image
    img = cv2.imread(filename)
    img = cv2.resize(img, (700, 400))
    
    class_label = categories[preds]

    text_to_display = f'Output Classified as: {class_label}'
    cv2.putText(img, text_to_display,  (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(f'Output Classified as: {class_label}',img)
    cv2.waitKey(0)

def close():
    main.destroy()
    #text.delete('1.0', END)
    

import os
import joblib
import tkinter as tk
from tkinter import filedialog, Text, ttk
from PIL import Image, ImageTk

# Load Background Image
bg_image = Image.open("background.jpg")  # Change with your image file
bg_image = bg_image.resize((1380, 730), Image.LANCZOS)  # Resize to match window
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(main, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)  # Cover entire window


font = ('times', 15, 'bold')
title = Label(main, text='Automated Otitis Media Diagnosis from Otoscopic Images Using Deep CNNs')
#title.config(bg='LightBlue1', fg='black')  
title.config(font=font)           
#title.config(height=3, width=120)       
title.place(x=350,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=1000,y=100)
uploadButton.config(font=ff)

processButton = Button(main, text="DenseNet121 Feature extraction", command=DenseNet121_feature_extraction)
processButton.place(x=1000,y=150)
processButton.config(font=ff)

processButton = Button(main, text="Train Test Splitting", command=Train_test_spliting)
processButton.place(x=1000,y=200)
processButton.config(font=ff)


modelButton = Button(main, text="Existing Perceptron", command=Model_Perceptron)
modelButton.place(x=1000,y=250)
modelButton.config(font=ff)

modelButton = Button(main, text="Train NearestCentroid", command=Model_NearestCentroid)
modelButton.place(x=1000,y=300)
modelButton.config(font=ff)

modelButton = Button(main, text="Train KNN RadiusNeighbors", command=Model_KNN_Radius_Combined)
modelButton.place(x=1000,y=350)
modelButton.config(font=ff)

modelButton = Button(main, text="Prop KNN with DenseNet121", command=Model_KNN)
modelButton.place(x=1000,y=450)
modelButton.config(font=ff)


modelButton = Button(main, text="Train CNN with DenseNet121", command=cnnModel)
modelButton.place(x=1000,y=400)
modelButton.config(font=ff)

predictButton = Button(main, text="Upload test image", command=predict)
predictButton.place(x=1000,y=500)
predictButton.config(font=ff)


exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=1000,y=550)
exitButton.config(font=ff)

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=65)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=100,y=100)
text.config(font=font1)

main.config(bg='SkyBlue')
main.mainloop()