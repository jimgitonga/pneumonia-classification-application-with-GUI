import os
import sys
from os import truncate
from tkinter import *

from tkinter import Frame, Tk, Text, Menu, Label, Entry, Button
from tkinter import filedialog
from tkinter import messagebox as mb
import tkinter as tk
# from tkinter.constants import END

from tkinter import ttk
from PIL import Image
import numpy as np
import cv2


from PIL import ImageFile
import PIL.Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import glob
import matplotlib.pyplot as plt
import torch
import time
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
import PIL
from torch.optim import lr_scheduler
import copy
import json

# from os.path import exists
from testing_code import get_device
# from  wrapping import 
# import photogallery
import pickle

from functools import partial


import sqlite3
global acc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True

print(device)


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class_names = ["NORMAL", "PNEUMONIA"]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ImageFile.LOAD_TRUNCATED_IMAGES = True

print(device)


def load_checkpoint(filepath):
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    checkpoint = torch.load(filepath,map_location=lambda storage, loc: storage, pickle_module=pickle)
    model = models.resnet152()

    # our input_size matches the in_features of pretrained model
    input_size = 2048
    output_size = 2

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(2048, 1024)),
        ('relu', nn.ReLU()),
        #('dropout1', nn.Dropout(p=0.2)),
        ('fc2', nn.Linear(1024, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    # Replacing the pretrained model classifier with our classifier
    model.fc = classifier

    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['class_to_idx']


# Get index to class mapping
loaded_model, class_to_idx = load_checkpoint(
    './pneumo_jim90.pth')
idx_to_class = {v: k for k, v in class_to_idx.items()}
print("classes are :", idx_to_class)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model

    size = 256, 256
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.

    imgA = npImage[:, :, 0]
    imgB = npImage[:, :, 1]
    imgC = npImage[:, :, 2]

    imgA = (imgA - 0.485)/(0.229)
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)

    npImage[:, :, 0] = imgA
    npImage[:, :, 1] = imgB
    npImage[:, :, 2] = imgC

    npImage = np.transpose(npImage, (2, 0, 1))

    return npImage


def predict(image_path, topk=2):

    global acc
    global Cell
    # image_path = 'test.jpg'
    model = loaded_model
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file

    image = torch.FloatTensor(
        [process_image(Image.open(image_path).convert('RGB'))])
    model.eval()
    # print(model)
    output = model.forward(Variable(image))
    probabilities = torch.exp(output).data.numpy()[0]
    # print(pr)

    top_idx = np.argsort(probabilities)[-topk:][::-1]
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = probabilities[top_idx]
    Cell = top_class[0]
    acc = top_probability[0]
    print("top class is", top_class[0])
    print("top probability", top_probability[0])

    return Cell, acc, top_class, top_probability

    # return top_probability, top_class


# predict('test.jpg')

# print(predict('chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg', loaded_model))


# Display an image along with the top  classes
def view_classify(img, probabilities, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img_filename = 'JIM_PREDICTOR'
    img = PIL.Image.open(img)

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 10),  ncols=1, nrows=2)
    ct_name = img_filename

    ax1.set_title(ct_name)
    ax1.imshow(img)  # i addeded cmap intially it was zero cmap='gray'
    ax1.axis('on')

    y_pos = np.arange(len(probabilities))
    ax2.barh(y_pos, probabilities, color='red')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(x for x in classes)
    ax2.invert_yaxis()
    plt.show()


def convert_to_array(img):
    im = cv2.imread(img)
    img_ = Image.fromarray(im, 'RGB')
    image = img_.resize((224, 224))
    return np.array(image)


def quit1():
    login_window.destroy()


def delete_auto_key():
    # create database or connect to one
    con = sqlite3.connect("Na.db")
    # create a cursor
    c = con.cursor()
    c.execute('DELETE FROM  auto;',)

    # commit changes
    con.commit()
    # close connections
    con.close()


def back_to_login_1():
    forgot_window.destroy()
    Login()


def confirm_security_quiz():
    global security_window

    con = sqlite3.connect("Na.db")
    # create a cursor
    c = con.cursor()
    c.execute("SELECT *,oid  FROM Gi")
    data = c.fetchall()

    sec_data = [list(ele) for ele in data]

    # commit changes
    con.commit()
    # close connections
    con.close()

    def back_to_login_4():
        security_window.destroy()
        forgot_window.deiconify()

    for a, b, c, d, e in sec_data[:]:
        name = str(a)

        if name == str(your_username.get().strip()):
            forgot_window.withdraw()
            security_window = Tk()
            security_window.title("DETECTION OF PNEUMONIA USING DEEP LEARNING")
            # signup_window.iconbitmap("/e/malaria_code/kip.ICO")
            security_window.geometry("1020x720")
            security_window.resizable(width=False, height=False)
            security_window .configure(bg="#00BFFF")
# "#00BFFF" "indigo"
            frame_14 = Frame(security_window, bg="indigo",
                             height=150, width=90)
            frame_14.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)

            blank = Label(frame_14, text="            ", bg="indigo")
            blank.configure(font=("Times New Roman", 18, "bold"))
            blank.grid(row=0, column=1,  pady=5)
            # blank.grid_forget()
            blank1 = Label(frame_14, text="            ", bg="grey")
            blank1.configure(font=("Times New Roman", 18, "bold"))
            #blank1.grid(row = 1,column = 1,  pady = 5)

            intruction7 = Label(frame_14, text='Answer your Security Question',
                                fg="magenta", bg="black", height=2, width=30)
            intruction7.configure(font=("Times New Roman", 14, "bold"))
            intruction7.grid(row=2, column=1, pady=5)

            your_quiz_label = Label(
                frame_14, text='Your Quiz', fg="magenta", bg="black", height=2, width=13)
            your_quiz_label.configure(font=("Times New Roman", 12, "bold"))
            your_quiz_label.grid(row=3, padx=2.5, pady=5, sticky="w")

            your_quiz_E = Entry(frame_14, width=35, font=(
                "Times New Roman", 12, "bold"), bg="white", fg="black")
            your_quiz_E.insert("end", str(c))
            your_quiz_E.grid(row=3, column=1, padx=8,
                             pady=5, ipady=8, sticky="w")
            your_quiz_E.config(state="disable")

            provide_answer = Label(
                frame_14, text='Your Answer', fg="magenta", bg="black", height=2, width=13)
            provide_answer.configure(font=("Times New Roman", 12, "bold"))
            provide_answer.grid(row=4, padx=2.5, pady=5, sticky="w")

            your_answerE = Entry(frame_14, width=35, font=(
                "Times New Roman", 12, "bold"), bg="white", fg="black")
            your_answerE.grid(row=4, column=1, padx=8,
                              pady=5, ipady=8, sticky="w")

            def confirm_answer():
                if your_answerE.get().capitalize().strip() == str(d):
                    security_window.destroy()
                    show_window = Tk()
                    show_window.title(
                        "DETECTION OF PNEUMONIA USING DEEP LEARNING")
                    show_window.geometry("1020x720")
                    show_window.resizable(width=False, height=False)
                    show_window.configure(bg="brown")
                    frame_12 = Frame(show_window, bg="grey",
                                     height=150, width=120)
                    frame_12.place(relheight=0.8, relwidth=0.8,
                                   relx=0.1, rely=0.1)

                    blank = Label(frame_12, text="            ", bg="grey")
                    blank.configure(font=("Times New Roman", 18, "bold"))
                    blank.grid(row=0, column=1,  pady=5)
                    # blank.grid_forget()
                    blank1 = Label(frame_12, text="            ", bg="grey")
                    blank1.configure(font=("Times New Roman", 18, "bold"))
                    #blank1.grid(row = 1,column = 1,  pady = 5)

                    intruction6 = Label(
                        frame_12, text='Your Username and Password are shown below', fg="magenta", bg="black", height=2, width=37)
                    intruction6.configure(font=("Times New Roman", 14, "bold"))
                    intruction6.grid(row=2, column=1,  pady=15)

                    username_label_1 = Label(
                        frame_12, text='Username', fg="white", bg="blue", height=2, width=15)
                    username_label_1.configure(
                        font=("Times New Roman", 12, "bold"))
                    username_label_1.grid(row=3, padx=2.5, pady=2, sticky="w")

                    password_label_1 = Label(
                        frame_12, text='Password', fg="white", bg="blue", height=2, width=15)
                    password_label_1.configure(
                        font=("Times New Roman", 12, "bold"))
                    password_label_1.grid(row=4, padx=2.5, pady=2, sticky="w")

                    username_label_1E = Entry(frame_12, width=40, font=(
                        "Times New Roman", 12, "bold"), bg="white", fg="black")
                    username_label_1E.insert(
                        0, str(your_username.get().strip()))
                    username_label_1E.grid(
                        row=3, column=1, padx=8, pady=2, ipady=8, sticky="w")
                    username_label_1E.config(state="disable")

                    password_label_1E = Entry(
                        frame_12, width=40, font=("Times New Roman", 12, "bold"))
                    password_label_1E.insert(0, b)
                    password_label_1E.grid(
                        row=4, column=1, padx=8, pady=2, ipady=8, sticky="w")
                    password_label_1E.config(state="disable")

                    def back_to_login_3():
                        show_window.destroy()
                        Login()

                    def change_password():
                        global ID_NO
                        global password2E
                        show_window.withdraw()

                        change_password_window = Tk()
                        change_password_window.title(
                            "DETECTION OF PNEUMONIA USING DEEP LEARNING")
                        change_password_window.geometry("1020x720")
                        change_password_window.resizable(
                            width=False, height=False)
                        change_password_window.configure(bg="brown")
                        frame_14 = Frame(change_password_window,
                                         bg="grey", height=150, width=120)
                        frame_14.place(
                            relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)

                        blank = Label(frame_14, text="            ", bg="grey")
                        blank.configure(font=("Times New Roman", 18, "bold"))
                        blank.grid(row=0, column=1,  pady=5)

                        intruction9 = Label(
                            frame_14, text='Please SET your NEW Password', fg="magenta", bg="black", height=2, width=32)
                        intruction9.configure(
                            font=("Times New Roman", 14, "bold"))
                        intruction9.grid(row=1, column=1,  pady=15)

                        password1 = Label(
                            frame_14, text='Old Password', fg="white", bg="blue", height=2, width=12)
                        password1.configure(
                            font=("Times New Roman", 12, "bold"))
                        password1.grid(row=2, padx=2.5, pady=5,
                                       ipadx=5, sticky="w")
                        # password1.grid_forget()

                        password1E = Entry(frame_14, width=40, font=(
                            "Times New Roman", 12, "bold"), bg="white", fg="black")
                        password1E.insert(
                            "end", password_label_1E.get().strip())
                        password1E.grid(row=2, column=1, padx=8,
                                        pady=5, ipady=8, sticky="w")
                        password1E.config(state="disabled")
                        # password1E.grid_forget()

                        password2 = Label(
                            frame_14, text='New Password', fg="white", bg="blue", height=2, width=12)
                        password2.configure(
                            font=("Times New Roman", 12, "bold"))
                        password2.grid(row=3, padx=2.5, pady=5,
                                       ipadx=5, sticky="w")

                        password2E = Entry(frame_14, width=40, font=(
                            "Times New Roman", 12, "bold"), bg="white", fg="black")
                        password2E.grid(row=3, column=1, padx=8,
                                        pady=5, ipady=8, sticky="w")

                        ID_NO = Entry(frame_14, width=40, font=(
                            "Times New Roman", 12, "bold"), bg="white", fg="black")
                        ID_NO.insert("end", str(e))
                        ID_NO.grid(row=4, column=1, padx=8,
                                   pady=5, ipady=8, sticky="w")
                        ID_NO.grid_forget()

                        def back_to_login_9():
                            change_password_window.destroy()
                            Login()

                        def save_password():
                            # create database or connect to one
                            con = sqlite3.connect("Na.db")
                            # create a cursor
                            c = con.cursor()

                            password_no = ID_NO.get()

                            c.execute("UPDATE Gi SET PASSWORD = (?) WHERE oid = (?)", [
                                      password2E.get().strip(), ID_NO.get().strip()])

                            # commit changes
                            con.commit()
                            # close connections
                            con.close()
                            change_password_window.destroy()
                            Login()

                        save_button = Button(frame_14, text="SAVE", fg="white", bg="blue", font=(
                            "Times New Roman", 12, "bold"), command=save_password)
                        save_button.grid(row=5, column=1, padx=45,
                                         pady=20, ipady=5, ipadx=30, sticky="e")

                        back_button_10 = Button(frame_14, text="BACK", fg="black", bg="orange", font=(
                            "Times New Roman", 12, "bold"), command=back_to_login_9)
                        back_button_10.grid(
                            row=5, column=1, padx=45, pady=20, ipady=5, ipadx=30, sticky="w")

                        change_password_window.mainloop()

                    back_login_Button = Button(frame_12, text='Back to Login', bg="blue", font=(
                        "Times New Roman", 12, "bold"), fg="white", command=back_to_login_3)
                    back_login_Button.grid(
                        row=5, column=1, padx=43, pady=20, ipadx=13, ipady=10, sticky="w")

                    change_Button = Button(frame_12, text='Change Password', bg="brown", font=(
                        "Times New Roman", 12, "bold"), fg="black", command=change_password)
                    change_Button.grid(
                        row=5, column=1, padx=43, pady=20, ipadx=13, ipady=10, sticky="e")

                    show_window.mainloop()
                else:
                    mb.showwarning(
                        "DETECTION OF PNEUMONIA USING DEEP LEARNING", "INCORRECT ANSWER!!")

            security_button = Button(frame_14, text="NEXT >>", fg="white", bg="blue", font=(
                "Times New Roman", 12, "bold"), command=confirm_answer)
            security_button.grid(row=5, column=1, padx=25,
                                 pady=10, ipady=5, ipadx=35, sticky="e")

            security_back_button = Button(frame_14, text="BACK", fg="black", bg="orange", font=(
                "Times New Roman", 12, "bold"), command=back_to_login_4)
            security_back_button.grid(
                row=5, column=1, padx=20, pady=10, ipady=5, ipadx=35, sticky="w")

            security_window.mainloop()


def confirm_user_name():

    con = sqlite3.connect("Na.db")
    # create a cursor
    c = con.cursor()
    c.execute("SELECT USER_NAME,oid FROM Gi")

    def Convertion(tup, dicti):

        dicti = dict(tup)
        return dicti
    tups = c.fetchall()
    dictionary = {}
    confirm_user = str(your_username.get().strip())
    final_user_dict = Convertion(tups, dictionary)

    # commit changes
    con.commit()
    # close connections
    con.close()

    if confirm_user in final_user_dict:

        confirm_security_quiz()
    else:
        mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                       "USERNAME DOES NOT EXIST")


def forgot_password():

    login_window.destroy()
    global forgot_window
    global final_username
    global your_username

    forgot_window = Tk()

    forgot_window.title("DETECTION OF PNEUMONIA USING DEEP LEARNING")
    forgot_window.geometry("1020x720")
    forgot_window.resizable(width=False, height=False)
    forgot_window.configure(bg="brown")
    frame_11 = Frame(forgot_window, bg="yellow", height=150, width=120)
    frame_11.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)

    blank = Label(frame_11, text="            ", bg="grey")
    blank.configure(font=("Times New Roman", 18, "bold"))
    blank.grid(row=0, column=1,  pady=5)
    # blank.grid_forget()
    blank1 = Label(frame_11, text="            ", bg="grey")
    blank1.configure(font=("Times New Roman", 18, "bold"))
    #blank1.grid(row = 4,column = 1,  pady = 5)

    intruction = Label(frame_11, text='Please Enter Username to Recover your Password',
                       fg="magenta", bg="black", height=2, width=38)
    intruction.configure(font=("Times New Roman", 14, "bold"))
    intruction.grid(row=2, column=1,  pady=10)
    your_User_label = Label(frame_11, text='Your Username',
                            fg="white", bg="blue", height=2, width=12)
    your_User_label.configure(font=("Times New Roman", 12, "bold"))
    your_User_label.grid(row=3, padx=2.5, pady=2, ipadx=5, sticky="w")

    your_username = Entry(frame_11, width=45, font=(
        "Times New Roman", 12, "bold"), bg="white", fg="black")
    your_username.grid(row=3, column=1, padx=8, pady=2, ipady=8, sticky="w")

    final_username = str(your_username.get().strip())

    forgot_enter_button = Button(frame_11, text="NEXT >>", fg="white", bg="blue", font=(
        "Times New Roman", 12, "bold"), command=confirm_user_name)
    forgot_enter_button.grid(row=5, column=1, padx=60,
                             pady=10, ipady=5, ipadx=40, sticky="e")

    forgot_back_button = Button(frame_11, text="BACK", fg="black", bg="orange", font=(
        "Times New Roman", 12, "bold"), command=back_to_login_1)
    forgot_back_button.grid(row=5, column=1, padx=35,
                            pady=10, ipady=5, ipadx=40, sticky="w")
    forgot_window.mainloop()


def Login():
    global enter_login_name
    global enter_login_password
    global login_window

    login_window = Tk()
    login_window.title("DETECTION OF PNEUMONIA USING DEEP LEARNING")
    login_window.geometry("1020x720")
    login_window.resizable(width=True, height=True)

    # login_window.columnconfigure(0, weight=1) # "#00BFFF" "indigo"
    #login_window.rowconfigure(0, weight=1)
    login_window.configure(bg="#00BFFF")
    frame_1 = Frame(login_window, bg="indigo", height=150, width=90)
    frame_1.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)

    # blank = Label(frame_1, text="", bg="grey")
    # blank.configure(font=("Times New Roman", 18, "bold"))
    # blank.grid(row=0, column=1,  pady=5)
    # blank.grid_forget()
    blank1 = Label(frame_1, text="            ", bg="grey")
    blank1.configure(font=("Times New Roman", 18, "bold"))
    blank1.grid(row=1, column=1,  pady=5)
    blank1.grid_forget()

    intruction = Label(frame_1, text='Login To The System A9',
                       fg="magenta", bg="black", height=2, width=40)
    intruction.configure(font=("Times New Roman", 14, "bold"))
    intruction.grid(row=3, column=1,  pady=5)

    username_label = Label(frame_1, text='Enter Username',
                           fg="white", bg="blue", height=2, width=15)
    username_label.configure(font=("Times New Roman", 12, "bold"))

    password_label = Label(frame_1, text='Enter Password',
                           fg="white", bg="blue", height=2, width=15)
    password_label.configure(font=("Times New Roman", 12, "bold"))

    username_label.grid(row=4, padx=2.5, pady=2, ipady=3, sticky="w")
    password_label.grid(row=5, padx=2.5, pady=2, ipady=3, sticky="w")

    enter_login_name = Entry(frame_1, width=35, font=(
        "Times New Roman", 12, "bold"), bg="white", fg="black")
    enter_login_password = Entry(frame_1, width=35, font=(
        "Times New Roman", 12, "bold"), show='*')

    enter_login_name.grid(row=4, column=1, padx=8, pady=2, ipady=8, sticky="w")
    enter_login_password.grid(row=5, column=1, padx=8,
                              pady=2, ipady=8, sticky="w")

    loginButton = Button(frame_1, text='Login', bg="#00BFFF", font=(
        "Times New Roman", 12, "bold"), fg="white", command=CheckLogin)
    loginButton.grid(row=6, column=1, padx=5, pady=5,
                     ipadx=28, ipady=4, sticky="w")

    forgotButton = Button(frame_1, text='Terminate', bg="red", font=(
        "Times New Roman", 12, "bold"), fg="black", command=quit1)
    forgotButton .grid(row=6, column=1, padx=5, pady=5,
                       ipadx=15, ipady=4, sticky="e")

    # signupButton = Button(frame_1, text='Create Account', fg="white", font=(
    #     "Times New Roman", 12, "bold"), bg="#556B2F", command=Signup)
    # signupButton.grid(row=7, column=1, columnspan=3,
    #                   padx=5, pady=5, ipadx=60, ipady=5)

    # # CREATE QUIT BUTTON
    # Q_button = Button(frame_1, text="Terminate", bg="#8B0000", fg="white", font=(
    #     "Times New Roman", 12, "bold"), command=quit1)
    # Q_button.grid(row=8, column=1, columnspan=2,
    #               padx=5, pady=5, ipadx=50, ipady=4)

    auto_button = Button(frame_1, text="Delete all KEYs", fg="black", bg="brown", font=(
        "Times New Roman", 12, "bold"), command=delete_auto_key)
    auto_button .grid(row=9, column=1, columnspan=3, padx=8, pady=5, ipadx=26)
    auto_button.grid_forget()

    login_window.mainloop()


def restart_program():
    os.execl(sys.executable, sys.executable, *sys.argv)


def Activation_key():
    global Admin
    global frame_3
    global check_window
    global Admin
    if len(str(combo.get())) > 0 and len(answer_E.get().strip()) >= 3:
        my_answer = answer_E.get().capitalize().strip()

        main.withdraw()
        check_window = Tk()
        check_window.title("DETECTION OF PNEUMONIA USING DEEP LEARNING")
        # signup_window.iconbitmap("/e/malaria_code/kip.ICO")
        check_window.geometry("1020x720")
        check_window.resizable(width=False, height=False)
        frame_3 = Frame(check_window, bg="grey", height=150, width=200)
        frame_3.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)
        check_window.configure(bg="brown")

        blank = Label(frame_3, text="            ", bg="grey")
        blank.configure(font=("Times New Roman", 18, "bold"))
        blank.grid(row=0, column=1,  pady=5)
        # blank.grid_forget()
        blank1 = Label(frame_3, text="            ", bg="grey")
        blank1.configure(font=("Times New Roman", 18, "bold"))
        #blank1.grid(row = 1,column = 1,  pady = 5)

        intruction3 = Label(frame_3, text='Enter the Activation kEY to Signup ',
                            fg="magenta", bg="black", height=2, width=35)
        intruction3.configure(font=("Times New Roman", 12, "bold"))
        intruction3.grid(row=2, column=1, padx=3, pady=5)

        key_label = Label(frame_3, text='Activation Key',
                          fg="white", bg="blue", height=2, width=13)
        key_label.configure(font=("Times New Roman", 12, "bold"))
        key_label.grid(row=3, padx=2.5, pady=5, sticky="w")

        active_key = Entry(frame_3, width=40, font=(
            "Times New Roman", 12, "bold"), show='*')
        active_key.grid(row=3, column=1, padx=4, pady=5, ipady=8, sticky="w")

        ansa = Entry(frame_3, width=40, font=("Times New Roman", 12, "bold"))
        ansa .insert("end", my_answer)
        ansa.grid(row=4, column=1, padx=4, pady=5, ipady=8, sticky="w")
        ansa.grid_forget()

        get_value = Entry(frame_3, width=35, font=(
            "Times New Roman", 12, "bold"))
        get_value .insert("end", str(combo.get()))
        get_value.grid(row=5, column=1, padx=8, pady=2, ipady=8, sticky="w")
        get_value.grid_forget()

        def check_key():

            con = sqlite3.connect("Na.db")
            # create a cursor
            c = con.cursor()
            c.execute("SELECT *,oid  FROM auto")
            data = c.fetchall()

            auto_data = [list(ele) for ele in data]

            # commit changes
            con.commit()
            # close connections
            con.close()

            if str(active_key.get().strip()) == auto_data[-1][0]:

                con = sqlite3.connect("Na.db")
                # create a cursor
                c = con.cursor()
                c.execute("SELECT *,oid  FROM Gi")
                my_data = c.fetchall()

                user_data = [list(ele) for ele in my_data]

                # commit changes
                con.commit()
                # close connections
                con.close()

                if len(user_data) == 0:
                    signup_username = str(enter_signup_name.get().strip())
                    Admin = signup_username
                    con = sqlite3.connect("Na.db")
                    # create a cursor
                    c = con.cursor()

                    c.execute("INSERT INTO Gi VALUES (?,?,?,?)", [enter_signup_name.get().strip(),
                                                                  enter_signup_password.get().strip(),
                                                                  get_value.get(),
                                                                  ansa.get().strip()
                                                                  ])

                    # commit changes
                    con.commit()
                    # close connections
                    con.close()
                    restart_program()

                elif len(user_data) > 0 and len(user_data) < 4:
                    con = sqlite3.connect("Na.db")
                    # create a cursor
                    c = con.cursor()
                    c.execute("SELECT USER_NAME, PASSWORD FROM Gi")

                    def Convertion(tup, dicti):

                        dicti = dict(tup)
                        return dicti
                    tups = c.fetchall()
                    dictionary = {}
                    signup_username_2 = str(enter_signup_name.get().strip())
                    signup_password_2 = str(
                        enter_signup_password.get().strip())

                    final_dictionary = Convertion(tups, dictionary)
                    if signup_username_2 in final_dictionary:
                        if final_dictionary[signup_username_2] != signup_password_2:

                            signup_username = str(
                                enter_signup_name.get().strip())

                            Admin = signup_username
                            con = sqlite3.connect("Na.db")
                            # create a cursor
                            c = con.cursor()
                            c.execute("INSERT INTO Gi VALUES (?,?,?,?)", [enter_signup_name.get().strip(),
                                                                          enter_signup_password.get().strip(),
                                                                          get_value.get().strip(),
                                                                          ansa.get().strip()
                                                                          ])

                            # commit changes
                            con.commit()
                            # close connections
                            con.close()
                            restart_program()
                        elif final_dictionary[signup_username_2] == signup_password_2:
                            mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                                           "YOU ALREADY HAVE AN ACCOUNT!!!, LOG IN INSTEAD")

                    else:
                        signup_username = str(enter_signup_name.get().upper())
                        Admin = signup_username
                        con = sqlite3.connect("Na.db")
                        # create a cursor
                        c = con.cursor()
                        c.execute("INSERT INTO Gi VALUES (?,?,?,?)", [enter_signup_name.get().strip(),
                                                                      enter_signup_password.get().strip(),
                                                                      get_value.get(),
                                                                      ansa.get().strip()
                                                                      ])

                        # commit changes
                        con.commit()
                        # close connections
                        con.close()
                        restart_program()

                elif len(user_data) >= 4:
                    mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                                   "THERE IS ALREADY MAXIMUM NUMBER OF USERS!")

            else:
                mb.showwarning(
                    "DETECTION OF PNEUMONIA USING DEEP LEARNING", "INVALID ACTIVATION KEY!")
                # signup_window.withdraw()
                # active_key.delete(0,END)

        def back_to_login():
            main.deiconify()
            main.destroy()
            signup_window.destroy()
            check_window.destroy()
            Login()

        enter_button = Button(frame_3, text="ENTER", fg="white", bg="blue", font=(
            "Times New Roman", 12, "bold"), command=check_key)
        enter_button.grid(row=6, column=1, padx=20, pady=15,
                          ipady=5, ipadx=30, sticky="e")

        back_button = Button(frame_3, text="BACK", fg="black", bg="orange", font=(
            "Times New Roman", 12, "bold"), command=back_to_login)
        back_button.grid(row=6, column=1, padx=25, pady=15,
                         ipady=5, ipadx=30, sticky="w")

    elif len(str(combo.get())) > 0 and len(answer_E.get().strip()) == 0:
        mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                       "YOU HAVE NOT ANSWERED THE SELECTED QUESTION, MUST BE ATLEAST THREE CHARACTERS!!")
    elif len(str(combo.get())) == 0 and len(answer_E.get().strip()) > 0:
        mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                       "YOU HAVE NOT SELECTED ANY QUESTION!!")
    elif len(str(combo.get())) > 0 and len(answer_E.get().strip()) < 3:
        mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                       "YOUR ANSWER IS TOO SHORT!!")

    elif len(str(combo.get())) == 0 and len(answer_E.get().strip()) == 0:
        mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                       "NO QUESTION WAS SELECTED AND NO ANSWER WAS PROVIDED!")


def back_to_login_2():
    main.destroy()
    signup_window.deiconify()
    signup_window.destroy()
    Login()

    # signup_window.destroy()


def security_quiz():
    global main
    global enter_signup_name_1
    global enter_signup_password_1
    global combo
    global my_answer
    global answer_E
# "#00BFFF" "indigo"
    if len(str(enter_signup_name.get().strip())) >= 4 and len(str(enter_signup_name.get().strip())) < 10 and len(str(enter_signup_password.get().strip())) >= 4:
        signup_window.withdraw()

        main = Tk()
        main.title("DETECTION OF Pneumonia USING DEEP LEARNING")
        # signup_window.iconbitmap("/e/malaria_code/kip.ICO")
        main.geometry("1020x720")
        main.resizable(width=False, height=False)
        main.configure(bg="#00BFFF")

        frame_13 = Frame(main, bg="indigo", height=150, width=90)
        frame_13.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)

        blank = Label(frame_13, text="            ", bg="indigo")
        blank.configure(font=("Times New Roman", 18, "bold"))
        blank.grid(row=0, column=1,  pady=5)
        # blank.grid_forget()
        blank1 = Label(frame_13, text="            ", bg="grey")
        blank1.configure(font=("Times New Roman", 18, "bold"))
        #blank1.grid(row = 1,column = 1,  pady = 5)

        intruction5 = Label(frame_13, text='ENTER TOKEN NUMBER',
                            fg="magenta", bg="black", height=2, width=30)
        intruction5.configure(font=("Times New Roman", 14, "bold"))
        intruction5.grid(row=2, column=1, pady=7)

        secu_quiz_label = Label(
            frame_13, text='Select Question', fg="magenta", bg="black", height=2, width=15)
        secu_quiz_label.configure(font=("Times New Roman", 12, "bold"))
        secu_quiz_label.grid(row=3, padx=2.5, pady=8, sticky="w")

        # ATTENTION: this applies the new style 'combostyle' to all ttk.Combobox

        answer_label = Label(frame_13, text='TOKEN',
                             fg="magenta", bg="black", height=2, width=15)
        answer_label.configure(font=("Times New Roman", 12, "bold"))
        answer_label.grid(row=4, padx=2.5, pady=6, sticky="w")

        username_label_1 = Label(
            frame_13, text='Username', fg="magenta", bg="black", height=2, width=16)
        username_label_1.configure(font=("Times New Roman", 12, "bold"))
        username_label_1.grid(row=4, padx=2.5, pady=2, sticky="w")
        username_label_1.grid_forget()

        password_label_1 = Label(
            frame_13, text='Password', fg="magenta", bg="black", height=2, width=16)
        password_label_1.configure(font=("Times New Roman", 12, "bold"))
        password_label_1.grid(row=5, padx=2.5, pady=2, sticky="w")
        password_label_1.grid_forget()

        enter_signup_name_1 = Entry(
            frame_13, width=35, font=("Times New Roman", 12, "bold"))
        enter_signup_name_1.insert(
            "end", str(enter_signup_name.get().strip()))
        enter_signup_name_1.grid(
            row=4, column=1, padx=8, pady=2, ipady=8, sticky="w")
        enter_signup_name_1.grid_forget()

        enter_signup_password_1 = Entry(
            frame_13, width=35, font=("Times New Roman", 12, "bold"))
        enter_signup_password_1.insert(
            "end", str(enter_signup_password.get().strip()))
        enter_signup_password_1.grid(
            row=5, column=1, padx=8, pady=2, ipady=8, sticky="w")
        enter_signup_password_1.grid_forget()

        answer_E = Entry(frame_13, width=35, font=(
            "Times New Roman", 12, "bold"))
        answer_E.grid(row=4, column=1, padx=8, pady=2, ipady=8, sticky="w")

        forgot_enter_button_1 = Button(frame_13, text="NEXT>>", fg="white", bg="blue", font=(
            "Times New Roman", 12, "bold"), command=Activation_key)
        forgot_enter_button_1.grid(
            row=6, column=1, padx=20, pady=20, ipady=5, ipadx=30, sticky="e")

        forgot_back_button_1 = Button(frame_13, text="BACK", fg="black", bg="orange", font=(
            "Times New Roman", 12, "bold"), command=back_to_login_2)
        forgot_back_button_1.grid(
            row=6, column=1, padx=20, pady=20, ipady=5, ipadx=30, sticky="w")

        main.mainloop()

    else:
        mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                       "INVALID DETAILS, USERNAME AND PASSWORD MUST BE ATLEAST FOUR CHARACTERS EACH BUT LESS THAN TEN")


def CheckLogin():
    global Admin
    global login_username
    global login_password

    con = sqlite3.connect("Na.db")
    # create a cursor
    c = con.cursor()
    c.execute("SELECT USER_NAME, PASSWORD FROM Gi")

    def Convertion(tup, dicti):

        dicti = dict(tup)
        return dicti
    tups = c.fetchall()
    dictionary = {}
    login_username = str(enter_login_name.get().strip())
    login_password = str(enter_login_password.get().strip())

    final_dictionary = Convertion(tups, dictionary)

    if login_username in final_dictionary:
        if final_dictionary[login_username] == login_password:
            Admin = login_username.upper()
            login_window.destroy()
            root1()
        else:
            mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                           f" { login_username} Invalid Login!,password is incorrect!!")
    else:
        mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                       "Invalid Login!, Either username or password is incorrect")

    # commit changes
    con.commit()
    # close connections
    con.close()


def back_to_login():
    signup_window.destroy()
    Login()


def Signup():
    login_window.destroy()
    global enter_signup_password
    global enter_signup_name
    global signup_window

    signup_window = Tk()
    signup_window.title("DETECTION OF PNEUMONIA USING DEEP LEARNING")
    # signup_window.iconbitmap("/e/malaria_code/kip.ICO")
    signup_window.geometry("1020x720")
    signup_window.resizable(width=False, height=False)
    frame_2 = Frame(signup_window, bg="indigo", height=150, width=90)
    frame_2.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)
    signup_window.configure(bg="#00BFFF")
# "#00BFFF" "indigo"

    # blank.grid_forget()
    blank1 = Label(frame_2, text="            ", bg="grey")
    blank1.configure(font=("Times New Roman", 18, "bold"))
    #blank1.grid(row = 1,column = 1,  pady = 5)

    intruction = Label(frame_2, text='Enter your details',
                       fg="magenta", bg="black", height=2, width=26)
    intruction.configure(font=("Times New Roman", 14, "bold"))
    intruction.grid(row=3, column=1, pady=5)

    username_label = Label(frame_2, text='Create Username',
                           fg="white", bg="blue", height=2, width=15)
    username_label.configure(font=("Times New Roman", 12, "bold"))
    username_label.grid(row=4, padx=2.5, pady=2, ipady=3, sticky="w")

    password_label = Label(frame_2, text='Create Password',
                           fg="white", bg="blue", height=2, width=15)
    password_label.configure(font=("Times New Roman", 12, "bold"))
    password_label.grid(row=5, padx=2.5, pady=2, ipady=3, sticky="w")

    enter_signup_name = Entry(
        frame_2, width=35, font=("Times New Roman", 12, "bold"))
    enter_signup_name.grid(row=4, column=1, padx=8,
                           pady=2, ipady=8, sticky="w")

    enter_signup_password = Entry(frame_2, width=35, font=(
        "Times New Roman", 12, "bold"), show='*')
    enter_signup_password.grid(
        row=5, column=1, padx=8, pady=2, ipady=8, sticky="w")
# "#00BFFF" "indigo"
    finishButton = Button(frame_2, text='SignUp', fg="Red", font=(
        "Times New Roman", 12, "bold"), bg="yellow", command=security_quiz)
    finishButton.grid(row=6, column=1, columnspan=3,
                      padx=5, pady=7, ipadx=90, ipady=4)

    logButton = Button(frame_2, text='Go Back', bg="#00BFFF", font=(
        "Times New Roman", 12, "bold"), fg="black", command=back_to_login)
    logButton.grid(row=7, column=1, columnspan=3,
                   padx=5, pady=7, ipadx=60, ipady=4)

    signup_window.mainloop()


def root1():
        # create global variables
    global f_name
    global o_name
    global age
    global gender
    global accuracy
    global status
    global records
    global root
    global frame_8

    root = Tk()
    root.title("DETECTION OF Pneumonia USING DEEP LEARNING")
    root.geometry("1020x720")
    root.resizable(width=True, height=True)
    root.configure(bg="#00BFFF")
# "#00BFFF" "indigo"
    # create a frame
    frame_8 = Frame(root, bg="indigo", height=400, width=400)
    frame_8.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)

    # DATABASES
    # create database or connect to one
    con = sqlite3.connect("Na.db")
    # create a cursor
    c = con.cursor()
    # create table()
    '''c.execute("""CREATE TABLE d (
           FIRST_NAME text,
           OTHER_NAME text,
           AGE integer,
           GENDER text,
           STATUS text,
           ACCURACY real,
           SERVED_BY text 
           )""")'''
    def return_to_root():
        change_activation_window.destroy()
        root1()

    def change_activation_key():
        global change_activation_window

        root.destroy()
        change_activation_window = Tk()
        change_activation_window.title(
            "DETECTION OF PNEUMONIA USING DEEP LEARNING")
        # signup_window.iconbitmap("/e/malaria_code/kip.ICO")
        change_activation_window.geometry("1020x720")
        change_activation_window.resizable(width=False, height=False)
        change_activation_window.configure(bg="brown")

        frame_10 = Frame(change_activation_window,
                         bg="grey", height=150, width=90)
        frame_10.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)

        blank = Label(frame_10, text="            ", bg="grey")
        blank.configure(font=("Times New Roman", 18, "bold"))
        blank.grid(row=0, column=1,  pady=5)

        intruction4 = Label(frame_10, text='Please Enter Old Key and New Key',
                            fg="magenta", bg="black", height=2, width=30)
        intruction4.configure(font=("Times New Roman", 14, "bold"))
        intruction4.grid(row=1, column=1, pady=5)

        old_act_key = Label(frame_10, text='Old Key',
                            fg="white", bg="blue", height=2, width=15)
        old_act_key.configure(font=("Times New Roman", 12, "bold"))

        new_act_key = Label(frame_10, text='New Key',
                            fg="white", bg="blue", height=2, width=15)
        new_act_key.configure(font=("Times New Roman", 12, "bold"))

        old_act_key.grid(row=2, padx=2.5, pady=4, sticky="w")
        new_act_key.grid(row=3, padx=2.5, pady=4, sticky="w")

        old_act_keyE = Entry(frame_10, width=35, font=(
            "Times New Roman", 12, "bold"), show="*")
        new_act_keyE = Entry(frame_10, width=35, font=(
            "Times New Roman", 12, "bold"), show='*')
        old_act_keyE.grid(row=2, column=1, padx=8, pady=4, ipady=8, sticky="w")
        new_act_keyE.grid(row=3, column=1, padx=8, pady=4, ipady=8, sticky="w")
        def change_activation_key1():

            con = sqlite3.connect("Na.db")
            # create a cursor
            c = con.cursor()
            c.execute("SELECT *,oid  FROM auto")
            data = c.fetchall()

            auto_data = [list(ele) for ele in data]

            # commit changes
            con.commit()
            # close connections
            con.close()

            if str(old_act_keyE.get().strip()) == auto_data[-1][0]:
                if len(str(new_act_keyE.get().strip())) >= 4:
                    con = sqlite3.connect("Na.db")
                    # create a cursor
                    c = con.cursor()
                    # INSERT INTO TABLE
                    c.execute("UPDATE auto SET ACTIVATION_KEY = (?) WHERE oid = 1", [
                              str(new_act_keyE.get().strip())])

                    # commit changes
                    con.commit()
                    # close connections
                    con.close()
                    change_activation_window.destroy()
                    root1()
                else:
                    mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                                   "YOUR NEW ACTIVATION KEY MUST BE ATLEAST FOUR CHARACTERS!!")
            else:
                mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                               "YOUR OLD ACTIVATION KEY IS INCORRECT!!")

        enter_button5 = Button(frame_10, text="CHANGE", fg="white", bg="blue", font=(
            "Times New Roman", 12, "bold"), command=change_activation_key1)
        enter_button5.grid(row=4, column=1, padx=18, pady=10,
                           ipady=5, ipadx=35, sticky="e")

        back_button6 = Button(frame_10, text="BACK", fg="black", bg="orange", font=(
            "Times New Roman", 12, "bold"), command=return_to_root)
        back_button6.grid(row=4, column=1, padx=18, pady=10,
                          ipady=5, ipadx=35, sticky="w")

        change_activation_window.mainloop()

    def return_to_main():
        change_auto.destroy()
        root1()

    def change_auto_key():
        root.destroy()
        global change_auto

        change_auto = Tk()
        change_auto.title("DETECTION OF PNEUMONIA USING DEEP LEARNING")
        # signup_window.iconbitmap("/e/malaria_code/kip.ICO")
        change_auto.geometry("1020x720")
        change_auto.resizable(width=False, height=False)
        change_auto.configure(bg="brown")

        frame_9 = Frame(change_auto, bg="grey", height=150, width=90)
        frame_9.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)

        blank = Label(frame_9, text="            ", bg="grey")
        blank.configure(font=("Times New Roman", 18, "bold"))
        blank.grid(row=0, column=1,  pady=5)

        intruction3 = Label(frame_9, text='Change Authorization Key ',
                            fg="magenta", bg="black", height=2, width=26)
        intruction3.configure(font=("Times New Roman", 14, "bold"))
        intruction3.grid(row=1, column=1, pady=5)

        old_auto_key = Label(frame_9, text='Old Key',
                             fg="white", bg="blue", height=2, width=15)
        old_auto_key.configure(font=("Times New Roman", 12, "bold"))

        new_auto_key = Label(frame_9, text='New Key',
                             fg="white", bg="blue", height=2, width=15)
        new_auto_key.configure(font=("Times New Roman", 12, "bold"))

        old_auto_key.grid(row=2, padx=2.5, pady=4, sticky="w")
        new_auto_key.grid(row=3, padx=2.5, pady=4, sticky="w")

        old_auto_keyE = Entry(frame_9, width=35, font=(
            "Times New Roman", 12, "bold"), show="*")
        new_auto_keyE = Entry(frame_9, width=35, font=(
            "Times New Roman", 12, "bold"), show='*')
        old_auto_keyE.grid(row=2, column=1, padx=8,
                           pady=4, ipady=8, sticky="w")
        new_auto_keyE.grid(row=3, column=1, padx=8,
                           pady=4, ipady=8, sticky="w")

        con = sqlite3.connect("Na.db")
        # create a cursor
        c = con.cursor()
        c.execute("SELECT *,oid FROM act")

        def Convert(tup, dicti):

            dicti = dict(tup)
            return dicti
        tups = c.fetchall()
        dictionary = {}

        auto_data_1 = Convert(tups, dictionary)

        def confirm_old_key():
            if str(old_auto_keyE.get().strip()) in auto_data_1:
                if len(str(new_auto_keyE.get().strip())) >= 4:
                    # create database or connect to one
                    con = sqlite3.connect("Na.db")
                    # create a cursor
                    c = con.cursor()
                    # INSERT INTO TABLE
                    c.execute("UPDATE act SET AUTHORIZATION_KEY = (?) WHERE oid = 1", [
                              str(new_auto_keyE.get().strip())])
                    # commit changes
                    con.commit()
                    # close connections
                    con.close()
                    change_auto .destroy()
                    root1()
                else:
                    mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                                   "YOUR NEW AUTHORIZATION KEY MUST BE ATLEAST FOUR CHARACTERS!!")
            else:
                mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                               "THE OLD KEY YOU PROVIDED IS INCORRECT!!")

        enter_button3 = Button(frame_9, text="CHANGE", fg="white", bg="blue", font=(
            "Times New Roman", 12, "bold"), command=confirm_old_key)
        enter_button3.grid(row=4, column=1, padx=14, pady=15,
                           ipady=5, ipadx=25, sticky="e")

        back_button4 = Button(frame_9, text="BACK", fg="black", bg="orange", font=(
            "Times New Roman", 12, "bold"), command=return_to_main)
        back_button4.grid(row=4, column=1, padx=14, pady=15,
                          ipady=5, ipadx=30, sticky="w")
        change_auto.mainloop()

    direction_label = Label(frame_8, text="Key Patient details",
                            fg="magenta", bg="black", height=3, width=40)
    direction_label.configure(font=("Times New Roman", 14, "bold"))
    direction_label.grid(row=0, column=1,  pady=10)

    def caps(f_uppercase):
        f.set(f.get().upper())
    f = StringVar()

    # create text boxes labels
    f_name_label = Label(frame_8, text="First Name",
                         fg="white", bg="black", width=10, height=2)
    f_name_label.configure(font=("Times New Roman", 12, "bold"))
    f_name_label.grid(row=1, column=0, padx=5, pady=2, sticky="E")

    o_name_label = Label(frame_8, text="Surname",
                         fg="white", bg="black", width=10, height=2)
    o_name_label.configure(font=("Times New Roman", 12, "bold"))
    o_name_label.grid(row=2, column=0, padx=5, pady=2, sticky="E")

    age_label = Label(frame_8, text="Age", fg="white",
                      bg="black", width=10, height=2)
    age_label.configure(font=("Times New Roman", 12, "bold"))
    age_label.grid(row=3, column=0, padx=5, pady=2, sticky="E")

    Locality = Label(frame_8, text="Locality", fg="white",
                     bg="black", width=10, height=2)
    Locality.configure(font=("Times New Roman", 12, "bold"))
    Locality.grid(row=4, column=0, padx=5, pady=2, sticky="E")
    # status_label.grid_forget()

    Phone_number = Label(frame_8, text="PhoneNumber",
                         fg="white", bg="black", width=10, height=2)
    Phone_number.configure(font=("Times New Roman", 12, "bold"))
    Phone_number.grid(row=5, column=0, padx=5, pady=2, sticky="E")
    # accuracy_label.grid_forget()

    Next_kin = Label(frame_8, text="NextOfKin", fg="white",
                     bg="black", width=10, height=2)
    Next_kin.configure(font=("Times New Roman", 12, "bold"))
    Next_kin.grid(row=6, column=0, padx=5, pady=2, sticky="E")

    gender_label = Label(frame_8, text="Gender", fg="white",
                         bg="black", width=10, height=2)
    gender_label.configure(font=("Times New Roman", 12, "bold"))
    gender_label.grid(row=7, column=0, padx=5, pady=2, sticky="E")

    # admin_label.grid_forget()

    # create textboxes
    f_name = Entry(frame_8, width=37, textvariable=f,
                   font=("Times New Roman", 12, "bold"))
    f_name.grid(row=1, column=1, padx=10, pady=2, ipady=10, sticky="w")
    f_name.bind("<KeyRelease>", caps)

    def caps(o_uppercase):
        o.set(o.get().upper().strip())
    o = StringVar()

    o_name = Entry(frame_8, width=37, textvariable=o,
                   font=("Times New Roman", 12, "bold"))
    o_name.grid(row=2, column=1, padx=10, pady=2, ipady=10, sticky="w")
    o_name.bind("<KeyRelease>", caps)

    age = Entry(frame_8, width=37, font=("Times New Roman", 12, "bold"))
    age.grid(row=3, column=1, padx=10, pady=2, ipady=10, sticky="w")

    locale = Entry(frame_8, width=37, font=("Times New Roman", 12, "bold"))
    locale.grid(row=4, column=1, padx=10, pady=2, ipady=10, sticky="w")

    number = Entry(frame_8, width=37, font=("Times New Roman", 12, "bold"))
    number.grid(row=5, column=1, padx=10, pady=2, ipady=10, sticky="w")

    kin = Entry(frame_8, width=37, font=("Times New Roman", 12, "bold"))
    kin.grid(row=6, column=1, padx=10, pady=2, ipady=10, sticky="w")

    gender = Entry(frame_8, width=12, font=("Times New Roman", 12, "bold"))
    gender.grid(row=7, column=1, padx=10, pady=2, ipady=10, sticky="w")

    gender.config(state="disable")

    var = StringVar()

    def gender_get():
        gender.delete(0, "end")
        if var.get() == "0":
            gender.config(state="normal")
            gender.config(state="normal")
            gender.insert("end", var.get())
            gender.delete(0, "end")

        else:
            gender.config(state="normal")
            gender.delete(0, "end")
            gender.insert("end", var.get())
            gender.config(state="normal")

    R_button0 = tk.Radiobutton(
        frame_8, text="Male", variable=var, value="Male", fg="magenta", bg="black", font=("Times New Roman", 12, "bold"), command=gender_get)

    R_button0.deselect()
    R_button0.grid(row=7, column=1, padx=10, pady=3, ipadx=3, ipady=7)

    R_button1 = tk.Radiobutton(
        frame_8, text="Female", variable=var, value="Female", fg="magenta", bg="black", font=("Times New Roman", 12, "bold"), command=gender_get)

    R_button1.deselect()
    R_button1.grid(row=7, column=1, padx=10,
                   pady=3, ipadx=3, ipady=7, sticky="e")

    R_button2 = tk.Radiobutton(
        frame_8, text="TransGender", variable=var, value="TransGender", fg="magenta", bg="black", font=("Times New Roman", 12, "bold"), command=gender_get)

    R_button2.deselect()
    R_button2.grid(row=7, column=2, padx=10,
                   pady=3, ipadx=3, ipady=7)

    # delete_record = Entry(root, width=30)
    # delete_record.grid(row=9, column=1, padx=20)

    # default values for text boxes
    status = Entry(frame_8, width=37, font=("Times New Roman", 12, "bold"))
    status.grid(row=5, column=1, padx=2, pady=2)
    status.insert(0, "status")
    status.grid_forget()

    accuracy = Entry(frame_8, width=37, font=("Times New Roman", 12, "bold"))
    accuracy.grid(row=6, column=1, padx=2, pady=2)
    accuracy.insert(0, "accuracy")
    accuracy.grid_forget()

    def caps(a_uppercase):
        a.set(a.get().upper())
    a = StringVar()
    admin = Entry(frame_8, width=37, textvariable=a,
                  font=("Times New Roman", 12, "bold"))
    admin.grid(row=7, column=1, padx=2, pady=2, ipady=10)
    admin.insert(0, str(Admin))
    admin.grid_forget()

    def save():
        global records

        if len(str(f_name.get().strip())) >= 3 and len(str(o_name.get().strip())) >= 3 and len(str(age.get().strip())) > 1:
            ages = age.get().strip()
            try:
                check_age = int(ages)

                records = [str(f_name.get().strip()), str(o_name.get().strip()), str(
                    age.get().strip()), str(gender.get().strip()), str(admin.get().strip())]

                f_name.delete(0, "end")
                o_name.delete(0, "end")
                age.delete(0, "end")
                gender.delete(0, "end")
                status.delete(0, "end")
                accuracy.delete(0, "end")
                admin.delete(0, "end")
                root.destroy()
                open1()
            except ValueError:

                mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                               "INVALID AGE, MUST BE DIGITS ONLY!!")

        else:
            mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                           "INVALID DETAILS, CHECK TO ENSURE YOU HAVE ENTERED CORRECT DETAILS")

    def quit_root():
        r_root.destroy()
        root.deiconify()

    def query():
        global r_root
        global delete_record

        root.withdraw()
        r_root = Tk()
        r_root.title("DETECTION OF PNEUMONIA USING DEEP LEARNING")
        r_root.geometry("1020x720")
        r_root.resizable(width=True, height=False)
        r_root.columnconfigure(0, weight=1)
        r_root.configure(bg="#00BFFF")
        # create database or connect to one
        con = sqlite3.connect("Na.db")
        # create a cursor
        c = con.cursor()
        df = pd.read_sql_query("SELECT *, oid FROM d ", con)

        t = Text(r_root)

        t.insert("end", df)
        t.config(state="disable")
        t.grid(row=0, column=0, pady=10, ipadx=70)

        # print(records)
        # commit changes
        con.commit()
        # close connections
        con.close()
        # def quit_root():
        # r_root.destroy()

        # create delete box label
        delete_box_label = Label(
            r_root, text="Select ID To Delete", fg="black", bg="green")
        delete_box_label.configure(font=("Times New Roman", 12, "bold"))
        delete_box_label.grid(row=1, column=0, padx=5, pady=5)

        # create entry for deleting records
        delete_record = Entry(r_root, width=20, font=(
            "Times New Roman", 12, "bold"))
        delete_record.grid(row=2, column=0, padx=5, ipady=7)
        # create delete button
        delete_button = Button(r_root, text="Click To Delete By ID", fg="black", bg="indigo", font=(
            "Times New Roman", 12, "bold"), command=delete_single)
        delete_button.grid(row=3, column=0, columnspan=2,
                           padx=8, pady=5, ipadx=26)

        # delete_all_button = Button(r_root, text="Delete All Records", fg="black", bg="red", font=(
        #     "Times New Roman", 12, "bold"), command=delete_all)
        # delete_all_button.grid(
        #     row=4, column=0, columnspan=2, padx=10, pady=3, ipadx=40)

        back_r_button = Button(r_root, text='GO_Back', fg="magenta", bg="black", font=(
            "Times New Roman", 12, "bold"), command=quit_root)
        back_r_button.grid(row=5, column=0, columnspan=2,
                           padx=3, pady=10, ipadx=20)

        r_root.mainloop()

    def delete_all():

        my_response = mb.askokcancel(
            "DETECTION OF PNEUMONIA USING DEEP LEARNING", "ALL RECORDS WILL BE DELETED")
        if my_response == 1:
            r_root.destroy()

            check_window2 = Tk()
            check_window2.title("DETECTION OF PNEUMONIA USING DEEP LEARNING")
            # signup_window.iconbitmap("/e/malaria_code/kip.ICO")
            check_window2.geometry("750x600")
            check_window2.resizable(width=False, height=False)
            frame_3 = Frame(check_window2, bg="grey", height=150, width=200)
            frame_3.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)
            check_window2.configure(bg="brown")

            blank = Label(frame_3, text="            ", bg="grey")
            blank.configure(font=("Times New Roman", 18, "bold"))
            blank.grid(row=0, column=1,  pady=5)

            intruction3 = Label(frame_3, text='Enter the Authorization kEY to Delete ALL Records',
                                fg="magenta", bg="black", height=2, width=40)
            intruction3.configure(font=("Times New Roman", 12, "bold"))
            intruction3.grid(row=1, column=1, padx=1, pady=5)

            key_label = Label(frame_3, text='Authorization Key',
                              fg="white", bg="blue", height=2, width=14)
            key_label.configure(font=("Times New Roman", 12, "bold"))
            key_label.grid(row=2, padx=2.5, pady=4, sticky="w")

            auto_key = Entry(frame_3, width=40, font=(
                "Times New Roman", 12, "bold"), show='*')
            auto_key.grid(row=2, column=1, padx=4, pady=4, ipady=8, sticky="w")

            def check_auto_key():
                con = sqlite3.connect("Na.db")
                # create a cursor
                c = con.cursor()
                c.execute("SELECT *,oid  FROM act")
                data = c.fetchall()

                act_data = [list(ele) for ele in data]

                # commit changes
                con.commit()
                # close connections
                con.close()

                if str(auto_key.get().strip()) == act_data[-1][0]:
                    # create database or connect to one
                    con = sqlite3.connect("Na.db")
                    # create a cursor
                    c = con.cursor()
                    c.execute('DELETE FROM d;',)

                    # commit changes
                    con.commit()
                    # close connections
                    con.close()

                    check_window2.destroy()
                    query()

                else:
                    mb.showwarning(
                        "DETECTION OF PNEUMONIA USING DEEP LEARNING", "INVALID AUTORIZATION KEY!")

            def back_to_root():
                check_window2.destroy()
                query()

            enter_button1 = Button(frame_3, text="CONFIRM", fg="white", bg="blue", font=(
                "Times New Roman", 12, "bold"), command=check_auto_key)
            enter_button1.grid(row=3, column=1, padx=22,
                               pady=10, ipady=5, ipadx=35, sticky="e")

            back_button2 = Button(frame_3, text="BACK", fg="black", bg="orange", font=(
                "Times New Roman", 12, "bold"), command=back_to_root)
            back_button2.grid(row=3, column=1, padx=18,
                              pady=10, ipady=5, ipadx=40, sticky="w")

        else:
            return

    def delete_single():

        # create database or connect to one
        con = sqlite3.connect("Na.db")
        # create a cursor
        c = con.cursor()
        c.execute("DELETE from d WHERE oid = " + delete_record.get().strip())
        # commit changes
        con.commit()
        # close connections
        con.close()

        r_root.destroy()
        root.deiconify()
        query()

    def log_out():
        sign_out_window.destroy()
        Login()

    def back1():
        root.destroy()
        Login()

    def delete_records():

        query()
        root.withdraw()

    def delete_account():
        global sign_out_window
        global login_username
        global login_password

        answer = mb.askquestion("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                                "ARE YOU SURE YOU WANT TO DELETE YOUR ACCOUNT??")
        if answer == "yes":
            root.destroy()

            sign_out_window = Tk()
            sign_out_window.title("DETECTION OF PNEUMONIA USING DEEP LEARNING")
            # signup_window.iconbitmap("/e/malaria_code/kip.ICO")
            sign_out_window.geometry("1020x720")
            sign_out_window.resizable(width=False, height=False)
            frame_4 = Frame(sign_out_window, bg="grey", height=150, width=90)
            frame_4.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)
            sign_out_window.configure(bg="brown")

            blank = Label(frame_4, text="            ", bg="grey")
            blank.configure(font=("Times New Roman", 18, "bold"))
            blank.grid(row=0, column=1,  pady=5)
            # blank.grid_forget()
            blank1 = Label(frame_4, text="            ", bg="grey")
            blank1.configure(font=("Times New Roman", 18, "bold"))
            #blank1.grid(row = 1,column = 1,  pady = 5)

            intruction3 = Label(frame_4, text='Please Enter Your Old Password and Username',
                                fg="magenta", bg="black", height=2, width=37)
            intruction3.configure(font=("Times New Roman", 14, "bold"))
            intruction3.grid(row=1, column=1, pady=5, sticky="w")

            old_username_label = Label(
                frame_4, text='Old Username', fg="white", bg="blue", height=2, width=15)
            old_username_label.configure(font=("Times New Roman", 12, "bold"))

            old_password_label = Label(
                frame_4, text='Old Password', fg="white", bg="blue", height=2, width=15)
            old_password_label.configure(font=("Times New Roman", 12, "bold"))

            old_username_label.grid(row=2, padx=2.5, pady=4, sticky="w")
            old_password_label.grid(row=3, padx=2.5, pady=4, sticky="w")

            old_username = Entry(frame_4, width=43, font=(
                "Times New Roman", 12, "bold"))
            old_password = Entry(frame_4, width=43, font=(
                "Times New Roman", 12, "bold"), show='*')
            old_username.grid(row=2, column=1, padx=14,
                              pady=4, ipady=8, sticky="w")
            old_password.grid(row=3, column=1, padx=14,
                              pady=4, ipady=8, sticky="w")

            def erase():
                # create database or connect to one
                con = sqlite3.connect("Na.db")
                # create a cursor
                c = con.cursor()
                c.execute("DELETE from Gi WHERE oid = " +
                          str(id_no.get().strip()))

                # commit changes
                con.commit()
                # close connections
                con.close()
                confrm.destroy()
                Login()

            def jump():
                confrm.destroy()

                root1()

            def confirm():
                global confrm
                global id_no
                confrm = Tk()
                confrm.title("DETECTION OF PNEUMONIA USING DEEP LEARNING")
                # signup_window.iconbitmap("/e/malaria_code/kip.ICO")
                confrm.geometry("1020x720")
                confrm.resizable(width=False, height=False)
                frame_6 = Frame(confrm, bg="grey", height=150, width=90)
                frame_6.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)
                confrm.configure(bg="brown")

                blank = Label(frame_6, text="            ", bg="grey")
                blank.configure(font=("Times New Roman", 18, "bold"))
                blank.grid(row=0, column=1,  pady=5)

                intruction3 = Label(frame_6, text='Please Confirm your Signout',
                                    fg="magenta", bg="black", height=2, width=30)
                intruction3.configure(font=("Times New Roman", 14, "bold"))
                intruction3.grid(row=1, column=1, pady=5, sticky="w")

                username_label = Label(
                    frame_6, text='Your name', fg="white", bg="blue", height=2, width=13)
                username_label.configure(font=("Times New Roman", 12, "bold"))
                username_label.grid(row=2, padx=2.5, pady=4, sticky="w")

                username = Entry(frame_6, width=35, font=(
                    "Times New Roman", 12, "bold"))
                username.insert(0, data_to_signout[0])
                username.grid(row=2, column=1, padx=14,
                              pady=4, ipady=8, sticky="w")
                username.config(state="disable")

                password_label = Label(
                    frame_6, text='Password', fg="white", bg="blue", height=2, width=13)
                password_label.configure(font=("Times New Roman", 12, "bold"))
                password_label.grid(row=3, padx=2.5, pady=4, sticky="w")

                password = Entry(frame_6, width=35, font=(
                    "Times New Roman", 12, "bold"))
                password.insert(0, data_to_signout[1])
                password.grid(row=3, column=1, padx=14,
                              pady=4, ipady=8, sticky="w")
                password.config(state="disable")

                id_no = Entry(frame_6, width=15, font=(
                    "Times New Roman", 12, "bold"))
                id_no.insert(0, data_to_signout[2])
                id_no.grid(row=4, column=1, padx=12,
                           pady=4, ipady=8, sticky="w")
                id_no.grid_forget()

                logButton = Button(frame_6, text='CONFIRM', bg="brown", font=(
                    "Times New Roman", 12, "bold"), fg="white", command=erase)
                logButton.grid(row=5, column=1, padx=30, pady=8,
                               ipadx=20, ipady=8, sticky="e")

                logButton1 = Button(frame_6, text='BACK', bg="orange", font=(
                    "Times New Roman", 12, "bold"), fg="black", command=jump)
                logButton1.grid(row=5, column=1, padx=18, pady=8,
                                ipadx=30, ipady=8, sticky="w")

                confrm.mainloop()

            def signout_user():
                global id_no
                global data_to_signout

                if str(old_username.get().strip()) == str(login_username) and str(old_password.get().strip()) == str(login_password):
                    con = sqlite3.connect("Na.db")
                    # create a cursor
                    c = con.cursor()
                    c.execute("SELECT USER_NAME,PASSWORD,oid  FROM Gi")
                    data = c.fetchall()

                    f_data = [list(ele) for ele in data]

                    # commit changes
                    con.commit()
                    # close connections
                    con.close()

                    for a, b, c in f_data:
                        if a == str(old_username.get().strip()) and b == str(old_password.get().strip()):
                            id_no = c
                            data_to_signout = [str(old_username.get().strip()), str(
                                old_password.get().strip()), str(id_no), ]

                            sign_out_window.destroy()
                            confirm()

                        elif a != str(old_username.get().strip()) and b == str(old_password.get().strip()):
                            pass
                        elif a == str(old_username.get().strip()) and b != str(old_password.get().strip()):
                            pass
                        else:
                            pass

                elif str(old_username.get().strip()) != str(login_username) and str(old_password.get().strip()) == str(login_password):
                    mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                                   "INVALID DETAILS, USERNAME IS INCORRECT!!")

                elif str(old_username.get().strip()) == str(login_username) and str(old_password.get().strip()) != str(login_password):
                    mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                                   "INVALID DETAILS, PASSWORD IS INCORRECT!!")

                else:
                    mb.showwarning("DETECTION OF PNEUMONIA USING DEEP LEARNING",
                                   "INVALID DETAILS, BOTH PASSWORD AND USERNAME IS INCORRECT!!")

            delete_account_Button = Button(frame_4, text="Signout", fg="black", font=(
                "Times New Roman", 12, "bold"), bg="red", command=signout_user)
            delete_account_Button.grid(
                row=4, column=1, padx=58, pady=10, ipadx=30, ipady=8, sticky="w")

            logButton = Button(frame_4, text='Logout Instead', bg="blue", font=(
                "Times New Roman", 12, "bold"), fg="white", command=log_out)
            logButton.grid(row=4, column=1, padx=80, pady=5,
                           ipadx=10, ipady=8, sticky="e")

            sign_out_window.mainloop()

            # sign_out_window.destroy()

        else:
            root.destroy()
            root1()

    # create submit button
    submit_button = Button(frame_8, text="Save&OpenImg", fg="white", bg="blue", font=(
        "Times New Roman", 12, "bold"), command=save)
    submit_button.grid(row=8, column=1, columnspan=2,
                       padx=10, pady=5, ipadx=50, ipady=8)

    # create query button
    query_button = Button(frame_8, text="Records", fg="magenta", bg="black", font=(
        "Times New Roman", 12, "bold"), command=query)
    query_button.grid(row=9, column=1, columnspan=2,
                      padx=10, pady=5, ipadx=50, ipady=8)

    # create delete button

    # create back button
    back_button1 = Button(frame_8, text='Log Out', fg="brown", font=(
        "Times New Roman", 12, "bold"), command=back1)
    back_button1.grid(row=9, column=2, columnspan=2,
                      padx=5, pady=8, ipadx=20, ipady=12)

    root.mainloop()


def back():
    main_window.destroy()
    root1()


def results():
    global f_name_results
    global o_name_results
    global age_results
    global gender_results
    global predicted_results
    global accuracy_results
    global served_by
    global acc

    def submit_and_hide():
        submit()
        hide_frame()

    def submit():
        # create database or connect to one
        con = sqlite3.connect("Na.db")
        # create a cursor
        c = con.cursor()
        # INSERT INTO TABLE
        c.execute("INSERT INTO d VALUES (:f_name, :o_name, :age, :gender,:status,:accuracy,:served_by)",
                  {
                      "f_name": f_name_results.get(),
                      "o_name": o_name_results.get(),
                      "age": age_results.get(),
                      "gender": gender_results.get(),
                      "status": predicted_results.get(),
                      "accuracy": accuracy_results.get(),
                      "served_by": served_by.get()
                  })
        R = c.fetchall()
        # commit changes
        con.commit()
        # close connections
        con.close()

        # clear text boxes
        f_name_results.delete(0, "end")
        o_name_results.delete(0, "end")
        age_results.delete(0, "end")
        gender_results.delete(0, "end")
        predicted_results.delete(0, "end")
        accuracy_results.delete(0, "end")
        served_by.delete(0, "end")

    def hide_frame():
        frame.destroy()

    frame = Frame(main_window, bg="grey", height=470, width=560)
    frame.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)

    # create text boxes labels
    f_name_label = Label(frame, text="First Name",
                         fg="green", bg="black", width=10, height=2)
    f_name_label.configure(font=("Times New Roman", 12, "bold"))
    f_name_label.grid(row=0, column=0, padx=20, pady=2)

    o_name_label = Label(frame, text="Other Name",
                         fg="green", bg="black", width=10, height=2)
    o_name_label.configure(font=("Times New Roman", 12, "bold"))
    o_name_label.grid(row=1, column=0, padx=20, pady=2, sticky="E")

    age_label = Label(frame, text="Age", fg="green",
                      bg="black", width=10, height=2)
    age_label.configure(font=("Times New Roman", 12, "bold"))
    age_label.grid(row=2, column=0, padx=20, pady=2, sticky="E")

    gender_label = Label(frame, text="Gender", fg="green",
                         bg="black", width=10, height=2)
    gender_label.configure(font=("Times New Roman", 12, "bold"))
    gender_label.grid(row=3, column=0, padx=20, pady=2, sticky="E")

    prediction_label = Label(frame, text=" Status",
                             fg="green", bg="black", width=10, height=2)
    prediction_label.configure(font=("Times New Roman", 12, "bold"))
    prediction_label.grid(row=5, column=0, padx=20, pady=2, sticky="E")

    accuracy_label = Label(frame, text="Accuracy",
                           fg="green", bg="black", width=10, height=2)
    accuracy_label.configure(font=("Times New Roman", 12, "bold"))
    accuracy_label.grid(row=6, column=0, padx=20, pady=2, sticky="E")

    served_by_l = Label(frame, text="served_by", fg="green",
                        bg="black", width=10, height=2)
    served_by_l.configure(font=("Times New Roman", 12, "bold"))
    served_by_l.grid(row=4, column=0, padx=20, pady=2, sticky="E")
    served_by_l.grid_forget()

    # create textboxes
    f_name_results = Entry(frame, width=30, font=(
        "Times New Roman", 12, "bold"))
    f_name_results.grid(row=0, column=1, padx=2, pady=10, ipady=10)

    o_name_results = Entry(frame,  width=30, font=(
        "Times New Roman", 12, "bold"))
    o_name_results.grid(row=1, column=1, padx=2, pady=2, ipady=10)

    age_results = Entry(frame, width=30, font=("Times New Roman", 12, "bold"))
    age_results.grid(row=2, column=1, padx=2, pady=2, ipady=10)

    gender_results = Entry(frame, width=30, font=(
        "Times New Roman", 12, "bold"))
    gender_results.grid(row=3, column=1, padx=2, pady=2, ipady=10)

    predicted_results = Entry(
        frame, width=30, font=("Times New Roman", 12, "bold"))
    predicted_results.grid(row=5, column=1, padx=2, pady=2, ipady=10)

    accuracy_results = Entry(
        frame, width=30, font=("Times New Roman", 12, "bold"))
    accuracy_results.grid(row=6, column=1, padx=2, pady=2, ipady=10)

    served_by = Entry(frame, width=30, font=("Times New Roman", 12, "bold"))
    served_by.grid(row=4, column=1, padx=2, pady=10, ipady=10)
    served_by.grid_forget()

    # gem1
    # CREATE A BUTTON TO SUBMIT DATA TO DATABSE
    submit_button = Button(frame, text="Update Records", fg="white", bg="blue", font=(
        "Times New Roman", 12, "bold"), command=submit_and_hide)
    submit_button.grid(row=7, column=1, columnspan=2, padx=5, pady=5, ipadx=50)

    submit_button = Button(frame, text="ROI", fg="white", bg="blue", font=(
        "Times New Roman", 12, "bold"), command=get_device)
    submit_button.grid(row=8, column=1, columnspan=2, padx=5, pady=5, ipadx=50)

    submit_button = Button(frame, text="SCROLL ROI_IMG", fg="white", bg="blue", font=(
        "Times New Roman", 12, "bold"), command=get_device)
    submit_button.grid(row=9, column=1, columnspan=2, padx=5, pady=5, ipadx=50)

    # insert other details of the patient into text boxes
    f_name_results.insert("end", records[0])
    o_name_results.insert("end", records[1])
    age_results.insert("end", records[2])
    gender_results.insert("end", records[3])
    served_by.insert("end", records[4])

    # get the predicted accuracy and either infected or uninfected into the text boxes
    accuracy_results.insert("end", acc)
    predicted_results.insert("end", Cell)

    # disable text boxes to avoid acciidental change of data
    f_name_results.config(state="disable")
    o_name_results.config(state="disable")
    age_results.config(state="disable")
    gender_results.config(state="disable")
    served_by.config(state="disable")
    predicted_results.config(state="disable")
    accuracy_results.config(state="disable")


def open1():
    global main_window
    global frame
    main_window = Tk()  # Opens new window
    main_window.configure(bg="grey")
    main_window.title('DETECTION OF PNEUMONIA USING DEEP LEARNING')
    main_window.geometry('1020x720')  # Makes the window a certain size
    main_window.resizable(width=False, height=False)
    main_window.configure(bg="#00BFFF")

    back_buttton = Button(main_window, text='GO_Back', fg="brown", font=(
        "Times New Roman", 12, "bold"), command=back)
    back_buttton.grid(row=0, column=0, padx=2, pady=2, ipadx=10)

    # CREATE AN OPEN BUTTON TO NAVIGATE THROUGH FILES
    open_button = Button(main_window, text='Open Folder TO PROCESS', fg="white", bg="blue", font=(
        "Times New Roman", 12, "bold"), command=LT2)
    open_button.grid(row=0, column=3, padx=2, pady=2, ipadx=10)
    #Button(command=lambda : [some_function(), some_other_function(), some_another_function()])
    results_button = Button(main_window, text="Results", font=(
        "Times New Roman", 12, "bold"), fg="blue", command=results)
    results_button.grid(row=0, column=6, padx=2, pady=2, ipadx=10)

    #hide_results_button = Button(main_window, text = "Hide Results", fg = "white",bg = "grey", command = hide)
    #hide_results_button.grid(row = 0, column = 9)

    main_window.mainloop()


def LT2():
    global c, s
    global s
    global c
    global fl
    ftypes = [('Image', ['*.jpeg', '*.png']), ('All files', '*')]
    dlg = filedialog.Open(filetypes=ftypes)
    fl = dlg.show()
    print(fl)
    # print(fl)
    c, s, top_class, top_probability = predict(fl)
    view_classify(fl, top_probability, top_class, class_names)

    # c, s = predict_cell(fl)
    #root = Tk()
    #T = Text(root, height=4, width=70)
    # T.pack()
    #T.insert(END, s)


Login()
