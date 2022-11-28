import streamlit as st
from PIL import Image
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import numpy as np
import pytesseract
from numpy import asarray
import plotly.figure_factory as ff
from matplotlib import pyplot as plt
import joblib
import cv2
title = "Classification_Deep_Learning"
sidebar_name = "Classification_Deep_Learning"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
      Notre allons classifier une image via Deep Learning
        """
    )
    image_file = st.file_uploader(label = "uploaded_image", type = 'jpg')
    def load_image(image_file):
      img = Image.open(image_file)
      return img

    img = load_image(image_file)
    st.image(img, width = 700)

    img = img.resize((260,260))
    img_array = np.array(img)
    img_batch = np.expand_dims(img_array, axis = 0)

    if image_file:
        dic = {0:"facture",
        1:"id_pieces",
        2:"justif_domicile",
        3:'passeport',
        4:'paye',
        5:'carte_postale',
        6:'rrc.csv',
        7:'form',
        8:'scientific_publication',
        9:'advetisement',
        10:'letter',
        11:'resume',
        12:'specification',
        13:'handwritten',
        14:'memo',
        15:'invoice',
        16:'budget',
        17:'news_article',
        18:'presentation',
        19:'file_folder',
        20:'scientific_report',
        21:'email',
        22:'questionnaire',
        }

        loaded_model = joblib.load(r'/home/clement/Documents/Datascientist/Git/ocr-classification/model/Deep.pkl')
        result = loaded_model.predict(img_batch)
        st.write("La classe prédite est :", dic[np.argmax(result[0])])

