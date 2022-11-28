import streamlit as st
from PIL import Image
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import numpy as np
import pytesseract
from numpy import asarray
import plotly.figure_factory as ff
from matplotlib import pyplot as plt

title = "OCR Classification"
sidebar_name = "Introduction"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
      Notre projet consiste à classifier des documents scannés dans différentes catégories.
      Vous pouvez uploader un document ci-dessous.
        """
    )
    image_file = st.file_uploader(label = "uploaded_image", type = 'jpg')
    def load_image(image_file):
      img = Image.open(image_file)
      return img

    if image_file:
      img = load_image(image_file)
      file_details = {"filename":image_file.name, "filetype":image_file.type, "filesize":image_file.size}
      st.write(file_details)
      st.image(img,width=700)
      img = asarray(img)




      model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

      result = model([img])


      synthetic_pages = result.synthesize()
      st.image(synthetic_pages[0])
