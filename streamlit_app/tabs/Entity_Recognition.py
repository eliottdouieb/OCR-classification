import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import time
from numpy import asarray

title = "Entity Recognition"
sidebar_name = "Entity_Recognition"


def to_string(json):
    transcript2 =""
    for i in range(len(json['pages'])):
        for j in range(len(json['pages'][i]['blocks'])):
            for k in range(len(json['pages'][i]['blocks'][j]['lines'])):
                for l in range(len(json['pages'][i]['blocks'][j]['lines'][k]['words'])):
                    transcript2 += json['pages'][i]['blocks'][j]['lines'][k]['words'][l]['value']+" "
    return transcript2
def load_image(image_file):
    img = Image.open(image_file)
    return img

def run():

    st.title(title)

    st.markdown(
        """
        Dans cet onglet, nous allons observer comment les algorithmes de NER (Named Entity Recognition)
        permettent d'extraire des éléments importants des documents.
        """
    )

    st.warning('Le modèle de NER peut mettre un peu de temps à charger.')

    st.warning('Attention, utilisez un document en français pour de meilleurs résultats !')
    image_file = st.file_uploader(label = "uploaded_image", type = 'jpg')
    if image_file:
        img = load_image(image_file)
        img = asarray(img)
        #image_file.seek(0)
        #img = Image.open(image_file)
        st.image(img)


    if st.button("Je veux charger ce modèle, j'ai un peu de temps devant moi."):
        tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
        model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
        nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

        if image_file:
            model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
            #image_file.seek(0)
            #img = Image.open(image_file)
            result = model([img])
            json_output = result.export()
            texte = to_string(json_output)
            st.write("Le texte lu par DocTR sur ce document est :", texte)

            compteur = 0
            for entity in nlp(texte):
                if (entity['score'] > 0.90) & (entity['end'] - entity['start'] > 2):
                        compteur += 1
                        st.write("On repère l'entité :", entity['word'], "avec le pourcentage de confiance :", entity['score'])
                        st.write("Il s'agit d'une entité de type ", entity['entity_group'])
            if compteur == 0:
                st.write("Le modèle n'a repéré aucune entité avec une confiance suffisante.")
