import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from numpy import asarray
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from transformers import pipeline
import re
import nltk
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
import nltk
from nltk.corpus import stopwords
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


title = "Analyse du document"
sidebar_name = "Classification_NLP"
def to_string(json):
    transcript2 =""
    for i in range(len(json['pages'])):
        for j in range(len(json['pages'][i]['blocks'])):
            for k in range(len(json['pages'][i]['blocks'][j]['lines'])):
                for l in range(len(json['pages'][i]['blocks'][j]['lines'][k]['words'])):
                    transcript2 += json['pages'][i]['blocks'][j]['lines'][k]['words'][l]['value']+" "
    return transcript2



def run():

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
      Nous allons classifier le document précédent via NLP.
        """
    )
    image_file = st.file_uploader(label = "uploaded_image", type = 'jpg')
    def load_image(image_file):
      img = Image.open(image_file)
      return img



    if image_file :
        st.title("Création du transcript")

        st.markdown(
            """
    Nous allons extraire le texte via Doctr
            """
        )

        img = load_image(image_file)
        st.image(img, width = 700)
        img = load_image(image_file)
        img = asarray(img)
        st.write(len(img))
        if (image_file.size < 20000):
            st.write("L'image est trop petite pour le modèle. Merci de choisir une image aux dimensions supérieures à 20ko.")
        else:
            st.write("L'image à un poids de :",round(image_file.size/1000),"ko ce qui est suffisant pour le modèle NLP" )


        model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        result = model([img])
        json_output = result.export()
        texte = to_string(json_output)
        st.write(texte)




        st.markdown(
            """
            ## Nettoyage des symboles et chiffres

            """
        )

        tokenizer = PunktSentenceTokenizer()
        r2 = re.compile(r"[^a-zA-ZéèàùôâçêëäïöüûÉ]")
        texte = r2.sub(' ', str(texte))
        st.write(texte)


        st.markdown(
            """
            ## Détection de la langue

            """
        )

        def get_lang_detector(nlp, name):
            return LanguageDetector()


        nlp = spacy.load('en_core_web_sm')
        Language.factory('language_detector_v1', func=get_lang_detector)
        nlp.add_pipe('language_detector_v1', last=True)

        def detect_lang(x):
            doc = nlp(x)
            dico = doc._.language
            langue = dico['language']
            return str(langue)

        trad = detect_lang(texte)
        st.write('La langue detectée est : ', trad)


        st.markdown(
            """
            ## Suppression des stop-words



            """
        )

        stop_words_en = set(stopwords.words('english'))
        texte_list = texte.split(' ')
        texte_list = [x for x in texte_list if x]
        texte2=[]
        stop_words_en.update(['»', ':', '{', '}', "'", '|', ',', '.', '""', ';'," "])
        stop_words_fr = set(stopwords.words('french'))


        if trad == 'en':
            stop_list = stop_words_en
        else:
            stop_list = stop_words_fr

        for i in range(len(texte_list)):
            if texte_list[i].lower() not in stop_list:
                texte2.append(texte_list[i].lower())


        st.write(texte2)

        st.markdown(
            """
            ## Traduction

        Nous traduisons uniquement vers l'anglais si le texte est en français.


            """
        )

        fr_to_en = pipeline("translation_fr_to_en",model = "Helsinki-NLP/opus-mt-fr-en")
        if trad=='fr':
            tradEN = fr_to_en(texte.lower())
            st.write('=> La traduction du texte en anglais est :', tradEN)
        else:
            st.write('=> Le texte est déja en anglais.')
            tradEN = texte.split()
        st.write(tradEN)

        st.markdown(
            """
            ## Supression des mots de moins de 3 lettres


            """
        )
        if trad=='fr':
            tradEN = tradEN[0]['translation_text'].split(' ')
        tradEN_v2 = []
        for i in tradEN:
            if len(i)>= 3:
                tradEN_v2.append(i)

        st.write(tradEN_v2)

        st.markdown(
            """
            ## Lemmatization

            On effectue une lemmatisation

            """
        )
        nlp_en = spacy.load('en_core_web_sm')

        def get_lemm_en(x):
            words_lemmas_list = []
            tokens = []
            for i in x:
                doc = nlp_en(i)
                liste = [token.lemma_ for token in doc]
                words_lemmas_list.append(liste[0])
            return words_lemmas_list

#On met le résultat dans une variable et on l'imprime.
        lemma_tradEN = get_lemm_en(tradEN_v2)
        st.write(lemma_tradEN)

        if len(get_lemm_en(tradEN_v2))>10:
            st.write('Il y a suffisamment de mots pour le modèle NLP : ',len(get_lemm_en(tradEN_v2)),'mots détectés')
        else:
            st.write('Nombre de mots insuffisant pour le modèle NLP : ',len(get_lemm_en(tradEN_v2)),'mots détectés')

        st.markdown(
            """
            ## Classification

            Cette lemmatisation est donnée en entrée à notre modèle, qui va retourner
            la classe de document la plus probable.
            """
            )

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
        lemma_tradEN = ' '.join(lemma_tradEN)

        loaded_vectorizer = joblib.load(r'/home/clement/Documents/Datascientist/Git/ocr-classification/model/vectorizer.sav')
        testinput = loaded_vectorizer.transform([lemma_tradEN]).todense()
        loaded_model = joblib.load(r'/home/clement/Documents/Datascientist/Git/ocr-classification/model/best_model.sav')
        result = loaded_model.predict(testinput)
        st.write("La classe prédite est :", dic[result[0]])
        #st.write(type(lemma_tradEN))
