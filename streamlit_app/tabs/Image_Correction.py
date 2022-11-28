import streamlit as st
import pandas as pd
import numpy as np
import cv2
#rz
title = "Redressement d'image"
sidebar_name = "Image_Correction"

template = cv2.imread('/home/clement/Documents/Datascientist/Git/ocr-classification/images/11.jpg', cv2.IMREAD_COLOR)
MAX_FEATURES = 10000
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2, MAX_FEATURES, GOOD_MATCH_PERCENT):

    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = list(matcher.match(descriptors1, descriptors2, None))

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]


    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))


    # Compute a euclidean distance between projection keypoints and keypoints matching
    points1_vect = np.ones([points1.shape[0], 3])
    points1_vect[...,:2] = points1
    projection =  points1_vect @ h.T
    projection = projection[...,:2]/projection[...,2:]
    distance =  np.sqrt(np.square(projection-points2).sum(-1)).mean()

    return im1Reg, h, distance

def run():

    st.title(title)

    st.markdown(
        """
        Dans cet onglet, nous allons observer comment les algorithmes d'OCR (tels
        que DocTR) redressent les images avant d'en extraire les informations.
        """
    )

    st.markdown(
        """
        Nous allons chercher à redresser des photos de cartes d'identité.
        Il faut d'abord définir un template :
        """
    )
    st.image(template, width = 1000)

    st.markdown(
        """
        Et voici les images que nous allons chercher à redresser :
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.header("Image 1")
        st.image('/home/clement/Documents/Datascientist/Git/ocr-classification/images/1.jpg')
    with col2:
        st.header("Image 2")
        st.image('/home/clement/Documents/Datascientist/Git/ocr-classification/images/2.jpg')

    col3, col4 = st.columns(2)
    with col3:
        st.header("Image 3")
        st.image('/home/clement/Documents/Datascientist/Git/ocr-classification/images/3.jpg')
    with col4:
        st.header("Image 4")
        st.image('/home/clement/Documents/Datascientist/Git/ocr-classification/images/4.jpg')


    #plt.figure(figsize=(20,15))
    #plt.subplot(2,2,1)
    #plt.imshow('..\\..\\images\\1.jpg')
    #plt.xticks([])
    #plt.yticks([])
    #plt.subplot(2,2,2)
    #plt.imshow('..\\..\\images\\2.jpg')
    #plt.xticks([])
    #plt.yticks([])
    #plt.subplot(2,2,3)
    #plt.imshow('..\\..\\images\\3.jpg')
    #plt.xticks([])
    #plt.yticks([])
    #plt.subplot(2,2,4)
    #plt.imshow('..\\..\\images\\4.jpg')
    #plt.xticks([])
    #plt.yticks([])
    #plt.show()

    st.subheader(
        """
        #Pouvez-vous retrouver les bons paramètres pour ces 4 exemples ?
        """
        )
    MAX_FEATURES = 10000
    GOOD_MATCH_PERCENT = 0.15

    st.markdown(
        """
        Exemple 1
        """
        )
    MAX_FEATURES = st.slider('MAX FEATURES',0, 30000, value = 10000, step = 5000, key = 'feature1')
    GOOD_MATCH_PERCENT = st.slider('GOOD MATCH PERCENT',0.0, 1.0, value = 0.2, step = 0.05, key = 'max1')
    im = cv2.imread('/home/clement/Documents/Datascientist/Git/ocr-classification/images/1.jpg', cv2.IMREAD_COLOR)
    imReg, h, distance = alignImages(im, template, MAX_FEATURES, GOOD_MATCH_PERCENT)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Original image")
        st.image(im[...,::-1])
    with col2:
        st.header("Template image")
        st.image(template[...,::-1])
    with col3:
        st.header("Result")
        st.image(imReg[...,::-1])



    #plt.figure(figsize=(20,15))
    #plt.subplot(131)
    #plt.imshow(im[...,::-1])
    #plt.title('Original image')
    #plt.subplot(132)
    #plt.imshow(template[...,::-1])
    #plt.title('Template image')
    #plt.subplot(133)
    #plt.imshow(imReg[...,::-1])
    #plt.title('Result')
    #plt.show()

    st.markdown(
        """
        Exemple 2
        """
        )
    MAX_FEATURES = st.slider('MAX_FEATURES',0, 30000, value = 10000, step = 5000, key = 'feature2')
    GOOD_MATCH_PERCENT = st.slider('GOOD_MATCH_PERCENT',0.0, 1.0, value = 0.2, step = 0.05, key = 'max2')
    im2 = cv2.imread('/home/clement/Documents/Datascientist/Git/ocr-classification/images/2.jpg', cv2.IMREAD_COLOR)
    imReg2, h, distance = alignImages(im2, template,MAX_FEATURES, GOOD_MATCH_PERCENT)

    col4, col5, col6 = st.columns(3)
    with col4:
        st.header("Original image")
        st.image(im2[...,::-1])
    with col5:
        st.header("Template image")
        st.image(template[...,::-1])
    with col6:
        st.header("Result")
        st.image(imReg2[...,::-1])

    #plt.figure(figsize=(20,15))
    #plt.subplot(131)
    #plt.imshow(im2[...,::-1])
    #plt.title('Original image')
    #plt.subplot(132)
    #plt.imshow(template[...,::-1])
    #plt.title('Template image')
    #plt.subplot(133)
    #plt.imshow(imReg[...,::-1])
    #plt.title('Result')
    #plt.show()

    st.markdown(
        """
        Exemple 3
        """
        )
    MAX_FEATURES = st.slider('MAX_FEATURES',0, 30000, value = 10000, step = 5000, key = 'feature3')
    GOOD_MATCH_PERCENT = st.slider('GOOD_MATCH_PERCENT',0.0, 1.0, value = 0.2, step = 0.05, key = 'max3')
    im3 = cv2.imread('/home/clement/Documents/Datascientist/Git/ocr-classification/images/3.jpg', cv2.IMREAD_COLOR)
    imReg3, h, distance = alignImages(im3, template,MAX_FEATURES, GOOD_MATCH_PERCENT)

    col7, col8, col9 = st.columns(3)
    with col7:
        st.header("Original image")
        st.image(im3[...,::-1])
    with col8:
        st.header("Template image")
        st.image(template[...,::-1])
    with col9:
        st.header("Result")
        st.image(imReg3[...,::-1])

    #plt.figure(figsize=(20,15))
    #plt.subplot(131)
    #plt.imshow(im3[...,::-1])
    #plt.title('Original image')
    #plt.subplot(132)
    #plt.imshow(template[...,::-1])
    #plt.title('Template image')
    #plt.subplot(133)
    #plt.imshow(imReg[...,::-1])
    #plt.title('Result')
    #plt.show()

    st.markdown(
        """
        Exemple 4
        """
        )
    MAX_FEATURES = st.slider('MAX_FEATURES',0, 30000, value = 10000, step = 5000, key = 'feature4')
    GOOD_MATCH_PERCENT = st.slider('GOOD_MATCH_PERCENT',0.0, 1.0, value = 0.2, step = 0.05, key = 'max4')
    im4 = cv2.imread('/home/clement/Documents/Datascientist/Git/ocr-classification/images/4.jpg', cv2.IMREAD_COLOR)
    imReg4, h, distance = alignImages(im4, template,MAX_FEATURES, GOOD_MATCH_PERCENT)

    col10, col11, col12 = st.columns(3)
    with col10:
        st.header("Original image")
        st.image(im4[...,::-1])
    with col11:
        st.header("Template image")
        st.image(template[...,::-1])
    with col12:
        st.header("Result")
        st.image(imReg4[...,::-1])

    #plt.figure(figsize=(20,15))
    #plt.subplot(131)
    #plt.imshow(im4[...,::-1])
    #plt.title('Original image')
    #plt.subplot(132)
    #plt.imshow(template[...,::-1])
    #plt.title('Template image')
    #plt.subplot(133)
    #plt.imshow(imReg[...,::-1])
    #plt.title('Result')
    #plt.show()

    st.markdown(
    """
    ## Essayez par vous-mêmes !
    """
    )
    image_file = st.file_uploader(label = "image à redresser", type = 'jpg')
    template_file = st.file_uploader(label = "image template", type = 'jpg')
    if image_file & template_file:
        image_file.seek(0)
        template_file.seek(0)
        MAX_FEATURES = st.slider('MAX_FEATURES',0, 30000, value = 10000, step = 5000, key = 'feature5')
        GOOD_MATCH_PERCENT = st.slider('GOOD_MATCH_PERCENT',0.0, 1.0, value = 0.2, step = 0.05, key = 'max5')
        im5 = cv2.imread(image_file, cv2.IMREAD_COLOR)
        template5 = cv2.imread(template_file, cv2.IMREAD_COLOR)
        imReg5, h, distance = alignImages(im5, template5, MAX_FEATURES, GOOD_MATCH_PERCENT)

        col13, col14, col15 = st.columns(3)
        with col13:
            st.header("Original image")
            st.image(im5[...,::-1])
        with col14:
            st.header("Template image")
            st.image(template[...,::-1])
        with col15:
            st.header("Result")
            st.image(imReg5[...,::-1])

        #plt.figure(figsize=(20,15))
        #plt.subplot(131)
        #plt.imshow(im5[...,::-1])
        #plt.title('Original image')
        #plt.subplot(132)
        #plt.imshow(template5[...,::-1])
        #plt.title('Template image')
        #plt.subplot(133)
        #plt.imshow(imReg[...,::-1])
        #plt.title('Result')
        #plt.show()
