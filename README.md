
This repository contains the code for our project **OCR CLASSIFICATION**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to be able to classify scanned documents according to their main type : a bill, an email, a ID card or a scientific publication for example.
Two approaches are observed :
- an NLP (Natural Language Processing) approach, using an OCR algorithm (such as DocTR), then vocabulary analysis through Machine Learning.
- a CV (Computer Vision) approach, using visual feature recognition through Deep Learning.

We also take a look at the existing possibilities of image correction (using a template to straighten other images through feature recognition) and entity extraction (using NLP to recognize and extract locations or names from documents).

This project was developed by the following team :

- Clément Paradan
- Alexandre Jaqua
- Eliott Douieb


You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```

## Streamlit App

The app contains 4 tabs :
- The "Introduction" tab presents the main topic and shows an OCR example.
- The "Classification NLP" tab presents the first approach : using NLP/ML to classify the input document.
- The "Classification CV" tab presents the second approach : using CV/DL to classify the input document.
- The "Image Correction" tab presents a few examples of images where the main feature is straightened and cropped according to a template.
- The "Entity Recognition" tab presents a few examples of images where the main entites (a name, a location, an organization) are extracted.

For all these tabs, you can upload your own image for further testing.

To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).
