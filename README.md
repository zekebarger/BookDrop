# BookDrop
A web app that predicts upcoming price drops for books on Amazon.com

See the demo here: www.book-drop.site

## How to run the app locally
You will need to install [Tesseract OCR](https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/) and modify
line 28 of BookDrop.py to reflect the location of your Tesseract installation.
Then clone the app, navigate to /heroku_app/, and run:

```
pip install -r requirements.txt
streamlit run BookDrop.py
```
