---
title: "Whose Art Is It?"
excerpt: "This is a deployment of the DeepArtist model to a simple web application using flask. The algorithm, which trained on `ResNet50`, identifies the artist when shown a painting.<br/><br><img src='/images/artist-classifier/titlecard.png'>"
collection: portfolio
---

<h2>Overview</h2>

This was a final project output for our <b>Data Applications and Web Development</b> course under Sir Eduardo David in the M.Sc. Data Science program. The goal of the project is to deploy a Machine Learning or Deep Learning model to a simple web application using flask. The model used for this project is based on the code of [github.com/SupratimH/DeepArtist](https://github.com/SupratimH/applying-ml-use-cases/tree/master/DeepArtist-Identify-Artist-From-Art).

<h1>Description</h1>
In the age of the internet, digital captures of art and paintings have become extensive - from the works of old masters to creations of modern painters. This online presence of artwork is important for art collectors, educators, and students alike for convenient viewing and analysis of pieces from our history. An accurate way to automatically classify paintings will help museums, curators, or collectors quickly organize digital collections. This will also help consumer art appreciators to quickly gain information and insights about a certain painting they are curios about.

This web app can identify paintings as old as 1471 with styles ranging from **renaissance**, **romanticism**, **baroque**, **impressionism**, and **post-impressionism**. This can also classify a modern painting and see whose style it most resembles to. 

<h2>Web App</h2>
This is how the web app looks like: 

<img src='/images/artist-classifier/artist-classifier-home-page.png' width='1000' height='1000'>

You can choose an image file of any painting from your desktop/device. 

<img src='/images/artist-classifier/artist-classifier-upload.png' width='1000' height='1000'>

Upon clicking the **Reveal Artist** button, it would tell you who the artist is. The deep learning model might take a few seconds to preidct but it will also show you the accuracy of the classification. 

<img src='/images/artist-classifier/artist-classifier-prediction.png' width='1000' height='1000'>


<h2>Try it for yourself</h2>
If you want to try it, you can find the files in my [GitHub](https://github.com/bengielynmae/cnn-artist-identifier-web-app).

**How to run it**<br>
Download all the files and folders from my repository and save them together in just one local folder. In your terminal, `cd` into the directory where you saved them and run the following:<br>
* `FLASK_APP=app_flask.py`
* `FLASK_ENV=development`
* `flask run`

  If you can't get it to work, try running `set FLASK_APP=app_flask.py` and `set_FLASK_ENV=development` instead before flask run. 

  For mac users, you can try running `export FLASK_APP=app_flask.py` instead and retaining the next 2 lines of codes. 

Then go to 127.0.0.1:5000 on your browser. 

A video on how it works can be downloaded [here](/files/artist-classifier.mov)


<h2>Acknowledgements</h2>
<p>I would like to thank Kyle Ong for his patience in explaining JavaScript to me. Without it, I wouldn't have been able to build this project.</p>