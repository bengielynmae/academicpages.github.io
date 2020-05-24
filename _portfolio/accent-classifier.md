---
title: "Detecting Accents of Non-Native English Speakers"
excerpt: "This project deploys an Accent Classifier model to a simple web application using flask. The algorithm was trained on multiple voice recordings of native and non-native english speakers.<br/><br><img src='/images/accent-classifier/cover.png'>"
collection: portfolio
---

<h2>Overview</h2>

This was a final project output for our <b>Data Applications and Web Development</b> course under Sir Eduardo David in the M.Sc. Data Science program. The goal of the project is to deploy a Machine Learning or Deep Learning model to a simple web application using flask. The model used for this project is based on the work of my class partner [Kyle Ong](https://kykyleoh.github.io).

<h2>Description</h2>
 English is a global language and it is spoken in many different parts of the world. Because of the diversity of people who speak the language, different ways of speaking it and different accents have developed over time. Using the `Wildcat Speech Accent Archive`, our app can classify accents among 6 different countries. 

 The model used was trained on recordings speaking the following phrase:

 *"Please call Stella. Ask her to bring these things with her from the store: Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob. We also need a small plastic snake and a big toy frog for the kids. She can scoop these things into three re bags, and we will go meet her Wednesday at the train statio."*

 If you're interested about how the model works, check out the machine learning project [here](https://kykyleoh.github.io/portfolio/accent-detection).

<h2>Web App</h2>

<h3> Here's a screen recording video of how it works:</h3>

You can upload any audio file (speaking the phrase specified) from your desktop/device into our application. Clicking the button **check your accent** will output the *top 5* predictions of the model along with their probabilities. You can also play and listen to your recording on our app.
<video width='1024' height='768' controls>
  <source src="/files/accent-classifier.mov">
</video>

<h2>Try it yourself!</h2>

We're trying to host this app on the cloud but for the meantime, if you want to try it, you can find the files in my [GitHub](https://github.com/bengielynmae/accent-classifier-web-app).

**How to run it**<br>
Download all the files and folders from my repository and save them together in just one local folder. In your terminal, `cd` into the directory where you saved them and run the following:<br>
* `FLASK_APP=app_flask.py`
* `FLASK_ENV=development`
* `flask run`

  If you can't get it to work, try running `set FLASK_APP=app_flask.py` and `set_FLASK_ENV=development` instead before flask run. 

  For mac users, you can try running `export FLASK_APP=app_flask.py` instead and retaining the next 2 lines of codes. 

Then go to `127.0.0.1:5000` on your browser. 


<h2>Acknowledgements</h2>
<p>This work was done and completed with my project partner Kyle Ong.</p>