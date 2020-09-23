# PartIIProject

Project which comprised 25% of my final degree grade. Was awarded a 72/100 (1st class).

This project implements a system for classifying what type of crime is being discussed in posts from CrimeBB, a dataset of posts scraped from a variety of underground forums (forums which allow criminals to trade in services and knowledge). The system is composed of two models:

 - A model to decide whether a post discusses crime or not, which was implemented using a Recurrent Neural Network (RNN).
 - A constrained k-means clustering model to decide which crime type is discussed in posts that are flagged as discussing crime in our first model.

We also explore whether combining the second model with a rule-based one, which classifies based on a set of keywords, can improve performance. All of the core goals of the project described in the project proposal were met.

Many many more details can be found in the chapters.tex file. This goes through, in over 10,000 words, the planning, underlying concepts, implementation details, and evaluation of the project, and is what was primarily graded by the examiners.
