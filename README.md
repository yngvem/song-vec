# SongVec — Exploring lyrics from Billboard Hot 100 with Python
## About the app
**[See the web app](https://song-vec.herokuapp.com)**

Word vectors are a way to represent words in a high-dimensional space. 
The most commonly known word vectors are those obtained from the `word2vec` 
algorithm, which tries to find vectors that are predictive of neighbouring
words. For a very good introduction to word vectors, we recommend 
[this](https://www.youtube.com/watch?v=vkfXBGnDplQ) presentation by Christopher
Moody.

The word vectors we used here are GloVe vectors trained on the 
[GloVe common crawl](https://nlp.stanford.edu/projects/glove/)
dataset. GloVe is a newer algorithm that is slightly different from `word2vec`,
but both algorithms gives us a vector representation of words. We used the the
pretrained `en_core_web_md` GloVe model in `spaCy`.

In this project, we wanted to investigate how word vectors can be used to 
explore song lyrics. We scraped the lyrics of 60 songs from the Billboard 
hot-100 for the past decade, and computed their average word vectors. From
these average vectors, we can investigate the songs similarities with specific
phrases, and find songs that are about similar topics.

To easily inspect if the results made sense, we also implemented a simple
keyword extraction algorithm that we used on all songs. This keyword extraction
algorithm was based on [this](https://medium.com/better-programming/extract-keywords-using-spacy-in-python-4a8415478fbf)
blog post. The algorithm works by extracting the most common propositions,
adjectives and nouns from the lyrics.

## Creating the web app
To create the web app, we used [Streamlit](https://streamlit.io), a free tool that makes it extremely easy to create a data science dashboard with Python. It is extremely intuitive to work with locally, and deploying the app on Heroku was straightforward once we found [this tutorial](https://github.com/Taxuspt/heroku_streamlit_nginx/). The [code for the live application](https://github.com/yngvem/song-vec/blob/master/SongVec.py) was less than 200 lines of code, and we are pretty happy with the results!

## Deploying your own app
If you want to deploy your own version of this app. Maybe with different songs,
then you only need to modify the [`prepare_app.py`](https://github.com/yngvem/song-vec/blob/master/prepare_app.py) file and
run it before running [`streamlit`](https://streamlit.io)` run `[`SongVec.py`](https://github.com/yngvem/song-vec/blob/master/SongVec.py).
