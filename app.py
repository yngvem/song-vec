import re
from collections import Counter
from string import punctuation

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spacy
import streamlit as st


def get_vector(phrase):
    vec = nlp(phrase).vector
    return vec / np.linalg.norm(vec)


@st.cache(allow_output_mutation=True)
def prepare_variables():
    print("Loading NLP model")
    nlp = spacy.load("en_core_web_md")

    print("Loading cache")
    data = pd.read_csv("data.csv")

    M = np.load("song_vectors.npy")
    scaled_song_vectors = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-16)

    return nlp, data, scaled_song_vectors


nlp, data, scaled_song_vectors = prepare_variables()

# Sidebar
phrase1 = st.sidebar.text_input("Phrase 1: ", "Dance")
phrase2 = st.sidebar.text_input("Phrase 2: ", "Love")
songs = {
    f"{row.Artist} - {row.Song}": row.id
    for _, row in data.sort_values("Artist").iterrows()
}
song = st.sidebar.selectbox("Song", list(songs))
selected_song_id = songs[song]
num_similar = st.sidebar.slider("Number of similar songs", 1, 40, 3)


# Main window
st.title("Song vector exploration")
st.header("Song lyric exploration with word vectors")
st.markdown(
    """\
Word vectors are a way to represent words in a high-dimensional space. The most commonly
known word vectors are those obtained from the `word2vec` algorithm, which tries to find
vectors that are predictive of neighbouring words. For a very good introduction to
word vectors, we recommend [this](https://www.youtube.com/watch?v=vkfXBGnDplQ) presentation
by Christopher Moody.

The word vectors we used here are GloVe vectors trained on the [GloVe common crawl](https://nlp.stanford.edu/projects/glove/)
dataset. GloVe is a newer algorithm that is slightly different from `word2vec`, but both algorithms
gives us a vector representation of words. We used the the pretrained `en_core_web_md` GloVe 
model in `spaCy`.

In this project, we wanted to investigate how word vectors can be used to explore song lyrics.
We scraped the lyrics of 60 songs from the billboard hot-100 for the past decade, and computed 
their average word vectors. From these average vectors, we can investigate the songs similarities
with specific phrases, and find songs that are about similar topics.

To easily inspect if the results made sense, we also implemented a simple keyword extraction algorithm
that we used on all songs. This keyword extraction algorithm was based on [this](https://medium.com/better-programming/extract-keywords-using-spacy-in-python-4a8415478fbf)
blog post. The algorithm works by extracting the most common propositions, adjectives and nouns from
the lyrics.
"""
)
st.subheader("Similarity between song lyrics and phrases")

## Create Plotly figure
### Compute phrase similarities
data[f"{phrase1} score"] = scaled_song_vectors @ get_vector(phrase1)
data[f"{phrase2} score"] = scaled_song_vectors @ get_vector(phrase2)

fig = go.FigureWidget(
    px.scatter(
        data,
        x=f"{phrase1} score",
        y=f"{phrase2} score",
        hover_data=["Song", "Artist", "Keywords"],
    )
)
scatter = fig.data[0]
fig.layout['autosize'] = True

### Set color
colors = ["navy"] * len(data)
colors[selected_song_id] = "tomato"
sizes = [10] * len(data)
sizes[selected_song_id] = 15
with fig.batch_update():
    scatter.marker.color = colors
    scatter.marker.size = sizes
st.plotly_chart(fig)

## Single song info
song = data.loc[selected_song_id, "Song"]
artist = data.loc[selected_song_id, "Artist"]
st.subheader(f"Information about {song} by {artist}")

### Keywords
st.markdown("#### Keywords")
st.text(data.loc[selected_song_id, "Keywords"])

### Similar songs
st.markdown("#### Most similar lyrics in other songs")
song_similarities = scaled_song_vectors @ scaled_song_vectors[selected_song_id]
top_song_ids = np.argsort(song_similarities)

df = {"Artist": [], "Song": [], "Keywords": [], "Similarity score": []}
for song_id in top_song_ids[-2 : -(2 + num_similar) : -1]:
    df["Artist"].append(data["Artist"][song_id])
    df["Song"].append(data["Song"][song_id])
    df["Keywords"].append(data["Keywords"][song_id])
    df["Similarity score"].append(song_similarities[song_id])

df = pd.DataFrame(df)
df.index = [""] * len(df)
st.table(df)

### Lyrics
st.markdown("#### Lyrics")
song_lyrics = data.loc[selected_song_id, "lyrics"]
st.text_area("", song_lyrics, height=400)


st.header("Why are songs with different lyrics so similar?")
st.markdown("""\
The lyrics are all very similar to each other and when we compare lyrics with single words, 
the similarity score decreases drastically. Our way of thinking about this is that mixing words 
is similar to mixing colours. If we mix many different colours, we get different shades of brown.
We can have a red-ish brown and we can have a green-ish brown, but those colours are more similar
to each other than they are to their red and green counterprats.

Similarly, when we mix many words, we get "word mushes" that we can think of as different shades 
of a sort of "word brown". Some word mushes are more similar to the word "love" than others, 
and some word mushes are more similar to the word "sad" than others. However, all word mushes 
are more similar to each other than they are to their single-word counterparts, just like how 
a red-ish brown is more similar to another shade of brown than it is to red.
""")
st.header("Credits")
st.markdown("""\
This app was made by Yngve Mardal Moe and Marie Roald.
To make the dashboard, we used [`streamlit`](https://www.streamlit.io/). The code is available
on [GitHub](https://github.com/yngvem/song-visualisation).
""")
