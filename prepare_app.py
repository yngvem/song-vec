import re
import textwrap
from collections import Counter
from string import punctuation

import bs4
import numpy as np
import pandas as pd
import requests
import spacy
import tswift
from sense2vec import Sense2VecComponent
from tqdm import tqdm


def get_hotwords(text):
    text = re.sub(r"\[.*?\]", "", text.lower())
    meta_song_words = ["verse", "chorus", "refrain"]

    # Below is from https://medium.com/better-programming/extract-keywords-using-spacy-in-python-4a8415478fbf
    # with an extra check to remove "meta words", like verse, chorus and refrain.
    result = []
    pos_tag = ["PROPN", "ADJ", "NOUN"]  # 1
    doc = nlp(text.lower())  # 2
    for token in doc:
        # 3
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        if token.text in meta_song_words:
            continue
        # 4
        if token.pos_ in pos_tag:
            result.append(token.text)

    return result  # 5


def get_keyword_string(text):
    output = get_hotwords(text)

    hashtags = [(x[0]) for x in Counter(output).most_common(5)]

    return ", ".join(hashtags)


# Get the songs from the billboard hot 100 for the past decade
print("Downloading Billboard hot-100 for 2010-2020...")
req = requests.get("https://www.billboard.com/charts/decade-end/hot-100")
soup = bs4.BeautifulSoup(req.content)

print("Scraping the Billboard hot-100 for artist and song names. Lyrics are obtained with the `tswift` library...")
songs = []
artists = []
lyrics = []
positions = []
for i, song in tqdm(
    list(enumerate(soup.find_all("div", {"class": "ye-chart-item__title"})))
):
    song_title = song.text.strip()
    artist = song.find_next_sibling(
        "div", {"class": "ye-chart-item__artist"}
    ).text.strip()
    artist = artist.split("featuring")[0].strip()
    try:
        lyrics.append(tswift.Song(song_title, artist).lyrics)
    except tswift.TswiftError:
        continue
    songs.append(song_title)
    artists.append(artist)
    positions.append(i + 1)


nlp_model = "en_core_web_md"
print(f"Loading NLP model: {nlp_model}...")
nlp = spacy.load(nlp_model)
print("Analysing lyrics...")
song_vectors = np.stack([nlp(lyric).vector for i, lyric in enumerate(tqdm(lyrics))])

print("Extracting keywords from lyrics...")
keywords = [get_keyword_string(lyric) for lyric in tqdm(lyrics)]

print("Constructing data frame...")
lyrics = [l for i, l in enumerate(lyrics)]
songs = [s for i, s in enumerate(songs)]
artists = [a for i, a in enumerate(artists)]

data = pd.DataFrame({
    "Song": songs,
    "Artist": artists,
    "position": positions,
    "id": range(len(songs)),
    "Keywords": keywords,
    "lyrics": lyrics
})

print("Saving data...")
data.to_csv("data.csv")
np.save("song_vectors.npy", song_vectors)
