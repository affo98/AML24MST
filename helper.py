genre_names = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]
genre_codes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Import these into your working script to make sure that we all have the same codes
id2label = {id_: label for id_, label in zip(genre_codes, genre_names)}
label2id = {label: id_ for label, id_ in zip(genre_names, genre_codes)}
