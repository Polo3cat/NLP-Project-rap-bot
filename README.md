# NLP-Project-rap-bot
Chat bot capable of answering to a sentence with a rhyiming one based on a hip-hop corpus.

## Files
+ **filter.py** : reduces the original corpus filtering by genre of music and limiting how many songs.
+ **tagger.py** : given a corpus it creates a pickle file containing a map that given a PoS tagged structure gives the PoS tagged structure that should follow next.
+ **grammer.py** : given a corpus it creates a pickle file with a mapping of each type with it's PoS to the most likely type given a PoS tag.

## Usage

1. Use filter to reduce corpus (optional)
2. Use tagger to get the grammatical structures.
3. Use grammer to generate your vocabulary.