import spacy


class Lemmatizer:
    def __init__(self, model="en_core_web_sm", batch_size=500):
        # Load spaCy once, when the object is created, with parser and ner switched off (faster).
        # The model must already be installed in the venv (one-time, in the terminal:
        #   python -m spacy download en_core_web_sm)
        self.nlp = spacy.load(model, disable=["parser", "ner"])
        self.batch_size = batch_size

    def lemmatize_clauses(self, df):

        results = []
        # nlp.pipe(...) - feeds cleaned clauses through whole chain, 500 at a time (batch_size).
        # For EACH clause, inside nlp.pipe:
        #   1. TOKENISATION : the tokenizer splits the clause into words (tokens).
        #   2. Tagger labels each token as noun, verb, adjective etc.
        #   3. LEMMATISATION : the lemmatizer uses labels to find word's base form.
        # What comes out is a `doc`: the processed clause, holding every token plus its lemma.
        # `for doc in` gives each processed clause the name `doc`, one at a time.
        for doc in self.nlp.pipe(df["clean_text"], batch_size=self.batch_size):

            words = []

            # A doc is a sequence of tokens, so we can loop through it word by word.
            # Tokenisation and lemmatisation have ALREADY happened by this point;
            # here we are just reading the results.
            for token in doc:

                # token.text   = the word as it appeared, e.g. "parties"
                # token.lemma_ = its base form found by the lemmatizer, e.g. "party"
                words.append(token.lemma_)

            # Join the base forms back into one string, so each clause is a single text again.
            # e.g. ["the", "party", "shall", "terminate"] -> "the party shall terminate"
            results.append(" ".join(words))


        # Add the lemmatised clauses as a new column, in the same order as the original rows.
        df["lemmatized_text"] = results
        return df