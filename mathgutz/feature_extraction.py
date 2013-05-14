from nltk import SnowballStemmer

# Here "words" is to be a list of tokens coming from happytokenize.py

stemmer = SnowballStemmer("english")

def snowball(word):
    if word.split("_")[0] == "meta":
        return str(word)
    elif word.split("_")[-1] == "neg":
        new_word = "".join(word.split("_")[:-1]).lower().encode('ascii', 'ignore')
        stem = "%s_neg" % str(stemmer.stem(new_word))
        return stem
    else:
        new_word = word.lower().encode('ascii', 'ignore')
        return str(stemmer.stem(new_word))

badwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
            'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
            'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
            'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'can', 'will', 'just', 'don', 'should', 'now']
badwords = set(badwords)
badwords |= set([snowball(word) for word in badwords])
badwords |= set("%s_neg" % word for word in badwords)

# Convert a list of tokens into a bag-of-words dictionary suitable for
# piping into an NLTK classifier

def bow(words):
    return dict([(word, 1) for word in set(words) if len(word) > 1])

# These functions preprocess a list of tokens in various ways
# before they are piped to "bow"

def filter_snow(words):
    return [snowball(word) for word in words]

def filter_stop(words):
    out = [word for word in words if len(word) > 2]
    return [word for word in out if word not in badwords]

def filter_stem(words):
    stems = [snowball(word) for word in filter_stop(words)]
    return [word for word in stems if word not in badwords]

def bytegram(words):
    good_words = [word for word in words if not word.split("_")[0] == "meta"]
    meta_words = [word for word in words if word.split("_")[0] == "meta"]
    content = "_".join(good_words)
    n = len(content)
    grams = [content[i:i+4] for i in range(0,n-3)] + meta_words
    return grams

def bigram(words):
    good_words = [word for word in words if not word.split("_")[0] == "meta"]
    n = len(good_words)
    bigrams = words + ["_".join(good_words[i:i+2]) for i in range(0,n-1)]

# This is the practical function combining the above.

def extract_features(words, feat_type):
    if feat_type == "bow_nometa":
        return bow([word for word in words if not word.split("_")[0] == "meta"])
    elif feat_type == "bow_stop":
        return bow(filter_stop(words))
    elif feat_type == "bow_snow":
        return bow(filter_snow(words))
    elif feat_type == "bow_stem":
        return bow(filter_stem(words))
    elif feat_type == "bow_bigram":
        return bow(bigram(words))
    elif feat_type == "bow_bytegram":
        return bow(bytegram(words))
    else:
        return bow(words)
