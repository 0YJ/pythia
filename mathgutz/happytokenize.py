import re
import htmlentitydefs

emoticon_string = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]\(\[dDpPoO0$/\:\}\{@\|\\] # mouth      
      |
      [\)\]\(\[dDpPoO0$/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""

phone_number_string = r"""
    (?:
      (?:            # (international)
        \+?[01]
        [\-\s.]*
      )?            
      (?:            # (area code)
        [\(]?
        \d{3}
        [\-\s.\)]*
      )?    
      \d{3}          # exchange
      [\-\s.]*   
      \d{4}          # base
    )"""

html_tag_string = r"""<[^>]+>"""

url_string = r"""
    (?:
      (?i)(?:f|ht)tps?:\/[^\s]+
    )"""

twitter_username_string = r"""(?:@[\w_]+)"""

twitter_hashtag_string = r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)"""

negative_string = r"""
    (?:
      ^(?:never|no|nothing|nowhere|noone|none|not|
          havent|hasnt|hadnt|cant|couldnt|shouldnt|
          wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint
      )$
    )
    |
    n't"""

punct_string = r"""^[.:;!?,]$"""

regex_strings = (
    # URLs:
    url_string
    ,
    # Phone numbers:
    phone_number_string
    ,
    # Emoticons:
    emoticon_string
    ,    
    # HTML tags:
    html_tag_string
    ,
    # Twitter username:
    twitter_username_string
    ,
    # Twitter hashtags:
    twitter_hashtag_string
    ,
    # Remaining word types:
    r"""
    (?:[a-z][a-z'\-_]+[a-z])       # Words with apostrophes or dashes.
    |
    (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
    |
    (?:[\w_]+)                     # Words without apostrophes or dashes.
    |
    (?:\.(?:\s*\.){1,})            # Ellipsis dots. 
    |
    (?:\S)                         # Everything else that isn't whitespace.
    """
    )

word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)
url_re = re.compile(regex_strings[0], re.VERBOSE | re.I | re.UNICODE)
phone_number_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)
emoticon_re = re.compile(regex_strings[2], re.VERBOSE | re.I | re.UNICODE)
html_tag_re = re.compile(regex_strings[3], re.VERBOSE | re.I | re.UNICODE)
twitter_username_re = re.compile(regex_strings[4], re.VERBOSE | re.I | re.UNICODE)
twitter_hashtag_re = re.compile(regex_strings[5], re.VERBOSE | re.I | re.UNICODE)
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"
negative_re = re.compile(negative_string, re.VERBOSE | re.I | re.UNICODE)
punct_re = re.compile(punct_string, re.VERBOSE | re.I | re.UNICODE)

def recognize(s):
    if s.split("_")[0] == "meta":
        return s.encode('ascii', 'ignore')
    elif url_re.search(s):
        if s.split(".")[-1].lower() in ["jpg", "png", "gif"]:
            return "meta_image"
        else:
            return "meta_url"
    elif phone_number_re.search(s):
        return "meta_phone_number"
    elif emoticon_re.search(s):
        return "meta_emoticon"
    elif html_tag_re.search(s):
        return "meta_html_tag"
    else:
        return s.lower().encode('ascii', 'ignore')
    
def html2unicode(s):
    # First the digits:
    ents = set(html_entity_digit_re.findall(s))
    if len(ents) > 0:
        for ent in ents:
            entnum = ent[2:-1]
            try:
                entnum = int(entnum)
                s = s.replace(ent, unichr(entnum))	
            except:
                pass
    # Now the alpha versions:
    ents = set(html_entity_alpha_re.findall(s))
    ents = filter((lambda x : x != amp), ents)
    for ent in ents:
        entname = ent[1:-1]
        try:            
            s = s.replace(ent, unichr(htmlentitydefs.name2codepoint[entname]))
        except:
            pass                    
        s = s.replace(amp, " and ")
    return s

def happy_tokenize(s):
    try:
        s = unicode(s)
    except UnicodeDecodeError:
        s = str(s).encode('string_escape')
        s = unicode(s)
    # Fix HTML character entitites:
    s = html2unicode(s)
    # Tokenize:
    words = word_re.findall(s)
    # Lowercase everything, scrub emoticons, HTML and links
    words = [recognize(x) for x in words]
    stuff = []
    neg = 0
    for word in words:
        if punct_re.search(word):
            neg = 0
        elif word.split("_")[0] == "meta":
            neg = 0
        elif negative_re.search(word):
            neg = ((neg + 1) % 2)
        elif neg == 1:
            word += "_neg"
        stuff.append(word)
    return stuff
