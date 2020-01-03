import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from get_threads import get_from_keyword


def remove_stops(text):
    return [word for word in text if ((word not in stopwords.words('english') and (len(word) >= 3)))]


def remove_non_ascii(text):
    return "".join(i for i in text if ord(i) < 128)


def remove_tags(text):
    tags_to_remove = [
        r"\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*(.|\n)*?\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*",
        r"\[IMG\](.|\n)*?\[/IMG\]",
        r"\s[^@\s]+@[^@]+\.[^@\s]+\s"
    ]

    for regex in tags_to_remove:
        text = re.sub(regex, "", text)

    return text


def process_text(text):
    return(remove_stops(simple_preprocess(remove_tags(remove_non_ascii(text)))))

