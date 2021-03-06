import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess


def remove_stops(text):
    return [word for word in text if word not in stopwords.words('english')]


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
    return remove_stops(simple_preprocess(remove_tags(remove_non_ascii(text)), min_len=3))


def process_text_s(text):
    return ' '.join(process_text(text))
