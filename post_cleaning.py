import re


def remove_non_ascii(text):
    return "".join(i for i in text if ord(i) < 128)


def remove_tags(text):
    tags_to_remove = [
        r"\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*(.|\n)*?\*\*\*(IMG|QUOTE|CODE|CITING|LINK|IFRAME)\*\*\*",
        r"\[IMG\](.|\n)*?\[/IMG\]"
    ]

    for regex in tags_to_remove:
        text = re.sub(regex, "", text)

    return text


def process_text(text):
    return(remove_tags(remove_non_ascii(text)))


with open('examples.data', 'r', encoding='utf8') as f:
    examples = f.read().split("===LINESPLIT===")[:-1]

examples.append("[IMG]wibawibdwd[/IMG]wjadviwoudvbw")

for e in examples:

    print(process_text(e))
