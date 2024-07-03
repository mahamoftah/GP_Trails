import re


def preprocessing(text):

    pattern = re.compile(r'[*]')
    text = pattern.sub('', text)

    return text
