"""Frozen r3.1 phrase copying rule; no learned evaluator or threshold tuning."""
import re
from porter import PorterStemmer

STEM = PorterStemmer()
# Closed-class function words only. Content words are retained, including negation.
STOP = set('a an the and or but if while as of to in on at for from by with into onto upon about than that this these those it its i me my mine we us our ours you your yours he him his she her hers they them their theirs is am are was were be been being do does did have has had can could will would shall should may might must'.split())

def tokens(text, stem=False, content=False):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    if content:
        words = [w for w in words if w not in STOP]
    return [STEM.stem(w) for w in words] if stem else words

def copying_reason(phrase, lyrics, dataset):
    stem = dataset == 'm4a'
    whole = tokens(phrase, stem)
    content = tokens(phrase, stem, True)
    if not whole:
        return 'empty_phrase'
    for line in lyrics.splitlines():
        if whole == tokens(line, stem):
            return 'verbatim_whole_line'
        source = tokens(line, stem, True)
        trigrams = {tuple(source[i:i+3]) for i in range(len(source)-2)}
        if any(tuple(content[i:i+3]) in trigrams for i in range(len(content)-2)):
            return 'verbatim_content_trigram'
    return None

def test_policy():
    assert copying_reason('lost love', 'lost love drift away', 'm4a') is None
    assert copying_reason('nostalgia', 'nostalgia brings me home', 'mpd') is None
    assert copying_reason('lost love', 'lost love', 'mpd') == 'verbatim_whole_line'
    assert copying_reason('drive through the night', 'We drive through the night together', 'mpd') == 'verbatim_content_trigram'
    assert copying_reason('driving through nights', 'drive through night together', 'm4a') == 'verbatim_content_trigram'
    assert copying_reason('love feels distant', 'love feels\ndistant tonight', 'mpd') is None
    assert copying_reason('lost love forever', 'lost and love forever', 'mpd') == 'verbatim_content_trigram'
    assert copying_reason('late-night driving', 'drive all night long', 'mpd') is None

if __name__ == '__main__':
    test_policy()
    print('r3.1 copying policy checks passed')
