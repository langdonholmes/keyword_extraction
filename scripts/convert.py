"""Convert sequence labels from (st, end) tuples to spaCy v3
.spacy format."""

import srsly
import typer
import warnings
from pathlib import Path
from tqdm.auto import tqdm
import re

import spacy
from spacy.tokens import Doc, DocBin
if not Doc.has_extension('name'):
    Doc.set_extension('name', default=None)
nlp = spacy.blank('en')

from sklearn.model_selection import train_test_split

def main(input_path: Path, train_percent: int) -> None:
    input_path = Path(input_path)
    
    if input_path.suffix == '.jsonl':
        raw = list(srsly.read_jsonl(input_path))
        if any([key not in raw[0].keys() for key in ['name', 'text', 'labels']]):
            warnings.warn("Raw data is expected to be json lines with name, text, labels keys.")
            return
        docs = list(convert(raw))
    elif input_path.suffix == '.spacy':
        docs = list(DocBin().from_disk(input_path).get_docs(nlp.vocab))
    else:
        warnings.warn("Unknown filetype. '.spacy' and '.jsonl' are supported.")
        return
    
    train, _remains = train_test_split(docs, train_size=train_percent/100, random_state=0)
    dev, test = train_test_split(_remains, train_size=0.5, random_state=0)
    to_docbin(train, 'corpus/train.spacy')
    to_docbin(dev, 'corpus/dev.spacy')
    to_docbin(test, 'corpus/test.spacy')

def to_docbin(docs, output_path):
    db = DocBin()
    for doc in docs:
        db.add(doc)
    db.to_disk(Path(output_path))
    
def get_label(sequence_label: iter, default_label_name='keyterm') -> iter:
    '''If no label name is provided, fill that value in with a default.
    Assumes that sequence_label is either (st, end) or (st, end, label)
    '''
    if len(sequence_label) == 2:
        st, end = sequence_label
        label = default_label_name
    else:
        st, end, label = sequence_label
    return st, end, label

def refine_span(doc, st, end, label):
    '''SpaCy sequence labels cannot have leading or trailing whitespace
    or cross token boundaries.
    Shrinks sequences and update their indices.
    If a label crosses a token boundary, msg the problem 
    and shrink the sequence to the nearest token boundary.
    '''
    orig_sequence = doc.text[st:end]
    
    # matches longest sequence that starts and ends with a letter, number, or round bracket.
    no_leading_trailing_space_or_punct = re.compile(r'[\(a-zA-Z0-9].*[a-zA-Z0-9\)]') 
    match = re.search(no_leading_trailing_space_or_punct, orig_sequence)
    new_sequence = match[0] if match else orig_sequence

    if new_sequence != orig_sequence:
        st = st + match.start() # offset start ind by match start
        end = end - (len(orig_sequence) - match.end()) # offset end index by (old string length - new string length)        
        msg = f'Entity {orig_sequence} transformed to {new_sequence} at [{st}, {end}]'
        warnings.warn(msg)
    
    span = doc.char_span(st, end, label=label, alignment_mode='strict')
    
    if span is None:
        msg = f'Entity [{st}, {end}] does not align with token boundaries.\nOriginal entity was "{orig_sequence}"'
        span = doc.char_span(st, end, label=label, alignment_mode='expand')
        msg += f'\nSetting entity as "{span}" at [{span.start_char}, {span.end_char}]'
        warnings.warn(msg)
    
    return span

def convert(texts: list[dict]):
    '''Constructs the training examples for spacy train.
    Creates a barebones spacy doc with no linguistic annotations (just tokenization) and assigns it a name.
    Processes the sequence labels and adds them to the doc as entities under doc.ents
    '''
    for sample in tqdm(texts):
        text = sample['text']
        doc = nlp.make_doc(text)
        doc._.name = sample['name']
        ents = []
        
        for sequence_label in sample['labels']:
            st, end, label = get_label(sequence_label, default_label_name='keyterm')
            span = refine_span(doc, st, end, label)
            if span:
                ents.append(span)

        if ents:
            doc.ents = ents
        yield doc

if __name__ == "__main__":
    typer.run(main)