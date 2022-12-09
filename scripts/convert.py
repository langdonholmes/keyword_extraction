"""Convert sequence labels from (st, end) tuples to spaCy v3
.spacy format."""

import srsly
import typer
import warnings
from pathlib import Path

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
    
def convert(text_lines: list):
    for line in text_lines:
        text = line['text']
        doc = nlp.make_doc(text)
        doc._.name = line['name']
        ents = []
        for start, end, label in line['labels']:
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is None:
                msg = f"Document: {line['name']}\nEntity [{start}, {end}, {label}] does not align with token boundaries.\nOriginal entity was '{doc.text[start:end]}'"
                span = doc.char_span(start, end, label=label, alignment_mode='contract')
                msg += f"\nAttempting to set entity as '{span}'"
                warnings.warn(msg)
            ents.append(span)
        doc.ents = ents
        yield doc



if __name__ == "__main__":
    typer.run(main)