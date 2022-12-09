import typer
import spacy

def main(model: str, default_text: str, output_path: str):
    nlp = spacy.load(model)
    doc = nlp(default_text)
    svg = spacy.displacy.render(doc, style='ent', page=True)
    with open(output_path, 'w', encoding='utf-8') as fn:
        fn.write(svg)
    
if __name__ == "__main__":
    # main('en_core_web_trf', 'The beginning of Frank\'s career at Microsoft began Tuesday, July 3rd.', '../example_vis.html')
    try:
        typer.run(main)
    except SystemExit:
        pass