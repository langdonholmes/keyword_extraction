import typer
from pathlib import Path
import spacy

def create_config(model_name: str, component_to_update: str, output_path: Path):
    nlp = spacy.load(model_name)

    # create a new config as a copy of the loaded pipeline's config
    config = nlp.config.copy()

    # revert most training settings to the current defaults
    default_config = spacy.blank(nlp.lang).config
    config["corpora"] = default_config["corpora"]
    config["training"]["logger"] = default_config["training"]["logger"]
    config["nlp"]["batch_size"] = 256
    
    for score_func in config['training']['score_weights']:
        config['training']['score_weights'][score_func] = None
    
    config['training']['score_weights']['ents_f'] = 1.0
    # copy tokenizer and vocab settings from the base model, which includes
    # lookups (lexeme_norm) and vectors, so they don't need to be copied or
    # initialized separately
    config["initialize"]["before_init"] = {
        "@callbacks": "spacy.copy_from_base_model.v1",
        "tokenizer": model_name,
        "vocab": model_name,
    }
    config["initialize"]["lookups"] = None
    config["initialize"]["vectors"] = None
    config["initialize"]["components"] = {}
    
    # source all components from the loaded pipeline and freeze all except the
    # component to update; replace the listener for the component that is
    # being updated so that it can be updated independently
    config["training"]["frozen_components"] = []
    for pipe_name in nlp.component_names:
        if pipe_name not in ["transformer", component_to_update]:
            config["components"][pipe_name] = {"source": model_name,
                                               "replace_listeners": ["model.tok2vec"]}
            config["training"]["frozen_components"].append(pipe_name)
            
        else:
            config["components"][pipe_name] = {
                "source": model_name,
                # "replace_listeners": ["model.tok2vec"],
            }
    
    
    
    # save the config
    config.to_disk(output_path)

if __name__ == "__main__":
    typer.run(create_config)