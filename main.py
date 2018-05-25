#!/bin/python3

# Imports
from models import CharacterModel

if __name__ == "__main__":
    model = CharacterModel()
    model.prepare_model(print_info=True)
    model.create_model()
    model.pattern_length = 100
    #model.train_model(batch_size=1024, epochs=100)

    model.load_weights('./generated_models/character-03-1.9901.hdf5')

    model.print_model_summary()
    print(model.generate_text_of_length(100, True))

