#!/usr/bin/env python
# coding: utf8
"""Example of training spaCy's named entity recognizer, starting off with an
existing model or a blank model.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function
import csv
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import json

# training data


with open('dataset.json') as f:
    data = json.loads("[" +
        f.read().replace("}\n{", "},\n{") +
    "]")

TRAIN_DATA=[]
for each_data in data:
	d={}
	d['entities']=each_data['labels']
	TRAIN_DATA.append((each_data['text'],d))




@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir="train_ner", n_iter=50):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'ml' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.35,  # dropout - make it harder to memorise data
                    losses=losses,
                )
           # print("Losses", losses)

    # test the trained model


    # save model to output directory

    #new_text = "കണ്ണൂര്‍: അതിർത്തിയിൽ നിന്നും ലഹരിക്കടത്ത് തടയുന്നതിനായി പുതുവത്സരാഘോഷങ്ങളുടെ ഭാഗമായി എക്സൈസ് വകുപ്പ് നടപ്പിലാക്കിയ സ്പെഷ്യല്‍ എന്‍ഫോഴ്സ്മെന്റ് ഡ്രൈവില്‍ ജില്ലയില്‍ 644 കേസുകള്‍ രജിസ്റ്റര്‍ ചെയ്തു. എക്സൈസ് മയക്കുമരുന്നുമായി ബന്ധപ്പെട്ട് 57 കേസുകളിലായി 9.547 കി.ഗ്രാം കഞ്ചാവ്, 18 ഗ്രാം ബ്രൗണ്‍ ഷുഗര്‍, 120.51 ഗ്രാം നൈട്രാസെപാം, 4.5 ഗ്രാം മെറ്റാഫറ്റമിന്‍, രണ്ട് കഞ്ചാവ് ചെടി, 0.44 ഗ്രാം എം.ഭി.എം.എയും മൂന്ന് വാഹനങ്ങളും കണ്ടെടുത്തു.,ഇതില്‍ ആഡംബര കാറില്‍ രഹസ്യ അറയില്‍ ഒളിപ്പിച്ച നിലയില്‍ ആറ് കി.ഗ്രാം കഞ്ചാവും, മുംബൈയില്‍ നിന്നും വിദ്യാര്‍ഥികള്‍ക്കിടയില്‍ വില്‍പന നടത്തുന്നതിനായി കൊണ്ടുവന്ന 18 ഗ്രാം ബ്രൗണ്‍ഷുഗറും ഇതര സംസ്ഥാന തൊഴിലാളികളുടെ താമസ സ്ഥലത്തു നിന്നും കണ്ടെത്തിയ രണ്ട് കഞ്ചാവ് ചെടികളും ഉള്‍പ്പെടുന്നു. 135 അബ്കാരി കേസുകളിലായി 2755 ലിറ്റര്‍ വാഷ്, 72 ലിറ്റര്‍ ചാരായം, 42.426 ലിറ്റര്‍ ഇതര സംസ്ഥാന മദ്യം, 429.7 ലിറ്റര്‍ ഇന്ത്യന്‍ നിര്‍മിത വിദേശ മദ്യം, 19.720 ലിറ്റര്‍ ബിയര്‍, ഒരു നാടന്‍ തോക്ക്, ആറ് വാഹനങ്ങള്‍ എന്നിവ പിടിച്ചെടുത്തു.,452 കോട്പ കേസുകളിലായി 827.51 കി.ഗ്രാം പുകയില ഉല്‍പ്പന്നങ്ങള്‍ പിടിച്ചെടുത്ത് 90,400 രൂപ പിഴ ഈടാക്കി. ഇതില്‍ പുതിയതെരു വാടക ക്വാര്‍ട്ടേഴസില്‍ നിന്നും 273 കി.ഗ്രാം പുകയില ഉല്‍പ്പന്നങ്ങളുമായി ഉത്തര്‍പ്രദേശ് സ്വദേശികളെ അറസ്റ്റ് ചെയ്ത് പോലീസിന് കൈമാറി. ജുവനൈല്‍ ജസ്റ്റിസ് ആക്ട് പ്രകാരം സംസ്ഥാനത്ത് ആദ്യമായി നടപടിയെടുത്തതും ഈ കഴിഞ്ഞ ദിവസമാണ്."

    # read news article from crawled news

    file_oneindia = Path("../newscrawl/newscrawl/spiders/news.csv")
    #file_asianet = Path("../newscrawl/newscrawl/spiders/asianet.csv")
    all_news = []


    #with open(file_oneindia,'r') as one_india:
    #	r_oneindia = csv.DictReader(one_india)
    #    r_asianet = csv.DictReader(asianet)
    #	for row in r_oneindia:
    #		all_news.append(row)



   if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        """print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)

        #for new_text in all_news:
        doc = nlp2(new_text['content'])
        #print(doc.ents)

        for ent in doc.ents:
            print(ent.label_, ent.text)
       # print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
            #print("\n\n")

            """

if __name__ == "__main__":
    plac.call(main)
