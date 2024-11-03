# from flask import Flask, request, jsonify
# import spacy
# import nltk
# from nltk.corpus import wordnet as wn
# from flask_cors import CORS
# from pywsd.lesk import simple_lesk


# # Download WordNet data if not already downloaded
# # nltk.download('wordnet')
# # nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('punkt_tab')
# # Initialize Flask app and spaCy model
# app = Flask(__name__)
# CORS(app)
# nlp = spacy.load("en_core_web_sm")

# # Lesk algorithm implementation
# def lesk_algorithm(sentence, target_word):
#     # # Tokenize the sentence using spaCy
#     # doc = nlp(sentence)
    
#     # # Find the target word in the sentence and get its POS
#     # target_token = None
#     # for token in doc:
#     #     if token.text.lower() == target_word.lower():
#     #         target_token = token
#     #         break

#     # if target_token is None:
#     #     return "Target word not found in the sentence."

#     # # Map spaCy POS tags to WordNet POS tags
#     # pos_mapping = {
#     #     "NOUN": wn.NOUN,
#     #     "VERB": wn.VERB,
#     #     "ADJ": wn.ADJ,
#     #     "ADV": wn.ADV
#     # }
#     # wordnet_pos = pos_mapping.get(target_token.pos_, wn.NOUN)  # Default to NOUN

#     # # Get all senses of the target word in WordNet
#     # senses = wn.synsets(target_word, pos=wordnet_pos)
    
#     # if not senses:
#     #     return "No senses found for the target word in WordNet."

#     # # Lesk algorithm: find the sense with the most overlap with the context
#     # best_sense = None
#     # max_overlap = 0
#     # context = set([w.lemma_ for w in doc if w.is_alpha and not w.is_stop])
    
#     # for sense in senses:
#     #     # Get definition and examples for each sense
#     #     definition = set(sense.definition().lower().split())
#     #     examples = set(word for example in sense.examples() for word in example.lower().split())
        
#     #     # Calculate overlap between context and (definition + examples)
#     #     overlap = len(context.intersection(definition.union(examples)))
        
#     #     if overlap > max_overlap:
#     #         max_overlap = overlap
#     #         best_sense = sense

#     # # Return the best sense's definition or a fallback message
#     # if best_sense:
#     #     return best_sense.definition()
#     # else:
#     #     return "Unable to determine the sense."
#     answer = simple_lesk(sentence, target_word, pos='n')
#     return answer.definition()

# # API endpoint to predict word sense
# @app.route('/predict', methods=['POST'])
# def predict_sense():
#     data = request.json
#     sentence = data['sentence']
#     target_word = data['target_word']

#     # Use Lesk algorithm to predict sense
#     sense = lesk_algorithm(sentence, target_word)

#     # Return the sense as the response
#     response = {"sense": sense}
#     return jsonify(response)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify
import spacy
import nltk
from nltk.corpus import wordnet as wn
from flask_cors import CORS
from pywsd.lesk import simple_lesk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Download WordNet data if not already downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Initialize Flask app and spaCy model
app = Flask(__name__)
CORS(app)
nlp = spacy.load("en_core_web_sm")

# Load the BERT model and tokenizer
model_name = "mwesner/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Lesk algorithm implementation
def lesk_algorithm(sentence, target_word):
    answer = simple_lesk(sentence, target_word, pos='n')
    return answer.definition() if answer else "Unable to determine the sense."

def bert_algorithm(sentence, target_word):
    # Prepare input for BERT
    input_text = f"[CLS] {sentence} [SEP] {target_word} [SEP]"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Define a mapping of class indices to meanings for the target word
    sense_mapping = {
    0: "A financial institution that accepts deposits from the public.",  # bank
    1: "The side of a river or stream.",  # bank
    2: "To rely on or trust someone or something.",  # bank
    3: "A place where something is stored or deposited.",  # bank
    4: "A soft cloth for cleaning, especially glass.",  # cloth
    5: "A woven fabric made from natural or synthetic fibers.",  # cloth
    6: "To cover or conceal something.",  # cloth
    7: "An essential part of a clothing ensemble.",  # cloth
    8: "A unit of measurement for length (approximately 0.3048 meters).",  # foot
    9: "The lower part of the leg, below the ankle.",  # foot
    10: "A base or foundation for standing or supporting something.",  # foot
    11: "A manner of walking or moving.",  # foot
    12: "To provide the means for something.",  # fund
    13: "A sum of money set aside for a specific purpose.",  # fund
    14: "A source of supply or support.",  # fund
    15: "A collective term for resources for an organization.",  # fund
    16: "An action of hitting something with a solid object.",  # strike
    17: "To remove or eliminate something.",  # strike
    18: "To work stoppage by employees to demand change.",  # strike
    19: "A movement towards a target in a sport or game.",  # strike
    20: "A light meal, often eaten in the evening.",  # snack
    21: "To consume a small amount of food between meals.",  # snack
    22: "An informal gathering for socializing over food.",  # snack
    23: "A quick bite to satisfy hunger.",  # snack
    24: "A large body of salt water.",  # sea
    25: "A vast expanse of water between continents.",  # sea
    26: "A metaphor for a great number or quantity.",  # sea
    27: "To navigate through water.",  # sea
    28: "A large, winged creature.",  # bird
    29: "An animal that lays eggs and has feathers.",  # bird
    30: "A symbol of freedom and flight.",  # bird
    31: "To travel through the air.",  # bird
    32: "A device for receiving or transmitting sound.",  # phone
    33: "A handheld device for communication.",  # phone
    34: "A technology used for distant conversations.",  # phone
    35: "A call made to someone.",  # phone
    36: "To utilize something for a specific purpose.",  # use
    37: "To consume or exhaust a resource.",  # use
    38: "A practical application of a concept or tool.",  # use
    39: "A behavior or practice considered acceptable in society.",  # custom
    40: "A tradition or habitual practice within a culture.",  # custom
    41: "A practice that is typically followed.",  # custom
    42: "A personal preference or habit.",  # custom
    43: "A perception or belief about something.",  # opinion
    44: "A viewpoint held by an individual or group.",  # opinion
    45: "A judgment or estimation regarding a subject.",  # opinion
    46: "A general feeling or belief about something.",  # opinion
    47: "The act of overseeing or managing a process.",  # control
    48: "The power to influence or direct behavior.",  # control
    49: "To regulate or command something.",  # control
    50: "A device used to operate a machine.",  # control
    51: "To experience a state of satisfaction or fulfillment.",  # enjoy
    52: "To take pleasure in an activity or event.",  # enjoy
    53: "To have a good time during a leisure activity.",  # enjoy
    54: "To participate in something that brings joy.",  # enjoy
    55: "A condition of being well or healthy.",  # health
    56: "The overall condition of a person’s body and mind.",  # health
    57: "A state of physical or mental well-being.",  # health
    58: "The absence of disease or illness.",  # health
    59: "The act of creating or making something.",  # create
    60: "To bring something into existence.",  # create
    61: "To produce a work of art or literature.",  # create
    62: "To cause something to happen.",  # create
    63: "A feeling of satisfaction or pleasure derived from achievement.",  # success
    64: "The accomplishment of a goal or purpose.",  # success
    65: "A favorable or desired outcome.",  # success
    66: "The attainment of wealth or status.",  # success
    67: "An expression of greeting or acknowledgment.",  # hello
    68: "A friendly or polite way to begin a conversation.",  # hello
    69: "A word used to attract attention.",  # hello
    70: "An informal salutation.",  # hello
    71: "An act of communicating information.",  # message
    72: "A written or spoken communication sent to someone.",  # message
    73: "An idea conveyed to another person.",  # message
    74: "A notification or alert about an event or change.",  # message
    75: "A feeling of great pleasure or joy.",  # happiness
    76: "A state of well-being and contentment.",  # happiness
    77: "The experience of positive emotions.",  # happiness
    78: "A condition of living life to the fullest.",  # happiness
    79: "To travel from one location to another.",  # move
    80: "To change position or place.",  # move
    81: "An action taken to achieve a goal.",  # move
    82: "To advance in a particular direction.",  # move
    83: "A physical activity involving coordination and strength.",  # exercise
    84: "An activity designed to improve fitness or health.",  # exercise
    85: "A systematic way of practicing a skill.",  # exercise
    86: "To exert effort for physical improvement.",  # exercise
    87: "A manner of expressing oneself through language.",  # speak
    88: "To communicate verbally with others.",  # speak
    89: "To articulate thoughts or feelings.",  # speak
    90: "To engage in conversation or dialogue.",  # speak
    91: "An arrangement of words that expresses a complete thought.",  # sentence
    92: "A group of words with a subject and predicate.",  # sentence
    93: "A unit of language in written or spoken form.",  # sentence
    94: "A statement made in spoken or written discourse.",  # sentence
    95: "The act of forming or making something.",  # form
    96: "A particular shape or configuration.",  # form
    97: "A document with blank spaces for information.",  # form
    98: "A particular type or kind of something.",  # form
    99: "The act of providing assistance or support.",  # help
    100: "To make it easier for someone to do something.",  # help
}


    # Return the corresponding meaning based on the predicted class
    meaning = sense_mapping.get(predicted_class, "Meaning not found.")
    return f"Predicted meaning for '{target_word}': {meaning}."


# API endpoint to predict word sense
@app.route('/predict', methods=['POST'])
def predict_sense():
    data = request.json
    sentence = data['sentence']
    target_word = data['target_word']
    use_bert = data.get('use_bert', False)  # Include algorithm choice

    # Use the specified algorithm to predict sense
    if use_bert:
        sense = bert_algorithm(sentence, target_word)
    else:
        sense = lesk_algorithm(sentence, target_word)

    # Return the sense as the response
    response = {"sense": sense}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
