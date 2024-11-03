import nltk
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')
from pywsd.lesk import simple_lesk
sent = 'A mouse consists of an object held in ones hand, with one or more buttons'
ambiguous = 'mouse'
answer = simple_lesk(sent, ambiguous, pos='n')
print(answer)
print(answer.definition())
