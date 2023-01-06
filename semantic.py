# Import npl library
import spacy

nlp = spacy.load('en_core_web_md')

tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

"""
Comments:
I can observe from the examples that the similarity model seem to find type relationships
between the words. For instance, higher correlation is found between 'cat' and 'monkey'
words that refer to animals, and by two fruits apple and banana. But also it pick up 
other associations between words; e.g. money and banana. I would make similar association 
which makes me think about how much human like interpretation comes from the model. 

Apple and banana are not fruits that I would highly link to cats, thus the word cat does
not show a correlation to fruits comparable to the monket-banana relationship.
"""

# Additional examples 
print("\nAdditional examples")
tokens = nlp("COVID-19 human computer forecast")

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))

"""
Comments:
For the example above I expected the program to pick up some relationships between COVID-19 and
forecast due to the fact that health authorities try to predict the spread. This assumption turned 
out to be somewhat true but the value was lower than I expected. Also, the words computer and forecast 
shows some similarity due to computers being used to generate forecasts. However, initially I did not 
expect that the words human and compute would yield higher correlation than the previous pair but it
makes sense since humans particularly in this digital era continously interact. 
"""

# Sentence similarity example
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
            "Hello, there is my car",
            "I\'ve lost my car in my car",
            "I\'d like my boat back",
            "I will name my dog Diana"
            ]
model_sentence = nlp(sentence_to_compare)
print("\n")
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ",similarity)

"""
Comments:
This example shows how powerful these tools are. I'm honestly impressed with how the program was able to
find similarities with the second sentence. The sentence involve a person looking at a cat on his/her car,
thus, it makes sense that this scenario could be preceded by the person finding the car. All the relationships 
between sentences yield values above 0.5, I believe this is due to individual word similarity between all of 
them with the sentence to compare (i.e. cat-dog car-car boat-car my-I). Perhaps the amount of similar words 
shared between the two sentence impact de score based on what I can see. 
"""    

"""
This section is dedicate to include notes on the experiment of running the example.py file with with both 
en_core_web_md and en_core_web_sm language models:
* Running the program show how correlated the complaints and recipes areto each other. This means that the 
model is able to accurately conclude that they belong to the same context. 
* While, comparing the complaints to the recipe results in expected lower similarity. 
* The first think I ran into with running the simpler model is that a user warning is displayed in the terminal
explaining that the model does not include word vectors and is limited to context-sensitive tensors.
* Users are able to include own vectors. However, is noticeable by the output evaluation that the model does not performs 
very well as it shows hight variability between the values and fails to identify significant relationships. 
* spaCy documentation validates this as accuracy evaluations shows that en_core_web_md should perform better:
https://spacy.io/models/en
"""