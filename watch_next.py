#Load the nlp library
import spacy

# Store the description for the movie Planet Hulk in a variable
mov_desc = """Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, 
the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. 
Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator."""

def find_similar_movie(description):
    # Load the spaCy English language model
    nlp = spacy.load("en_core_web_md")

    # Read the contents of the movies.txt file into a list
    with open("T38/movies.txt", "r") as f:
        movies = f.readlines()

    # Process the input movie description and all movie descriptions using the spaCy model
    input_description = nlp(description)
    movies_doc = [nlp(d) for d in movies]

    # Calculate the similarity scores for each movie using spaCy's similarity function
    scores = []
    for movie_doc in movies_doc:
        scores.append(input_description.similarity(movie_doc))
    # Return the title of the movie with the highest similarity score
    return movies[scores.index(max(scores))]

# Print the suggested movie to the user
print(f"Tomorrow I suggest you watch the following movie:\n{find_similar_movie(mov_desc)}")