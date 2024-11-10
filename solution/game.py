from .lib.trained_llm import TrainedLLM
from .lib.word_association import WordAssociations
import json
import re


async def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
    """
    _______________________________________________________
    Parameters:
    words - 1D Array with 16 shuffled words
    strikes - Integer with number of strikes
    isOneAway - Boolean if your previous guess is one word away from the correct answer
    correctGroups - 2D Array with groups previously guessed correctly
    previousGuesses - 2D Array with previous guesses
    error - String with error message (0 if no error)

    Returns:
    guess - 1D Array with 4 words
    endTurn - Boolean if you want to end the puzzle
    _______________________________________________________
    """
    
    state = None
    match len(correctGroups):
        case 0:
            state = 'green'
        case 1:
            state = 'yellow'
        case 2:
            state = 'blue'
        case 3:
            state = 'purple'
        case 4:
            state = 'yellow'
            
    if state == 'purple':
        # Extract the four words that are not in the correctGroups
        return[word for word in words if word not in correctGroups[0] and word not in correctGroups[1] and word not in correctGroups[2]], False   
    
    if state == 'blue':
        trained_llm = TrainedLLM()
        response = trained_llm.run2(words, correctGroups, previousGuesses)
        cleaned_response = re.sub(r'```json|```', '', response.text).strip()
        json_response = json.loads(cleaned_response)
        return json_response, False
    
    if state == 'yellow':
        trained_llm = TrainedLLM()
        print('Yellow')
        response = trained_llm.run(words, correctGroups, previousGuesses)
        cleaned_response = re.sub(r'```json|```', '', response.text).strip()
        json_response = json.loads(cleaned_response)
        return json_response, False
    
    if state == 'green':
        wa = WordAssociations()
        return await wa.get_most_similar_words(words, 1), False
    
    return [], True
