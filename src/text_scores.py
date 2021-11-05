import textstat

def dale_chall_score(text, bulk=False, level=False):
    """
    Function to get the Dale-Chall Readability Score. Formula given by
    
    Parameters
    ----------
    text : list OR str
        Text (or array of texts to measure)
    bulk : bool
        If True, then 'text' should be an array, and we
        process all elements.
        If False, 'text' should be a str and we only 
        process that one text instance.
    level : bool
        If True converts the result into standard school
        level.
        If False simply returns the result in float.
    
    Returns
    -------
    score : float OR list
        Approximation of the Dale-Chall Readability Score,
        either in float or converted to school level.
    """
    if bulk:
        if isinstance(text, str):
            raise TypeError('If \'bulk\' is True, \'text\' needs to be a list.')
        score = 0
        for t in text:
            score += textstat.dale_chall_readability_score(t)
        score /= len(text)
    else:
        score = textstat.dale_chall_readability_score(text)
        
    if level:
        score = convert_dale_chall(score)
        
    return score

def convert_dale_chall(score):
    if score <= 4.9:
        return ['1', '2', '3', '4']
    if score <= 5.9:
        return ['5', '6']
    if score <= 6.9:
        return ['7', '8']
    if score <= 7.9:
        return ['9', '10']
    if score <= 8.9:
        return ['11', '12']
    return ['13', '14', '15']

if __name__ == '__main__':
    test_str1 = 'This seems like an easy sentence. Now I can add a new sentence that I still feel will be easy. One more just for volume.'
    score1 = dale_chall_score(test_str1)
    score_converted1 = dale_chall_score(test_str1, level = True)
    
    test_str2 = 'Furthermore, I desired to test a fairly more advanced sentence. Following that line of thought, I expressed myself as eloquently as possible.'
    score2 = dale_chall_score(test_str2)
    score_converted2 = dale_chall_score(test_str2, level = True)
    
    print(f'The simpler sentence \n\n\t{test_str1}\n\nhas a Dale-Chall score of {score1}, which converted to school levels is {score_converted1}.')
    print(f'\nThe more advanced sentence \n\n\t{test_str2}\n\nhas a Dale-Chall score of {score2}, which converted to school levels is {score_converted2}.')

            
    