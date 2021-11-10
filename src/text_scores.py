import textstat

def dale_chall_score(text, bulk=False, level=False):
    """
    Function to get the Dale-Chall Readability Score.

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
        levels.
        If False simply returns the result in float.

    Returns
    -------
    score : float OR list
        Approximation of the Dale-Chall Readability Score, either
        in float or converted to range of school levels.
    """
    if bulk:
        if isinstance(text, str):
            raise TypeError('If \'bulk\' is True, \'text\' needs to be a list.')
        score = 0
        for t in text:
            score += textstat.dale_chall_readability_score(t)
        score /= len(text)
    else:
        if isinstance(text, list):
            raise TypeError('If \'bulk\' is False, \'text\' needs to be a string.') 
        score = textstat.dale_chall_readability_score(text)

    if level:
        score = convert_dale_chall(score)

    return score

def convert_dale_chall(score):
    """
    Convertion from raw score to school levels.
    Convertion table taken from https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula .
    """
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
