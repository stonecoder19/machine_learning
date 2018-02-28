
def is_terminal(token):
    return token[0] != "_"


def expand(grammar, tokens):
   
    for i, token in enumerate(tokens):
        if is_terminal(token): continue

        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]

        return expand(grammar, tokens)

    return tokens

def generate_sentence(grammar):
    return expand(grammar, ["_S"])
