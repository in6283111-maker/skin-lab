"""Conservative human-skin prompts: one value per exclusive attribute."""
import random
import re

ALIASES = {'girls': 'girl', 'boys': 'boy', 'female': 'girl', 'male': 'boy',
           'woman': 'girl', 'man': 'boy', 'brow hair': 'brown hair',
           'brunette': 'brown hair', 'blonde': 'blonde hair', 'blond': 'blonde hair',
           'blond hair': 'blonde hair', 'grey hair': 'gray hair', 'grey eyes': 'gray eyes'}
HAIR = ['brown hair', 'blonde hair', 'black hair', 'white hair', 'red hair', 'orange hair',
        'pink hair', 'purple hair', 'blue hair', 'green hair', 'gray hair']
EYES = ['blue eyes', 'green eyes', 'brown eyes', 'gray eyes', 'red eyes', 'purple eyes']
THEMES = [(['casual'], ['hoodie', 'sweater', 'jacket', 'shirt'], ['jeans', 'ripped jeans'], ['glasses', 'headphones', 'cap']),
          (['winter'], ['sweater', 'jacket', 'hoodie'], ['jeans'], ['scarf', 'hat']),
          (['summer'], ['shirt', 't-shirt', 'dress'], ['shorts'], ['sunglasses', 'flower crown']),
          (['gamer'], ['hoodie', 'jacket', 'shirt'], ['jeans'], ['headphones', 'glasses']),
          (['medieval', 'fantasy'], ['armor', 'robe', 'dress'], [], ['crown', 'hood']),
          (['anime'], ['hoodie', 'jacket', 'dress'], [], ['glasses', 'headband']),
          (['aesthetic', 'pastel'], ['sweater', 'hoodie', 'dress'], ['jeans'], ['flower crown', 'glasses']),
          (['dark', 'goth', 'emo'], ['hoodie', 'jacket', 'sweater'], ['ripped jeans'], ['choker', 'headphones']),
          ([], ['suit', 'tuxedo'], [], ['glasses', 'sunglasses']),
          (['ninja'], [], [], ['mask', 'headband']),
          (['adventurer'], ['jacket', 'hoodie'], ['jeans'], ['scarf', 'glasses'])]


def normalized(tag):
    tag = re.sub(r'\s+', ' ', tag.strip().casefold().replace('-', ' '))
    return ALIASES.get(tag, tag)


def conflicts(tags):
    attributes = {}
    problems = []
    for raw in tags:
        tag = normalized(raw)
        groups = []
        gender = {'girl': 'girl', 'girly': 'girl', 'egirl': 'girl', 'princess': 'girl',
                  'boy': 'boy', 'eboy': 'boy', 'king': 'boy', 'businessman': 'boy'}
        if tag in gender:
            groups.append(('gender', gender[tag]))
        if tag in HAIR or tag == 'rainbow hair' or tag == 'bald':
            groups.append(('hair color', tag))
        if tag.endswith(' eyes'):
            groups.append(('eye color', tag))
        if tag in ('short hair', 'long hair'):
            groups.append(('hair length', tag))
        if tag in ('summer', 'winter'):
            groups.append(('season', tag))
        for group, value in groups:
            if group in attributes and attributes[group] != value:
                problems.append(f'{group}: {attributes[group]} / {value}')
            attributes[group] = value
    return problems


def make_prompt(labels, rng=None):
    rng = rng or random.SystemRandom()
    allowed = set(labels)
    def pick(options):
        choices = [tag for tag in options if tag in allowed]
        return rng.choice(choices) if choices else None
    gender = pick(['boy', 'girl'])
    tags = [gender, pick(HAIR), pick(EYES)]
    style, tops, bottoms, accessories = rng.choice(THEMES)
    style = pick(style)
    top = pick([tag for tag in tops if gender == 'girl' or tag != 'dress'])
    tags.extend([style, top])
    if top != 'dress':
        tags.append(pick(bottoms))
    if rng.random() < .7:
        tags.append(pick(accessories))
    tags = sorted(set(tag for tag in tags if tag))
    if len(tags) < 3 or conflicts(tags):
        raise ValueError(f'Invalid generated prompt: {tags}')
    return tags
