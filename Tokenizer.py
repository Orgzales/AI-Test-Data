import spacy
import torch
# spacy.cli.download("en_core_web_sm") //if new code is made in a new project


nlp = spacy.load("en_core_web_sm")
names = ["Wii Sports", "Duck Hunt", "Pokemon Emerald", "Wii Play", "Pokemon Diamond"]
vocabulary = set()
for name in names:
    doc = nlp(name)
    for token in doc:
        print(token.text)
        vocabulary.add(token.text)

for name in names:
    doc = nlp(name)
    word_frequency = {}
    for token in doc:
        if token.text in word_frequency:
            word_frequency[token.text] += 1
        else:
            word_frequency[token.text] = 1
    print(word_frequency)
    bag = []
    for w in sorted(vocabulary):
        if w in word_frequency:
            bag.append(word_frequency[w])
        else:
            bag.append(0)
    print(bag)


    t = torch.tensor(bag)
    print(t)

    t2 = torch.tensor([2,25,42])
    t3 = torch.cat([t,t2])
    print(t3)

# TODO:
#     1) load the csv file -results is pandas dataframe
#        -output: Global Sales <- float w/O nan                 |VString
#        -input: Critic score, publisher, user-score,   genre, name, user-count, critic-count
#                 ^float w/ nan | ^Category | ^foat w/ nan |^Category| ^float w/ nan |^float w/ nan
#     2) Tokenize Names ( the loop that builds vocab then counts how many times that word appeared)
#        -result: a list of ints and how many ints in tht list is the size of the n (that is vocab unqie)
#     3) Collect results from all rows (16720, Nv) - Save as tensoor and call t_names
#     4) take user-scoe column
#         4a) remove rows with NaNs - this function is dropNA() in pandas
#         4b) imputation - imputate missing data
#             4b.I) Use the average value to fill in missing values - fillna() in pandas
#             4b.II) Make it zer0
#             4b.III) Train a mdel on other colums to predict missing column (do not use global sales)
#       Results: tensor(16720, 1)
#     5) publisher - convert to one-hot get_dummies()
#       Results: (16720, Np) p = for publisher
#     6) Genre
#       Results: (16720, Ng) g = for Genre
#     7) torch.cat([t_names, t_userscors, ...])
#       Results: (16720, Nv + 1 + 1  + 1 + 1 + Np + Ng) = [Nv + Np + Ng + 4] == input_data for 1-nueron
#     model = nn.Linear(input_data, 1) 




