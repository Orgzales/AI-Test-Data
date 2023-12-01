from transformers import GPT2Tokenizer, AutoModelForCausalLM
import torch
import numpy as np
import sys



tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

msg = """
Review: I hate this.
Sentiment: Negative

Review: I love this.
Sentiment: Positive

Review: I'm not having any success.
Sentiment: Negative
 
Review: I need this in my live.
Sentiment: Positive

Review: This is the worst model ever.
Sentiment: Negative
"""

def User_input(input_msg):

    global msg
    msg += "\nReview: " + str(input_msg) + "\nSentiment:"


    inputs = tokenizer([msg], return_tensors="pt")
    outputs = model.forward(**inputs, labels=inputs.input_ids)

    #grab the last logit
    logits = outputs.logits.detach()
    logits = logits[0, -1, :] #0 = first input of token, -1 is getting the last one and : means getting the rest

    best = torch.argsort(logits, descending=True)
    b = tokenizer.decode(best[0].item())
    msg += tokenizer.decode(torch.argmax(logits).item())
    return msg

print("Your input: ")
user = input("")
print(User_input(user))

# i = 0
# while i < 1:
#     i += 1
#     # print(msg)
#     inputs = tokenizer([msg], return_tensors="pt")
#     outputs = model.forward(**inputs, labels=inputs.input_ids)
#
#     #grab the last logit
#     logits = outputs.logits.detach()
#     logits = logits[0, -1, :] #0 = first input of token, -1 is getting the last one and : means getting the rest
#
#     best = torch.argsort(logits, descending=True)
#     b = tokenizer.decode(best[0].item())
#     msg += tokenizer.decode(torch.argmax(logits).item())
#     print(msg)


# print(b)


#Make a sentiment identificating mode


# sys.exit()








