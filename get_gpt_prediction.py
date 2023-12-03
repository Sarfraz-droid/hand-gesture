import g4f

messages = [
    {
        "role": "user",
        "content": 'These sequences are results of Hand Sign recognition. Your task is to find what the person is trying to tell you.  I\'ll share the sequences below, your job is to find what the person is saying. You have to try and what the person is saying from the sentence! Make it very short in one sentence. Here is the data: \n Closed fist : Dominance \n Open Palm : Welcoming \n'
    },
    {
        "role": "assistant",
        "content": "Certainly! Please provide the hand sign sequences, and I'll do my best to interpret them."
    }
]

def get_gpt_prediction(sequence=""):
    temp = messages.copy()
    
    temp.append({
        "role": "user",
        "content": sequence
    })
    
    response = g4f.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=temp,
        stream=True,
    )
    
    res = []
    for message in response:
        res.append(message)
        # print(message, flush=True, end='')
    
    return ' '.join(res)

    
