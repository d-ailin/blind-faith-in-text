INSTRUCT_TEMP = {
    'describe': 'Please describe this image.',
    'desc_one_sent': 'Please describe this image in one sentence.',
    'what_is_written': 'nwhat is written in the image? Please only output the answer directly.',
    'replace_noun': '''
        Given an input sentence describing a scene, your task is to:
        1. Locate the noun words in the sentence.
        2. Randomly pick one noun word.
        3. Replace the selected noun word with a new noun word to make a new sentence.

        The new sentence must meet the following three requirements:
        1. The new sentence must be describing a scene that is as different as possible from the original scene.
        2. The new sentence must be fluent and grammatically correct.
        3. The new sentence must make logical sense.

        Here are some examples:
        Original sentence: A man is in a kitchen making pizzas.
        Nouns: ["man", "kitchen", "pizzas"]
        Selected noun: man
        New noun: woman
        New sentence: A woman is in a kitchen making pizzas.

        Original sentence: {caption}
        ''',
    'conflict_answer': '''
    Given an input sentence describing a scene, your task is to modify an original description to make the answer to a specific question has a different answer.
    For example:
    Question: Is the train shiny?
    Original answer: No.
    Original sentence: A train is driving down the tracks in front of a building.
    New answer: Yes
    New sentence: A polished train is driving down the tracks in front of a building.

    Question: What color is the sky?
    Original answer: blue.
    Original sentence: A man standing in a field wearing a hat and jacket.
    New answer: black
    New sentence: A man standing in a field wearing a hat and jacket at night.
    
    Question: {question}
    Original answer: {answer}
    Original sentence: {caption}
    ''',
    'conflict_answer_wo_ref': '''
    Your task is to generate a new description to make the answer to a specific question has a different answer.
    For example:
    Question: Is the train shiny?
    Original answer: No.
    New answer: Yes
    New sentence: The train gleamed under the sunlight.
    
    Question: What color is the sky?
    Original answer: blue.
    New answer: black
    New sentence: The sky was pitch black, as if night had fallen early.
    
    Question: {question}
    Original answer: {answer}
    ''',
    'noisy_but_same': '''
    Given an input sentence describing a scene, a question and an answer, your task is to:
    1. Modify the original sentence by adding or replacing nouns or adjectives to make the new sentence.
    2. Make sure the new sentence still have the same answer to the question.
    2. Make sure the new sentence is fluent and grammatically correct.
    3. Make sure the new sentence makes logical sense.
    
    Here are some examples:
    Question: Is there a man in the kitchen?
    Answer: Yes.
    Original sentence: There is a man standing in a kitchen making pizzas.
    Operation: Replace
    Change: making pizzas -> washing dishes
    New sentence: There is a man standing in a kitchen washing dishes.

    Question: Is the train shiny?
    Answer: Yes.
    Original sentence: A shiny train is driving down the tracks in front of a building.
    Operation: Add
    Change: Adding "beside a river"
    New sentence: A shiny train is driving down the tracks in front of a building beside a river.
    
    Question: {question}
    Answer: {answer}
    Original sentence: {caption}
    ''',
     'noisy_but_same_1': '''
    Given an input sentence describing a scene, a question and an answer, your task is to:
    1. Modify the original sentence by adding new words to make the new sentence.
    2. Make sure the new sentence still have the same answer to the question.
    2. Make sure the new sentence is fluent and grammatically correct.
    3. Make sure the new sentence makes logical sense.
    
    Here are some examples:
    Question: Is there a man in the kitchen?
    Answer: Yes.
    Original sentence: There is a man standing in a kitchen making pizzas.
    Reasoning: As the answer is Yes, the new sentence should still have a man in the kitchen but only add some new words.
    New sentence: There is a man standing in a kitchen making pizzas with a cat watching him.

    Question: Is the train shiny?
    Answer: Yes.
    Original sentence: A shiny train is driving down the tracks in front of a building.
    Reasoning: As the answer is Yes, the new sentence should still have a shiny train but only add some new words.
    New sentence: A shiny train is driving down the tracks in front of a building with a rainbow in the sky.
    
    Question: {question}
    Answer: {answer}
    Original sentence: {caption}
    ''',
    'supp_contradict_stat': '''
    Given the image, the question and the answer, your task is to:
    1. Generate an accurate <Description 1> which can be used for answering the question correctly without using the image.
    2. Generate a wrong description <Description 2> which can be used for answering the question with a completely wrong answer <answer 2> without using the image.
    3. Make sure both descriptions are sound and concise.
    4. The wrong description's sentence structure should be similar to the correct description.
    
    Here are the questions and answers:
    Question: {question}
    Answer: {answer}
    
    Please output the two statements in this format:
    Description 1: <Description 1>
    Description 2: <Description 2>
    Answer 2: <answer 2>
    ''',
    
}