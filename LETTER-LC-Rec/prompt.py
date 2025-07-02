sft_prompt = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    "\n\n### Instruction:\n{instruction}\n\n### Response:{response}"
)


all_prompt: dict[str, list[dict[str, str]]] = {}

# =====================================================
# Task 1 -- Sequential Recommendation -- 17 Prompt
# =====================================================

seqrec_prompt = [
    {
        "instruction": (
            "The user has interacted with items {inters} in chronological order. "
            "Can you predict the next possible item that the user may expect?"
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            "I find the user's historical interactive items: {inters}, and I want to know what next item the user needs. "
            "Can you help me decide?"
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            "Here are the user's historical interactions: {inters}, try to recommend another item to the user. "
            "Note that the historical interactions are arranged in chronological order."
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            "Based on the items that the user has interacted with: {inters}, "
            "can you determine what item would be recommended to him next?"
        ),
        "response": "{item}"
    },
    {
        "instruction": "The user has interacted with the following items in order: {inters}. What else do you think the user need?",
        "response": "{item}"
    },
    {
        "instruction": "Here is the item interaction history of the user: {inters}, what to recommend to the user next?",
        "response": "{item}"
    },
    {
        "instruction": "Which item would the user be likely to interact with next after interacting with items {inters}?",
        "response": "{item}"
    },
    {
        "instruction": "By analyzing the user's historical interactions with items {inters}, what is the next expected interaction item?",
        "response": "{item}"
    },
    {
        "instruction": "After interacting with items {inters}, what is the next item that could be recommended for the user?",
        "response": "{item}"
    },
    {
        "instruction": (
            "Given the user's historical interactive items arranged in chronological order: {inters}, "
            "can you recommend a suitable item for the user?"
        ),
        "response": "{item}"
    },
    {
        "instruction": "Considering the user has interacted with items {inters}. What is the next recommendation for the user?",
        "response": "{item}"
    },
    {
        "instruction": "What is the top recommended item for the user who has previously interacted with items {inters} in order?",
        "response": "{item}"
    },
    {
        "instruction": (
            "The user has interacted with the following items in the past in order: {inters}. "
            "Please predict the next item that the user most desires based on the given interaction records."
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            "Using the user's historical interactions as input data, suggest the next item that the user is highly likely to enjoy. "
            "The historical interactions are provided as follows: {inters}."
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            "You can access the user's historical item interaction records: {inters}. "
            "Now your task is to recommend the next potential item to him, considering his past interactions."
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            "You have observed that the user has interacted with the following items: {inters}, "
            "please recommend a next item that you think would be suitable for the user."
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            "You have obtained the ordered list of user historical interaction items, which is as follows: {inters}. "
            "Using this history as a reference, please select the next item to recommend to the user."
        ),
        "response": "{item}"
    }
]
all_prompt["seqrec"] = seqrec_prompt


# ========================================================
# Task 2 -- Item2Index -- 19 Prompt
# ========================================================
# Remove periods when inputting

item2index_prompt = [
    # Title2Index
    {
        "instruction": 'Which item has the title: "{title}"?',
        "response": "{item}"
    },
    {
        "instruction": 'Which item is assigned the title: "{title}"?',
        "response": "{item}"
    },
    {
        "instruction": 'An item is called "{title}", could you please let me know which item it is?',
        "response": "{item}"
    },
    {
        "instruction": 'Which item is called "{title}"?',
        "response": "{item}"
    },
    {
        "instruction": 'One of the items is named "{title}", can you tell me which item this is?',
        "response": "{item}"
    },
    {
        "instruction": 'What is the item that goes by the title "{title}"?',
        "response": "{item}"
    },
    # Description2Index
    {
        "instruction": 'An item can be described as follows: "{description}". Which item is it describing?',
        "response": "{item}"
    },
    {
        "instruction": 'Can you tell me what item is described as "{description}"?',
        "response": "{item}"
    },
    {
        "instruction": 'Can you provide the item that corresponds to the following description: "{description}"?',
        "response": "{item}"
    },
    {
        "instruction": 'Which item has the following characteristics: "{description}"?',
        "response": "{item}"
    },
    {
        "instruction": 'Which item is characterized by the following description: "{description}"?',
        "response": "{item}"
    },
    {
        "instruction": 'I am curious to know which item can be described as follows: "{description}". Can you tell me?',
        "response": "{item}"
    },
    # Title and Description to Index
    {
        "instruction": 'An item is called "{title}" and described as "{description}", can you tell me which item it is?',
        "response": "{item}"
    },
    {
        "instruction": 'Could you please identify what item is called "{title}" and described as "{description}"?',
        "response": "{item}"
    },
    {
        "instruction": 'Which item is called "{title}" and has the characteristics described below: "{description}"?',
        "response": "{item}"
    },
    {
        "instruction": 'Please show me which item is named "{title}" and its corresponding description is: "{description}".',
        "response": "{item}"
    },
    {
        "instruction": 'Determine which item this is by its title and description. The title is: "{title}", and the description is: "{description}".',
        "response": "{item}"
    },
    {
        "instruction": 'Based on the title: "{title}", and the description: "{description}", answer which item is this?',
        "response": "{item}"
    },
    {
        "instruction": 'Can you identify the item from the provided title: "{title}", and description: "{description}"?',
        "response": "{item}"
    },
]
all_prompt["item2index"] = item2index_prompt


# ========================================================
# Task 3 -- Index2Item --17 Prompt
# ========================================================
# Remove periods when inputting

index2item_prompt = [
    # Index2Title
    {
        "instruction": "What is the title of item {item}?",
        "response": "{title}"
    },
    {
        "instruction": "What title is assigned to item {item}?",
        "response": "{title}"
    },
    {
        "instruction": "Could you please tell me what item {item} is called?",
        "response": "{title}"
    },
    {
        "instruction": "Can you provide the title of item {item}?",
        "response": "{title}"
    },
    {
        "instruction": "What item {item} is referred to as?",
        "response": "{title}"
    },
    {
        "instruction": "Would you mind informing me about the title of item {item}?",
        "response": "{title}"
    },
    # Index2Description
    {
        "instruction": "Please provide a description of item {item}.",
        "response": "{description}"
    },
    {
        "instruction": "Briefly describe item {item}.",
        "response": "{description}"
    },
    {
        "instruction": "Can you share with me the description corresponding to item {item}?",
        "response": "{description}"
    },
    {
        "instruction": "What is the description of item {item}?",
        "response": "{description}"
    },
    {
        "instruction": "How to describe the characteristics of item {item}?",
        "response": "{description}"
    },
    {
        "instruction": "Could you please tell me what item {item} looks like?",
        "response": "{description}"
    },
    # Index to Title and Description
    {
        "instruction": "What is the title and description of item {item}?",
        "response": "{title}\n\n{description}"
    },
    {
        "instruction": "Can you provide the corresponding title and description for item {item}?",
        "response": "{title}\n\n{description}"
    },
    {
        "instruction": "Please tell me what item {item} is called, along with a brief description of it.",
        "response": "{title}\n\n{description}"
    },
    {
        "instruction": "Would you mind informing me about the title of the item {item} and how to describe its characteristics?",
        "response": "{title}\n\n{description}"
    },
    {
        "instruction": "I need to know the title and description of item {item}. Could you help me with that?",
        "response": "{title}\n\n{description}"
    }
]
all_prompt["index2item"] = index2item_prompt


# ========================================================
# Task 4 -- FusionSequentialRec -- Prompt
# ========================================================


fusionseqrec_prompt = [
    {
        "instruction": (
            "The user has sequentially interacted with items {inters}. "
            "Can you recommend the next item for him? Tell me the title of the item？"
        ),
        "response": "{title}"
    },
    {
        "instruction": "Based on the user's historical interactions: {inters}, try to predict the title of the item that the user may need next.",
        "response": "{title}"
    },
    {
        "instruction": (
            "Utilizing the user's past ordered interactions, which include items {inters}, "
            "please recommend the next item you think is suitable for the user and provide its title."
        ),
        "response": "{title}"
    },
    {
        "instruction": (
            "After interacting with items {inters}, what is the most probable item for the user to interact with next? "
            "Kindly provide the item's title."
        ),
        "response": "{title}"
    },
    {
        "instruction": "Please review the user's historical interactions: {inters}, and describe what kind of item he still needs.",
        "response": "{description}"
    },
    {
        "instruction": "Here is the item interaction history of the user: {inters}, please tell me what features he expects from his next item.",
        "response": "{description}"
    },
    {
        "instruction": (
            "By analyzing the user's historical interactions with items {inters}, "
            "can you infer what the user's next interactive item will look like?"
        ),
        "response": "{description}"
    },
    {
        "instruction": (
            "Access the user's historical item interaction records: {inters}. "
            "Your objective is to describe the next potential item for him, taking into account his past interactions."
        ),
        "response": "{description}"
    },
    {
        "instruction": (
            "Given the title sequence of user historical interactive items: {inter_titles}, "
            "can you recommend a suitable next item for the user?"
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            "I possess a user's past interaction history, denoted by the title sequence of interactive items: {inter_titles}, "
            "and I am interested in knowing the user's next most desired item. Can you help me?"
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            "Considering the title sequence of user history interaction items: {inter_titles}. "
            "What is the next recommendation for the user?"
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            "You have obtained the ordered title list of user historical interaction items, as follows: {inter_titles}. "
            "Based on this historical context, kindly choose the subsequent item for user recommendation."
        ),
        "response": "{item}"
    }
]
all_prompt["fusionseqrec"] = fusionseqrec_prompt


# ========================================================
# Task 5 -- ItemSearch -- Prompt
# ========================================================


itemsearch_prompt = [
    {
        "instruction": (
            'Here is the historical interactions of a user: {inters}. '
            'And his personalized preferences are as follows: "{explicit_preference}". '
            "Your task is to recommend an item that is consistent with the user's preference."
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            'The user has interacted with a list of items, which are as follows: {inters}. '
            'Based on these interacted items, the user current intent is as follows '
            '"{user_related_intention}", and your task is to generate an item that matches '
            "the user's current intent."
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            'As a recommender system, you are assisting a user who has recently interacted with the following items: {inters}. '
            'The user expresses a desire to obtain another item with the following characteristics: "{item_related_intention}". '
            'Please recommend an item that meets these criteria.'
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            'Using the user\'s current query: "{query}" and his historical interactions: {inters}, '
            'you can estimate the user\'s preferences "{explicit_preference}". '
            "Please respond to the user's query by selecting an item that best matches his preference and query."
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            'The user needs a new item and searches for: "{query}". '
            "In addition, he has previously interacted with: {inters}. "
            'You can obtain his preference by analyzing his historical interactions: "{explicit_preference}". '
            "Can you recommend an item that best matches the search query and preferences?"
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            'Based on the user\'s historical interactions with the following items: {inters}. '
            'You can infer his preference by observing the historical interactions: "{explicit_preference}". '
            'Now the user wants a new item and searches for: "{query}". '
            'Please select a suitable item that matches his preference and search intent.'
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            'Suppose you are a search engine, now a user searches that: "{query}", '
            "can you select an item to respond to the user's query?"
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            'As a search engine, your task is to answer the user\'s query by generating a related item. '
            'The user\'s query is provided as "{query}". Please provide your generated item as your answer.'
        ),
        "response": "{item}"
    },
    {
        "instruction": (
            'As a recommender system, your task is to recommend an item that is related to the user\'s request, '
            'which is specified as follows: "{query}". Please provide your recommendation.'
        ),
        "response": "{item}"
    },
    {
        "instruction": 'You meet a user\'s query: "{query}". Please respond to this user by selecting an appropriate item.',
        "response": "{item}"
    },
    {
        "instruction": (
            'Your task is to recommend the best item that matches the user\'s query. '
            'Here is the search query of the user: "{query}", tell me the item you recommend.'
        ),
        "response": "{item}"
    }
]

all_prompt["itemsearch"] = itemsearch_prompt


# ========================================================
# Task 6 -- PreferenceObtain -- Prompt
# ========================================================

preferenceobtain_prompt = [
    {
        "instruction": "The user has interacted with items {inters} in chronological order. Please estimate his preferences.",
        "response": "{explicit_preference}"
    },
    {
        "instruction": "Based on the items that the user has interacted with: {inters}, can you infer what preferences he has?",
        "response": "{explicit_preference}"
    },
    {
        "instruction": "Can you provide a summary of the user's preferences based on his historical interactions: {inters}?",
        "response": "{explicit_preference}"
    },
    {
        "instruction": "After interacting with items {inters} in order, what preferences do you think the user has?",
        "response": "{explicit_preference}"
    },
    {
        "instruction": "Here is the item interaction history of the user: {inters}, could you please infer the user's preferences.",
        "response": "{explicit_preference}"
    },
    {
        "instruction": "Based on the user's historical interaction records: {inters}, what are your speculations about his preferences?",
        "response": "{explicit_preference}"
    },
    {
        "instruction": (
            "Given the user's historical interactive items arranged in chronological order: {inters}, "
            "what can be inferred about the preferences of the user?"
        ),
        "response": "{explicit_preference}"
    },
    {
        "instruction": "Can you speculate on the user's preferences based on his historical item interaction records: {inters}?",
        "response": "{explicit_preference}"
    },
    {
        "instruction": "What is the preferences of a user who has previously interacted with items {inters} sequentially?",
        "response": "{explicit_preference}"
    },
    {
        "instruction": (
            "Using the user's historical interactions as input data, summarize the user's preferences. "
            "The historical interactions are provided as follows: {inters}."
        ),
        "response": "{explicit_preference}"
    },
    {
        "instruction": (
            "Utilizing the ordered list of the user's historical interaction items as a reference, "
            "please make an informed estimation of the user's preferences. "
            "The historical interactions are as follows: {inters}."
        ),
        "response": "{explicit_preference}"
    }
]

all_prompt["preferenceobtain"] = preferenceobtain_prompt
