import re
import html
import json
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel


def set_device(gpu_id: int) -> torch.device:
    if gpu_id == -1:
        return torch.device("cpu")
    else:
        return torch.device(
            "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu"
        )


def load_plm(model_path: str = "bert-base-uncased") -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )
    print("Load Model:", model_path)

    model = AutoModel.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
    )
    return tokenizer, model


def load_json(file: str) -> dict:
    with open(file, "r") as f:
        data = json.load(f)
    return data


def clean_text(raw_text: str | list[str] | dict) -> str:
    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r"</?\w+[^>]*>", "", raw)
            raw = re.sub(r'["\n\r]*', "", raw)
            new_raw_text.append(raw.strip())
        cleaned_text = " ".join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r"</?\w+[^>]*>", "", cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."
    if len(cleaned_text) >= 2000:
        cleaned_text = ""
    return cleaned_text


intention_prompt = (
    'After purchasing a {dataset_full_name} item named "{item_title}", the user left a comment '
    'expressing his opinion and personal preferences. The user\'s comment is as follows: \n"{review}" '
    "\nAs we all know, user comments often contain information about both their personal preferences "
    "and the characteristics of the item they interacted with. From this comment, you can infer both "
    "the user's personal preferences and the characteristics of the item. "
    "Please describe your inferred user preferences and item characteristics in the first person and "
    "in the following format:\n\nMy preferences: []\nThe item's characteristics: []\n\n"
    "Note that your inference of the personalized preferences should not include any information about "
    "the title of the item."
)


preference_prompt_1 = (
    "Suppose the user has bought a variety of {dataset_full_name} items, they are: \n{item_titles}. \n"
    "As we all know, these historically purchased items serve as a reflection of the user's personalized preferences. "
    "Please analyze the user's personalized preferences based on the items he has bought and provide a brief third-person "
    "summary of the user's preferences, highlighting the key factors that influence his choice of items. Avoid listing "
    "specific items and do not list multiple examples. "
    "Your analysis should be brief and in the third person."
)

preference_prompt_2 = (
    "Given a chronological list of {dataset_full_name} items that a user has purchased, "
    "we can analyze his long-term and short-term preferences. Long-term preferences are inherent "
    "characteristics of the user, which are reflected in all the items he has interacted with over time. "
    "Short-term preferences are the user's recent preferences, which are reflected in some of the items he has bought more recently. "
    "To determine the user's long-term preferences, please analyze the contents of all the items he has bought. "
    "Look for common features that appear frequently across the user's shopping records. "
    "To determine the user's short-term preferences, focus on the items he has bought most recently. "
    "Identify any new or different features that have emerged in the user's shopping records. "
    "Here is a chronological list of items that the user has bought: \n{item_titles}. \n"
    "Please provide separate analyses for the user's long-term and short-term preferences. "
    "Your answer should be concise and general, without listing specific items. "
    "Your answer should be in the third person and in the following format:\n\n"
    "Long-term preferences: []\nShort-term preferences: []\n\n"
)


# remove 'Magazine', 'Gift', 'Music', 'Kindle'
amazon18_dataset_list = [
    "Appliances",
    "Beauty",
    "Fashion",
    "Software",
    "Luxury",
    "Scientific",
    "Pantry",
    "Instruments",
    "Arts",
    "Games",
    "Office",
    "Garden",
    "Food",
    "Cell",
    "CDs",
    "Automotive",
    "Toys",
    "Pet",
    "Tools",
    "Kindle",
    "Sports",
    "Movies",
    "Electronics",
    "Home",
    "Clothing",
    "Books",
]

amazon18_dataset2fullname = {
    "Beauty": "All_Beauty",
    "Fashion": "AMAZON_FASHION",
    "Appliances": "Appliances",
    "Arts": "Arts_Crafts_and_Sewing",
    "Automotive": "Automotive",
    "Books": "Books",
    "CDs": "CDs_and_Vinyl",
    "Cell": "Cell_Phones_and_Accessories",
    "Clothing": "Clothing_Shoes_and_Jewelry",
    "Music": "Digital_Music",
    "Electronics": "Electronics",
    "Gift": "Gift_Cards",
    "Food": "Grocery_and_Gourmet_Food",
    "Home": "Home_and_Kitchen",
    "Scientific": "Industrial_and_Scientific",
    "Kindle": "Kindle_Store",
    "Luxury": "Luxury_Beauty",
    "Magazine": "Magazine_Subscriptions",
    "Movies": "Movies_and_TV",
    "Instruments": "Musical_Instruments",
    "Office": "Office_Products",
    "Garden": "Patio_Lawn_and_Garden",
    "Pet": "Pet_Supplies",
    "Pantry": "Prime_Pantry",
    "Software": "Software",
    "Sports": "Sports_and_Outdoors",
    "Tools": "Tools_and_Home_Improvement",
    "Toys": "Toys_and_Games",
    "Games": "Video_Games",
}

amazon14_dataset_list = ["Beauty", "Toys", "Sports"]

amazon14_dataset2fullname = {
    "Beauty": "Beauty",
    "Sports": "Sports_and_Outdoors",
    "Toys": "Toys_and_Games",
}

# c1. c2. c3. c4.
amazon_text_feature1 = ["title", "category", "brand"]

# re-order
amazon_text_feature1_ro1 = ["brand", "main_cat", "category", "title"]

# remove
amazon_text_feature1_re1 = ["title"]

amazon_text_feature2 = ["title"]

amazon_text_feature3 = ["description"]

amazon_text_feature4 = ["description", "main_cat", "category", "brand"]

amazon_text_feature5 = ["title", "description"]
