class Prompt():
    def __init__(self):
        self.system_msg = [
            [
{"role": "system", "content": "You are a helpful assistant on movie recommendation. "}
            ],
            [
{"role": "system", "content": "You are a helpful assistant on CDs and Vinyl recommendation."}
            ],
            [
{"role": "system", "content": "You are a helpful assistant on office products recommendation."}
            ]
        ]
        self.user_prompt = [
            [
{"role": "system", "content": "You are a helpful assistant on recommendation. "},
{"role": "user", "content": """Now I will give you:
1. The Original Profile of a user, describing the user's current preferences and dislikes; 
2. The profile of an item interacted by the user, containing the item's name, category and some descriptions;
3. The rating of the item scored by user.
Now you are considering to generate New User Profile of the user, based on Original User Profile, Item Profile and Rating information.
To do this, please follow the steps below:
1. Review the given Original User Profile and Item Profile, remember the profile terms that already exist. You should AVOID generating such terms repeatedly.
2. Extract the currently derived preferences and dislikes of the user from the user's Original Profile.
3. Give a Thought according to user's preferences and dislikes, item profile and the user's rating, try to explain what main features make the user give the item a low/high rating. Specially, you should try to find out the relationship between the rating and the features in Item Profile. The Thought MUST NOT contain information not related to the current asked item, and MUST NOT have unfounded assumptions about the item.
4. Generate the New User Profile based on the information derived in the Thought. The New User Profile should ONLY contain newly inferred features, AVOID generating features that you already know about the user.

Important note:
1. Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. You should pay more attention to the given Item Profile first, try to make speculations about what features in the Item Profile makes the user like / not like the given item. You may also use the Original User Profile as a reference, to find out how the terms in the Original User Profile can align with the given Item Profile. Making unfounded assumptions about the given item, or using terms in the Original User Profile as the complement of the item's profile are STRICTLY PROHIBITED, unless you are really sure you know other information about this item.
3. Your output should ONLY contain newly inferred features, and be specific and personalized, not too general, highlighting detailed and distinguishable feature of the user. DO NOT fabricate anything that cannot be inferred from input.
4. You may use some adverbs to describe a user when you are not completely sure about the preference of a user. For example, you may use phrases like "may like", "like to some extent" or "moderately like" when describing a user's mid-level fondness of a genre.
5. When rating is 1, 2 or 3, you need to consider it as a low score, showing that the user do not like this item. When rating is 4 or 5, it is a high score, showing that the user like this item.

An example would be:
"""},
{"role": "assistant", "content": "Sure, I will generate the user's new profile after I see the rating and the item's attribute according to the requirements. "}
],
            [
# {"role": "system", "content": "You are a helpful assistant on movie recommendation. "},
{"role": "user", "content": """Now I will give you:
1. The Original Profile of a user, describing the user's current preferences and dislikes; 
2. The profile of an item interacted by the user, containing the item's name, category and some descriptions;
3. The rating of the item scored by user. Rating less or equal than 3 is considered as a negative attitude towards the item.
Now you are considering to generate New User Profile of the user, based on Original User Profile, Item Profile and Rating information.
To do this, please follow the steps below:
1. Review the given Original User Profile and Item Profile, remember the profile terms that already exist. You should AVOID generating such terms repeatedly.
2. Extract the currently derived preferences and dislikes of the user from the user's Original Profile.
3. Give a Thought according to user's preferences and dislikes, item profile and the user's rating, try to explain what main features make the user give the item a low/high rating. Specially, you should try to find out the relationship between the rating and the features in Item Profile. The Thought MUST NOT contain information not related to the current asked item, and MUST NOT have unfounded assumptions about the item.
4. Generate the New User Profile based on the information derived in the Thought. The New User Profile should ONLY contain newly inferred features, AVOID generating features that you already know about the user.

Important note:
1. Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. You should pay more attention to the given Item Profile first, try to make speculations about what features in the Item Profile makes the user like / not like the given item. You may also use the Original User Profile as a reference, to find out how the terms in the Original User Profile can align with the given Item Profile. Making unfounded assumptions about the given item, or using terms in the Original User Profile as the complement of the item's profile are STRICTLY PROHIBITED. You are welcomed to use other knowledge inside your memory about this item, if you are really sure it is correct and accurate.
3. Your output should ONLY contain newly inferred features, and be specific and personalized, not too general, highlighting detailed and distinguishable feature of the user. DO NOT fabricate anything that cannot be inferred from input.
4. You may use some adverbs to describe a user when you are not completely sure about the preference of a user. For example, you may use phrases like "may like", "like to some extent" or "moderately like" when describing a user's mid-level fondness of a genre.
5. When rating is 1, 2 or 3, you MUST consider it as a LOW score, showing that the user do not like this item. When rating is 4 or 5, it is a high score, showing that the user like this item.

For a specific scenario, an example on movie recommendation would be:
"""},
{"role": "assistant", "content": "Sure, I will generate the user's new profile after I see the rating and the item's attribute according to the requirements. "}
],
            [
# {"role": "system", "content": "You are a helpful assistant on movie recommendation. "},
{"role": "user", "content": """Now I will give you:
1. The Original Profile of a user, describing the user's current preferences and dislikes;
2. An item list, consisting of one item or several items interacted by the user. The given items in the item list are arranged in chronological order, from recent to far, and the items given to you this round is interacted earlier than the items in the last round. In the item list, each item includes a profile containing the item's name, category and some descriptions, and the rating of the item scored by the user. Rating less or equal than 3.0 is considered as a negative attitude towards the item.
Now you are considering to generate New User Profile of the user, based on Original User Profile, Item Profiles and Ratings in the item list.
To do this, please follow the steps below:
1. Review the given Original User Profile and Item Profiles, remember the profile terms that already exist. You should AVOID generating such terms repeatedly.
2. Extract the currently derived preferences and dislikes of the user from the user's Original Profile.
3. Give a Thought according to user's preferences and dislikes, item profiles and the user's ratings, try to explain what main features make the user give the items low/high ratings. Specially, you should try to find out the relationship between the ratings and the features in Item Profiles, and try to extract the common features between the given items that the user likes/dislikes. You MUST examine the item list comprehensively, not omitting any item. The Thought MUST NOT contain information not related to the current asked items, and MUST NOT have unfounded assumptions about the items.
4. Generate the New User Profile based on the information derived in the Thought. The New User Profile should ONLY contain newly inferred features, AVOID generating features that you already know about the user.

Important note:
1. Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. You should pay more attention to the given Item Profiles first, try to make speculations about what features in the Item Profiles make the user like/dislike the given items. You may also use the Original User Profile as a reference, to find out how the terms in the Original User Profile can align with the given Item Profiles. Making unfounded assumptions about the given items, or using terms in the Original User Profile as the complement of the item's profile are STRICTLY PROHIBITED. You are welcomed to use other knowledge inside your memory about the items, if you are really sure it is correct and accurate.
3. Your output should ONLY contain newly inferred features, and be specific and personalized, not too general, highlighting detailed and distinguishable feature of the user. DO NOT fabricate anything that cannot be inferred from input.
4. You may use some adverbs to describe a user when you are not completely sure about the preference of a user. For example, you may use phrases like "may like", "like to some extent" or "moderately like" when describing a user's mid-level fondness of a genre.
5. In order to ensure the comprehensiveness of the generated new user profile, the number of generated terms MUST be at least the length of the item list, e.g. if you are given 5 items, your generated New User Profile MUST contain at least 5 terms. You need to fully examine the given features of the items.
6. When rating of an item is lower or equal than 3.0, you MUST consider it as a LOW score, showing that the user do not like this item. When rating is higher or equal than 3.5, it is a high score, showing that the user like this item.

For a specific scenario, an example on movie recommendation would be:
"""},
{"role": "user", "content": """Above is the item list given to you. Please be aware that you are given a task of generating New User Profile of the user, based on Original User Profile, Item Profiles and Ratings in the item list. The given items in the item list are arranged in chronological order, from recent to far, and the items given to you this round is interacted earlier than the items in the last round.
You MUST strictly follow the task instructions and give the output in correct format. You are FORBIDDEN to give freestyled recommendations and description summarizations, because they are NOT YOUR TASK.
Now we emphasize the instructions of the task to you again:
To do this, please follow the steps below:
1. Review the given Original User Profile and Item Profiles, remember the profile terms that already exist. You should AVOID generating such terms repeatedly.
2. Extract the currently derived preferences and dislikes of the user from the user's Original Profile.
3. Give a Thought according to user's preferences and dislikes, item profiles and the user's ratings, try to explain what main features make the user give the items low/high ratings. Specially, you should try to find out the relationship between the ratings and the features in Item Profiles, and try to extract the common features between the given items that the user likes/dislikes. You MUST examine the item list comprehensively, not omitting any item. The Thought MUST NOT contain information not related to the current asked items, and MUST NOT have unfounded assumptions about the items.
4. Generate the New User Profile based on the information derived in the Thought. The New User Profile should ONLY contain newly inferred features, AVOID generating features that you already know about the user.

Important note:
1. Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. You should pay more attention to the given Item Profiles first, try to make speculations about what features in the Item Profiles make the user like/dislike the given items. You may also use the Original User Profile as a reference, to find out how the terms in the Original User Profile can align with the given Item Profiles. Making unfounded assumptions about the given items, or using terms in the Original User Profile as the complement of the item's profile are STRICTLY PROHIBITED. You are welcomed to use other knowledge inside your memory about the items, if you are really sure it is correct and accurate.
3. Your output should ONLY contain newly inferred features, and be specific and personalized, not too general, highlighting detailed and distinguishable feature of the user. DO NOT fabricate anything that cannot be inferred from input.
4. You may use some adverbs to describe a user when you are not completely sure about the preference of a user. For example, you may use phrases like "may like", "like to some extent" or "moderately like" when describing a user's mid-level fondness of a genre.
5. In order to ensure the comprehensiveness of the generated new user profile, the number of generated terms MUST be at least the length of the item list, e.g. if you are given 5 items, your generated New User Profile MUST contain at least 5 terms. You need to fully examine the given features of the items.
6. When rating of an item is lower or equal than 3.0, you MUST consider it as a LOW score, showing that the user do not like this item. When rating is higher or equal than 3.5, it is a high score, showing that the user like this item.

No matter how you can fully understand the item list, your output should strictly follow the given format. Otherwise you will be punished.
"""},
{"role": "assistant", "content": "Sure, I will generate the user's new profile after I see the rating and the items' attribute according to the requirements. "}
],
            
[
# {"role": "system", "content": "You are a helpful assistant on movie recommendation. "},
{"role": "user", "content": """Now I will give you:
1. The Original Profile of a user, describing the user's current preferences and dislikes;
2. An item list, consisting of one item or several items interacted by the user. The given items in the item list are arranged in chronological order, from recent to far, and the items given to you this round is interacted earlier than the items in the last round.
In the item list, each item includes a profile containing the item's name, category and some descriptions, the rating of the item scored by the user, and the collaborative score derived from the collaborative filtering infomation between users and items. Rating less or equal than 3.5 is considered as a negative attitude towards the item. The collaborative score range is [-1.0, 1.0], which is comparable between items, higher score indicates more fondness on this item, vice versa.
Now you are considering to generate New User Profile of the user, based on the item list.
To do this, please follow the steps below:
1. Review the given Original User Profile and Item Profiles, remember the profile terms that already exist. You should AVOID generating such terms repeatedly.
2. Extract the currently derived preferences and dislikes of the user from the user's Original Profile.
3. Compare the ratings and collaborative scores between the items, as they reflect the relative fondness of the items by the user. Higher rating and score means that the user likes this item and its genres more than others, vice versa. You should try your best to differentiate and reflect the degrees of preferences of the user with different types of items in the New User Profile.
4. Give a Thought according to user's original preferences and dislikes, item profiles, the user's ratings and the collaborative scores, containing the above extractions and comparations, try to explain what main features make the user give the items low/high ratings, and result in these collaborative scores. Specially, you should find out the relationship between the ratings, scores and the features in Item Profiles, and extract the common and specific features between the given items that the user likes/dislikes. You MUST examine the item list comprehensively, not omitting any item. The Thought MUST NOT contain information not related to the current asked items, and MUST NOT have unfounded assumptions about the items. 
5. Generate the New User Profile based on the information derived in the Thought. The New User Profile should ONLY contain newly inferred features, AVOID generating features that you already know about the user.

Important note:
1. Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. You should pay more attention to the given Item Profiles first, try to make speculations about what features in the Item Profiles make the user like/dislike the given items. You may also use the Original User Profile as a reference, to find out how the terms in the Original User Profile can align with the given Item Profiles. Making unfounded assumptions about the given items, or using terms in the Original User Profile as the complement of the item's profile are STRICTLY PROHIBITED. You are welcomed to use other knowledge inside your memory about the items, if you are really sure it is correct and accurate.
3. Your output should ONLY contain newly inferred features, and be specific and personalized, not too general, highlighting detailed and distinguishable feature of the user. DO NOT fabricate anything that cannot be inferred from input.
4. You may use some adverbs to describe a user when you are not completely sure about the preference of a user. For example, you may use phrases like "may like", "like to some extent" or "moderately like" when describing a user's mid-level fondness of a genre.
5. In order to ensure the comprehensiveness of the generated new user profile, the number of generated terms MUST be at least the length of the item list, e.g. if you are given 5 items, your generated New User Profile MUST contain at least 5 terms. You need to fully examine the given features of the items.
6. When rating of an item is lower or equal than 3.5, you MUST consider it as a LOW score, showing that the user do not like this item. When rating is higher or equal than 4.0, it is a high score, showing that the user like this item.

For a specific scenario, an example on movie recommendation would be:
"""},
# {"role": "user", "content": """Above is the item list given to you. Please be aware that you are given a task of generating New User Profile of the user, based on Original User Profile, Item Profiles Ratings and Collaborative Scores in the item list. The given items in the item list are arranged in chronological order, from recent to far, and the items given to you this round is interacted earlier than the items in the last round.
# You MUST strictly follow the task instructions and give the output in correct format. You are FORBIDDEN to give freestyled recommendations and description summarizations, because they are NOT YOUR TASK.
# Now we emphasize the instructions of the task to you again:
# To do this, please follow the steps below:
# 1. Review the given Original User Profile and Item Profiles, remember the profile terms that already exist. You should AVOID generating such terms repeatedly.
# 2. Extract the currently derived preferences and dislikes of the user from the user's Original Profile.
# 3. Compare the ratings and collaborative scores between the items, as they reflect the relative fondness of the items by the user. Higher rating and score means that the user likes this item and its genres more than others, vice versa. You should try your best to differentiate and reflect the degrees of preferences of the user with different types of items in the New User Profile.
# 4. Give a Thought according to user's original preferences and dislikes, item profiles, the user's ratings and the collaborative scores, containing the above extractions and comparations, try to explain what main features make the user give the items low/high ratings, and result in these collaborative scores. Specially, you should find out the relationship between the ratings, scores and the features in Item Profiles, and extract the common and specific features between the given items that the user likes/dislikes. You MUST examine the item list comprehensively, not omitting any item. The Thought MUST NOT contain information not related to the current asked items, and MUST NOT have unfounded assumptions about the items. 
# 5. Generate the New User Profile based on the information derived in the Thought. The New User Profile should ONLY contain newly inferred features, AVOID generating features that you already know about the user.

# Important note:
# 1. Your output should in the format:
# Thought: Plain text.
# New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
# 2. You should pay more attention to the given Item Profiles first, try to make speculations about what features in the Item Profiles make the user like/dislike the given items. You may also use the Original User Profile as a reference, to find out how the terms in the Original User Profile can align with the given Item Profiles. Making unfounded assumptions about the given items, or using terms in the Original User Profile as the complement of the item's profile are STRICTLY PROHIBITED. You are welcomed to use other knowledge inside your memory about the items, if you are really sure it is correct and accurate.
# 3. Your output should ONLY contain newly inferred features, and be specific and personalized, not too general, highlighting detailed and distinguishable feature of the user. DO NOT fabricate anything that cannot be inferred from input.
# 4. You may use some adverbs to describe a user when you are not completely sure about the preference of a user. For example, you may use phrases like "may like", "like to some extent" or "moderately like" when describing a user's mid-level fondness of a genre.
# 5. In order to ensure the comprehensiveness of the generated new user profile, the number of generated terms MUST be at least the length of the item list, e.g. if you are given 5 items, your generated New User Profile MUST contain at least 5 terms. You need to fully examine the given features of the items.
# 6. When rating of an item is lower or equal than 3.0, you MUST consider it as a LOW score, showing that the user do not like this item. When rating is higher or equal than 3.5, it is a high score, showing that the user like this item.

# No matter how you can fully understand the item list, your output should strictly follow the given format. Otherwise you will be punished.
# """},
{"role": "user", "content": """Above is the item list given to you. Please be aware that you are given a task of generating New User Profile of the user, based on Original User Profile, Item Profiles Ratings and Collaborative Scores in the item list.
You MUST strictly follow the task instructions and give the output in correct format. You are FORBIDDEN to give freestyled recommendations and description summarizations, because they are NOT YOUR TASK.
Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
No matter how you can fully understand the item list, your output should strictly follow the given format. Otherwise you will be punished.
"""},
{"role": "assistant", "content": "Sure, I will generate the user's new profile after I see the rating and the items' attribute according to the requirements. "}
],

[
{"role": "user", "content": """Now I will give you:
1. The Original Profile of a user, describing the user's current preferences and dislikes;
2. An item list, consisting of one item or several items interacted by the user. The given items in the item list are arranged in chronological order, from recent to far, and the items given to you this round is interacted earlier than the items in the last round.
In the item list, each item includes a profile containing the item's name, category and some descriptions, the rating of the item scored by the user, and the collaborative hedonic scale derived from the collaborative filtering infomation between users and items. Rating less or equal than 3.5 is considered as a negative attitude towards the item. The collaborative hedonic scale is comparable between items, indicating the fondness level of each item.
Now you are considering to generate New User Profile of the user, based on the item list.
To do this, please follow the steps below:
1. Review the given Original User Profile and Item Profiles, remember the profile terms that already exist. You should AVOID generating such terms repeatedly.
2. Extract the currently derived preferences and dislikes of the user from the user's Original Profile.
3. Compare the ratings and collaborative hedonic scales between the items, as they reflect the relative fondness of the items by the user. Higher rating and score means that the user likes this item and its genres more than others, vice versa. You should try your best to differentiate and reflect the degrees of preferences of the user with different types of items in the New User Profile.
4. Give a Thought according to user's original preferences and dislikes, item profiles, the user's ratings and the collaborative hedonic scales, containing the above extractions and comparations, try to explain what main features make the user give the items low/high ratings, and result in these collaborative hedonic scales. Specially, you should find out the relationship between the ratings, scores and the features in Item Profiles, and extract the common and specific features between the given items that the user likes/dislikes. You MUST examine the item list comprehensively, not omitting any item. The Thought MUST NOT contain information not related to the current asked items, and MUST NOT have unfounded assumptions about the items. 
5. Generate the New User Profile based on the information derived in the Thought. The New User Profile should ONLY contain newly inferred features, AVOID generating features that you already know about the user.

Important note:
1. Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. You should pay more attention to the given Item Profiles first, try to make speculations about what features in the Item Profiles make the user like/dislike the given items. You may also use the Original User Profile as a reference, to find out how the terms in the Original User Profile can align with the given Item Profiles. Making unfounded assumptions about the given items, or using terms in the Original User Profile as the complement of the item's profile are STRICTLY PROHIBITED. You are welcomed to use other knowledge inside your memory about the items, if you are really sure it is correct and accurate.
3. Your output should ONLY contain newly inferred features, and be specific and personalized, not too general, highlighting detailed and distinguishable feature of the user. DO NOT fabricate anything that cannot be inferred from input.
4. You may use some adverbs to describe a user when you are not completely sure about the preference of a user. For example, you may use phrases like "may like", "like to some extent" or "moderately like" when describing a user's mid-level fondness of a genre.
5. In order to ensure the comprehensiveness of the generated new user profile, the number of generated terms MUST be at least the length of the item list, e.g. if you are given 5 items, your generated New User Profile MUST contain at least 5 terms. You need to fully examine the given features of the items.
6. When rating of an item is lower or equal than 3.5, you MUST consider it as a LOW score, showing that the user do not like this item. When rating is higher or equal than 4.0, it is a high score, showing that the user like this item.

For a specific scenario, an example on movie recommendation would be:
"""},
{"role": "user", "content": """Above is the item list given to you. Please be aware that you are given a task of generating New User Profile of the user, based on Original User Profile, Item Profiles Ratings and Collaborative Hedonic Scales in the item list.
You MUST strictly follow the task instructions and give the output in correct format. You are FORBIDDEN to give freestyled recommendations and description summarizations, because they are NOT YOUR TASK.
Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
No matter how you can fully understand the item list, your output should strictly follow the given format. Otherwise you will be punished.
"""},
{"role": "assistant", "content": "Sure, I will generate the user's new profile after I see the rating and the items' attribute according to the requirements. "}
],

[
{"role": "user", "content": """Now I will give you:
1. The Original Profile of a user, describing the user's current preferences and dislikes;
2. An item list, consisting of one item or several items interacted by the user. The given items in the item list are arranged in chronological order, from recent to far, and the items given to you this round is interacted earlier than the items in the last round.
In the item list, each item includes a profile containing the item's name, category and some descriptions, and the rating of the item scored by the user. Rating less or equal than 3.5 is considered as a negative attitude towards the item.
Now you are considering to generate New User Profile of the user, based on the item list.
To do this, please follow the steps below:
1. Review the given Original User Profile and Item Profiles, remember the profile terms that already exist. You should AVOID generating such terms repeatedly.
2. Extract the currently derived preferences and dislikes of the user from the user's Original Profile.
3. Compare the ratings between the items, as they reflect the relative fondness of the items by the user. Higher rating means that the user likes this item and its genres more than others, vice versa. You should try your best to differentiate and reflect the degrees of preferences of the user with different types of items in the New User Profile.
4. Give a Thought according to user's original preferences and dislikes, item profiles and the user's ratings, containing the above extractions and comparations, try to explain what main features make the user give the items low/high ratings. Specially, you should find out the relationship between the ratings and the features in Item Profiles, and extract the common and specific features between the given items that the user likes/dislikes. You MUST examine the item list comprehensively, not omitting any item. The Thought MUST NOT contain information not related to the current asked items, and MUST NOT have unfounded assumptions about the items. 
5. Generate the New User Profile based on the information derived in the Thought. The New User Profile should ONLY contain newly inferred features, AVOID generating features that you already know about the user.

Important note:
1. Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. You are FORBIDDEN to directly use genres' names to represent user's preferences!!!
3. You should pay more attention to the given Item Profiles first, try to make speculations about what features in the Item Profiles make the user like/dislike the given items. You may also use the Original User Profile as a reference, to find out how the terms in the Original User Profile can align with the given Item Profiles. Making unfounded assumptions about the given items, or using terms in the Original User Profile as the complement of the item's profile are STRICTLY PROHIBITED. You are welcomed to use other knowledge inside your memory about the items, if you are really sure it is correct and accurate.
4. Your output should ONLY contain newly inferred features, and be specific and personalized, not too general, highlighting detailed and distinguishable feature of the user. DO NOT fabricate anything that cannot be inferred from input.
5. You may use some adverbs to describe a user when you are not completely sure about the preference of a user. For example, you may use phrases like "may like", "like to some extent" or "moderately like" when describing a user's mid-level fondness of a genre.
6. In order to ensure the comprehensiveness of the generated new user profile, the number of generated terms MUST be at least the length of the item list, e.g. if you are given 5 items, your generated New User Profile MUST contain at least 5 terms. You need to fully examine the given features of the items.
7. When rating of an item is lower or equal than 3.5, you MUST consider it as a LOW score, showing that the user do not like this item. When rating is higher or equal than 4.0, it is a high score, showing that the user like this item.

For a specific scenario, an example on movie recommendation would be:
"""},
{"role": "user", "content": """To summarize, please be aware that you are given a task of generating New User Profile of the user, based on Original User Profile, Item Profiles and Ratings in the item list.
You MUST strictly follow the task instructions and give the output in correct format. You are FORBIDDEN to give freestyled recommendations and description summarizations, because they are NOT YOUR TASK.
Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
No matter how you can fully understand the item list, your output should strictly follow the given format. Otherwise you will be punished.
"""},
{"role": "assistant", "content": "Sure, I will generate the user's new profile after I see the rating and the items' attribute according to the requirements. "}
]



#             [
# {"role": "user", "content": """Please generate New User Profile based on:
# (1) The Original Profile of a user, describing the user's current preferences and dislikes;
# (2) An item list, consisting of several items interacted by the user. The items are arranged in chronological order, from recent to far, the items given to you this round is earlier than the items given before. Each item includes a profile containing the item's name, category and some descriptions, and the rating of the item scored by the user.
# You should complete the generation following these steps:
# 1. Review the terms in given Original User Profile and Item Profiles, as well as the ratings, analyze the combinations of them, considering any patterns of preferences and dislikes.
# 2. Extract the currently derived preferences and dislikes of the user from the user's Original Profile.
# 3. Deduce the user's interactive intent within each combination, give a Thought according to the analyses of user's preferences and dislikes, based on the given profiles and ratings. Specifically, find out the relationship between the low/high ratings and the features in Item Profiles, and extract the common features that the user likes/dislikes. Also, determine the most representative intents (current user preference/dislikes) from the inferred ones.
# 4. Generate the New User Profile based on the selected intents and the information derived in the Thought.

# Important note:
# 1. Your output should in the format:
# Thought: Plain text
# New User Profile: [List of terms], MUST in Python List format, each term string reflects one user preference, and should be wrapped with quotation marks. Terms are seperated by comma. The quotation marks inside the term should use the escape character to avoid mismatch.
# 2. In order to ensure the comprehensiveness of the new user profile, the number of generated terms MUST be at least the number of items given to you (e.g. 5 terms for 5 items). You MUST examine the item list comprehensively, not omitting any item.
# 3. Each term should explicitly retain all emotional phrases (e.g. like/dislike). It is FORBIDDEN to directly use the name of genres to represent user's preferences.
# 4. You may make speculations about what features in the Item Profiles make the user like/dislike the given items, or examine how Original User Profile can align with the given Item Profiles. You are encouraged to use other knowledge inside your memory about the items, if you are sure it is accurate. You are FORBIDDEN to make unfounded assumptions about the given items, and fabricate anything that cannot be inferred from the input.
# 5. You should ONLY generate newly inferred features, be specific and personalized, highlighting detailed and distinguishable feature of the user.
# 6. If you are not completely sure about the preference of the user, you may use phrases like "may like", "like to some extent" or "moderately like" when describing a user's mid-level fondness of a genre.
# 7. When rating of an item is lower than or equal to 3.0, you MUST consider that the user does not like this item. Otherwise, when rating is above than or equal to 3.5, it is a high score, showing that the user likes this item.

# For a specific scenario, an example on movie recommendation would be:
# """},
# {"role": "assistant", "content": "Sure, I will generate the user's new profile after I see the rating and the items' attribute according to the requirements. "}
# ], # newprompt，之前测试的效果不太好

#             [
# {"role": "system", "content": "You are a helpful assistant on movie recommendation. "},
# {"role": "user", "content": """Now I will give you:
# 1. The Original Profile of a user, describing the user's current preferences and dislikes based on the previous items the user interacted; 
# 2. The profile of an item interacted by the user, containing the item's name, category and some descriptions;
# 3. The rating of the item scored by user.
# Now you are considering to generate a New User Profile of the user, based on Original User Profile, Item Profile and Rating information.
# To do this, please follow the steps below:
# 1. Review the given Original User Profile and Item Profile, and remember the profile terms that already exist in the Original User Profile. You should AVOID generating such terms repeatedly.
# 2. Extract the preferences and dislikes of the user from the user's Original Profile.
# 3. Give a Thought according to user's preferences and dislikes, item profile and the user's rating, try to explain try to explain what main features make the user give the item a low/high rating. Specially, you should try to find out the relationship between the rating and the features in Item Profile. The Thought MUST NOT contain information not related to the current asked item, and MUST NOT have unfounded assumptions about the item.
# 4. Generate the New User Profile based on the information derived in the Thought. The New User Profile should ONLY contain newly inferred features, AVOID generating features that you already know about the user, and AVOID repeating terms generated in the last few interaction rounds. 

# Important note:
# 1. Your output should in the format:
# Thought: Plain text.
# New User Profile: [list of terms], MUST be in Python List format, each term string should be wrapped with quotation marks, and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
# 2. Your generated New User Profile should only contain newly inferred user profile terms, AVOID generating terms that semantically similar to terms already exist in the Original Profile, especially terms that only have a few word's difference. If you think that there is no need to update the user profile based on the given information, you can generate an empty Python List, i.e. New User Profile: [].
# 3. You should pay more attention to the given Item Profile first, try to make speculations about what features in the Item Profile makes the user like / not like the given item. You may also use the Original User Profile as a reference, to find out how the terms in the Original User Profile can align with the given Item Profile. It is STRICTLY PROHIBITED to make unfounded assumptions about the given item, or use terms in the Original User Profile as the complement of the item's profile, unless you are really sure you know other information about this item.
# 4. Your output's language should be diverse, should not contain repeated information, or simply repeat information in the given example.
# 5. Your output should not be too general - it should be specific and personalized, highlighting detailed feature of the user. Any ambiguous preferences and dislikes that cannot distinguish the user from others are not worth recording.
# 6. You may use some adverbs to describe a user when you are not completely sure about the preference of a user. For example, you may use phrases like "may like", "like to some extent" or "moderately like" when describing a user's mid-level fondness of a genre, but DO NOT abuse these phrases in every circumstances.
# 7. Your output should be contain more detailed information you can derive, specifically, put all credible conclusions you obtained from the Thought into the New User Profile. DO NOT fabricate anything that cannot be inferred from input.
# 8. When rating is lower or equals to 3, you need to consider it as a low score, showing that the user do not like this item. Otherwise, it is a high score, showing that the user like this item.

# An example would be:
# """},
# {"role": "assistant", "content": "Sure, I will generate the user's new profile after I see the rating and the movie's attribute according to the requirements. "}
# ]
        ]
        self.condense_prompt = [
            [
# {"role": "system", "content": "You are a helpful assistant on recommendation."},
{"role": "user", "content": """Now I will give you the Original Profile of a user, describing the user's preferences and dislikes.
The profile might be too long, redundant and superficial. Now you are considering to summarize and distill the Original User Profile into a little bit shorter New User Profile.
To do this, please follow the steps below:
1. Review the Original Profile of the user, then extract the preferences and dislikes from it. You should review every terms in the Original Profile one after another, from the beginning to the end. The extraction MUST be comprehensive - every term, especially those only mentioned once in the Original Profile, MATTERS. You MUST pay attention to all information mentioned.
2. Give a Thought according to user's preferences and dislikes, summarize and distill the features of the Original User Profile. Be noted that directly discarding features inside the Original User Profile is STRICTLY PROHIBITED. Also, MUST NOT be too summarized - it is BETTER to obtain a longer textual result having more valuable information than a shorter one.
3. Generate the New User Profile based on the information derived in the Thought. The total text length of the New User Profile MUST be MORE than half of the Original User Profile, and less than three forth of the Original User Profile, in order to make sure that all the details in Original User Profile can be retained. Moreover, the number of terms in New User Profile should be MORE THAN half, and LESS THAN three forth of the total number of terms in Original User Profile, in order to leave enough space for further generated terms. You should generate New User Profile that is long enough to retain all the information in Original User Profile - within the limits above, it is always better to derive longer New User Profile than shorter ones.

Important note:
1. Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST be in List format, each term string should be wrapped with quotation marks and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. Your output should explicitly retain all emotional phrases in each term, which directly reflect the user's preferences and dislikes. It is FORBIDDEN to directly use the name of genres to represent user's fondness by default.
3. When there exist terms with conflicting meanings, e.g. "like comedy" and "not a fan of comedy", you may use some mid-level adverbs to express the user's mid-level interest to neutralize this conflict, e.g. "moderately like comedy", "like comedy to some extent", or "having complicated feeling about comedy".
4. Please pay extra attention on the terms which only have tiny differences literally and semantically. You should integrate them into one term. Integration is IMPORTANT, because the total length of New User Profile is limited.
5. You should MAKE SURE the completeness of the New User Profile, i.e. every identifiable user feature in Original Profile and Thought MUST BE retained, before and after condensing.
6. Your output should not only contain the features user like, but also include features that user do not like, e.g. "not a fan of comedy".
7. Your output should have more detail and comprehensive, be specific and personalized, highlighting detailed and distinguishable feature of the user, and AVOID fabricating features that not exist in Original Profile.
8. Do not output duplicate terms.

For a specific scenario, an example on movie recommendation would be:
"""},
{"role": "assistant", "content": """Sure, I will try my best to summarize the user's profile to New User Profile."""}
],
            [
# {"role": "system", "content": "You are a helpful assistant on office products recommendation."},
{"role": "user", "content": """Now I will give you the Original Profile of a user, describing the user's preferences and dislikes.
The profile might be too long, redundant and superficial. Now you are considering to summarize and distill the Original User Profile into a shorter New User Profile.
To do this, please follow the steps below:
1. Review the Original Profile of the user, then extract the preferences and dislikes from it. You should review every terms in the Original Profile one after another, from the beginning to the end. The extraction MUST be comprehensive - every term, especially those only mentioned once in the Original Profile, MATTERS. You MUST pay attention to all information mentioned.
2. Give a Thought according to user's preferences and dislikes, summarize and distill the features of the Original User Profile. Your summarization and distillation must be comprehensive, containing all information mentioned in the original user profile. Directly discarding features inside the Original User Profile is STRICTLY PROHIBITED.
3. Generate the New User Profile based on the information derived in the Thought. The total text length of the New User Profile MUST be MORE than half of the Original User Profile, and less than three forth of the Original User Profile, in order to make sure that all the details in Original User Profile can be retained. Moreover, the number of terms in New User Profile should be MORE THAN half, and LESS THAN three forth of the total number of terms in Original User Profile, in order to leave enough space for further generated terms. You should generate New User Profile that is long enough to retain all the information in Original User Profile - within the limits above, it is always better to derive longer New User Profile than shorter ones.

Important note:
1. Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST be in List format, each term string should be wrapped with quotation marks and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. Your output should explicitly retain all emotional phrases in each term, which directly reflect the user's preferences and dislikes. It is FORBIDDEN to directly use the name of genres to represent user's fondness by default.
3. You should pay extra attention on the terms which only have tiny differences literally and semantically. It is IMPORTANT to integrate them into one term.
4. When there exist terms with conflicting meanings, e.g. "like comedy" and "not a fan of comedy", you may use some mid-level adverbs to express the user's mid-level interest to neutralize this conflict, e.g. "moderately like comedy", "like comedy to some extent", or "having complicated feeling about comedy".
5. You should MAKE SURE the completeness of the New Item Profile, i.e. every identifiable user feature in Original Profile and Thought MUST BE retained, before and after condensing. The total text length of the New User Profile MUST be MORE than half of the Original User Profile, and the number of terms in New User Profile should be MORE THAN half of the total number of terms in Original User Profile.
6. Your output should not only contain the features user like, but also include features that user do not like, e.g. "not a fan of comedy".
7. Your output should have more detail and comprehensive, be specific and personalized, highlighting detailed and distinguishable feature of the user, and AVOID generating too short New User Profile, or fabricating features that not exist in Original Profile.
8. Do not output duplicate terms.
 
For a specific scenario, an example on movie recommendation would be:
"""},
# {"role": "user", "content": """Above is the user profile list given to you. Please be aware that you are given a task of summarizing and distilling the Original User Profile into a little bit shorter New User Profile, based on the User Profile list given to you.
# You MUST strictly follow the task instructions and give the output in correct format. You are FORBIDDEN to give freestyled recommendations and description summarizations, because they are NOT YOUR TASK.
# Now we emphasize the instructions of the task to you again:
# To do this, please follow the steps below:
# 1. Review the Original Profile of the user, then extract the preferences and dislikes from it. You should review every terms in the Original Profile one after another, from the beginning to the end. The extraction MUST be comprehensive - every term, especially those only mentioned once in the Original Profile, MATTERS. You MUST pay attention to all information mentioned.
# 2. Give a Thought according to user's preferences and dislikes, summarize and distill the features of the Original User Profile. Your summarization and distillation must be comprehensive, containing all information mentioned in the original user profile. Directly discarding features inside the Original User Profile is STRICTLY PROHIBITED.
# 3. Generate the New User Profile based on the information derived in the Thought. The total text length of the New User Profile MUST be MORE than half of the Original User Profile, and less than three forth of the Original User Profile, in order to make sure that all the details in Original User Profile can be retained. Moreover, the number of terms in New User Profile should be MORE THAN half, and LESS THAN three forth of the total number of terms in Original User Profile, in order to leave enough space for further generated terms. You should generate New User Profile that is long enough to retain all the information in Original User Profile - within the limits above, it is always better to derive longer New User Profile than shorter ones.

# Important note:
# 1. Your output should in the format:
# Thought: Plain text.
# New User Profile: [list of terms], MUST be in Python List format, each term string should be wrapped with quotation marks and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
# 2. Your output should explicitly retain all emotional phrases in each term, which directly reflect the user's preferences and dislikes. It is FORBIDDEN to directly use the name of genres to represent user's fondness by default.
# 3. You should pay extra attention on the terms which only have tiny differences literally and semantically. It is IMPORTANT to integrate them into one term.
# 4. When there exist terms with conflicting meanings, e.g. "like comedy" and "not a fan of comedy", you may use some mid-level adverbs to express the user's mid-level interest to neutralize this conflict, e.g. "moderately like comedy", "like comedy to some extent", or "having complicated feeling about comedy".
# 5. You should MAKE SURE the completeness of the New Item Profile, i.e. every identifiable user feature in Original Profile and Thought MUST BE retained, before and after condensing. The total text length of the New User Profile MUST be MORE than half of the Original User Profile, and the number of terms in New User Profile should be MORE THAN half of the total number of terms in Original User Profile.
# 6. Your output should not only contain the features user like, but also include features that user do not like, e.g. "not a fan of comedy".
# 7. Your output should have more detail and comprehensive, be specific and personalized, highlighting detailed and distinguishable feature of the user, and AVOID generating too short New User Profile, or fabricating features that not exist in Original Profile.

# No matter how you can fully understand the user profile list, your output should strictly follow the given format. Otherwise you will be punished.
# """},
{"role": "user", "content": """Above is the user profile list given to you. Please be aware that you are given a task of summarizing and distilling the Original User Profile into a little bit shorter New User Profile, based on the User Profile list given to you.
You MUST strictly follow the task instructions and give the output in correct format. You are FORBIDDEN to give freestyled recommendations and description summarizations, because they are NOT YOUR TASK.
Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST be in Python List format, each term string should be wrapped with quotation marks and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
No matter how you can fully understand the user profile list, your output should strictly follow the given format. Otherwise you will be punished.
"""},
{"role": "assistant", "content": "Sure, I will try my best to summarize the user's profile to New User Profile."}
],
#             [
# # {"role": "system", "content": "You are a helpful assistant on office products recommendation."},
# {"role": "user", "content": """Now I will give you the Original Profile of a user, describing the user's preferences and dislikes.
# The profile might be too long, redundant and superficial. Now you are considering to summarize and distill the Original User Profile into a shorter New User Profile.
# To do this, please follow the steps below:
# 1. Review the Original Profile of the user, then extract the preferences and dislikes from it. You should review every terms in the Original Profile one after another, from the beginning to the end. The extraction MUST be comprehensive - every term, especially those only mentioned once in the Original Profile, MATTERS. You MUST pay attention to all information mentioned.
# 2. Give a Thought according to user's preferences and dislikes, summarize and distill the features of the Original User Profile. Be noted that directly discarding features inside the Original User Profile is STRICTLY PROHIBITED.
# 3. Generate the New User Profile based on the information derived in the Thought. You should control the total text length of your output - the sum of the text length of terms in New User Profile MUST be MORE THAN 60%, and MUST BE LESS than 75% of the sum of the text length of terms in Original User Profile, in order to make sure that all the details in Original User Profile can be retained. However, the total number of terms in New User Profile should be MORE THAN 50%, and LESS than 60% of the total number of terms in Original User Profile, in order to leave enough space for further generated terms. You may use a combined longer terms to represent several short terms.

# Important note:
# 1. Your output should in the format:
# Thought: Plain text.
# New User Profile: [list of terms], MUST be in Python List format, each term string should be wrapped with quotation marks and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
# 2. Your output should explicitly retain all emotional phrases in each term, which directly reflect the user's preferences and dislikes. It is FORBIDDEN to directly use the name of genres to represent user's fondness by default.
# 3. BE AWARE that the total length of New User Profile is limited to up to 75% of the original one. Therefore, you should pay extra attention on the terms which only have tiny differences literally and semantically. It is IMPORTANT to integrate them into one term.
# 4. When there exist terms with conflicting meanings, e.g. "like comedy" and "not a fan of comedy", you may use some mid-level adverbs to express the user's mid-level interest to neutralize this conflict, e.g. "moderately like comedy", "like comedy to some extent", or "having complicated feeling about comedy".
# 5. You should MAKE SURE the completeness of the New Item Profile, i.e. every identifiable user feature in Original Profile and Thought MUST BE retained, before and after condensing.
# 6. Your output should not only contain the features user like, but also include features that user do not like, e.g. "not a fan of comedy".
# 7. Your output should have more detail and comprehensive, be specific and personalized, highlighting detailed and distinguishable feature of the user, and AVOID fabricating features that not exist in Original Profile.

# For a specific scenario, an example on movie recommendation would be:
# """},
# {"role": "user", "content": """Above is the user profile list given to you. Please be aware that you are given a task of summarizing and distilling the Original User Profile into a little bit shorter New User Profile, based on the User Profile list given to you.
# You MUST strictly follow the task instructions and give the output in correct format. You are FORBIDDEN to give freestyled recommendations and description summarizations, because they are NOT YOUR TASK.
# Now we emphasize the instructions of the task to you again:
# To do this, please follow the steps below:
# 1. Review the Original Profile of the user, then extract the preferences and dislikes from it. You should review every terms in the Original Profile one after another, from the beginning to the end. The extraction MUST be comprehensive - every term, especially those only mentioned once in the Original Profile, MATTERS. You MUST pay attention to all information mentioned.
# 2. Give a Thought according to user's preferences and dislikes, summarize and distill the features of the Original User Profile. Be noted that directly discarding features inside the Original User Profile is STRICTLY PROHIBITED, and MUST NOT be too summarized.
# 3. Generate the New User Profile based on the information derived in the Thought. You should control the total text length of your output - the sum of the text length of terms in New User Profile MUST be MORE THAN 60%, and MUST BE LESS than 75% of the sum of the text length of terms in Original User Profile, in order to make sure that all the details in Original User Profile can be retained. However, the total number of terms in New User Profile should be MORE THAN 50%, and LESS than 60% of the total number of terms in Original User Profile, in order to leave enough space for further generated terms. You may use a combined longer terms to represent several short terms.

# Important note:
# 1. Your output should in the format:
# Thought: Plain text.
# New User Profile: [list of terms], MUST be in Python List format, each term string should be wrapped with quotation marks and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
# 2. Your output should explicitly retain all emotional phrases in each term, which directly reflect the user's preferences and dislikes. It is FORBIDDEN to directly use the name of genres to represent user's fondness by default.
# 3. BE AWARE that the total length of New User Profile is limited to up to 75% of the original one. Therefore, you should pay extra attention on the terms which only have tiny differences literally and semantically. It is IMPORTANT to integrate them into one term.
# 4. When there exist terms with conflicting meanings, e.g. "like comedy" and "not a fan of comedy", you may use some mid-level adverbs to express the user's mid-level interest to neutralize this conflict, e.g. "moderately like comedy", "like comedy to some extent", or "having complicated feeling about comedy".
# 5. You should MAKE SURE the completeness of the New Item Profile, i.e. every identifiable user feature in Original Profile and Thought MUST BE retained, before and after condensing.
# 6. Your output should not only contain the features user like, but also include features that user do not like, e.g. "not a fan of comedy".
# 7. Your output should have more detail and comprehensive, be specific and personalized, highlighting detailed and distinguishable feature of the user, and AVOID fabricating features that not exist in Original Profile.

# No matter how you can fully understand the user profile list, your output should strictly follow the given format. Otherwise you will be punished.
# """},
# {"role": "assistant", "content": "Sure, I will try my best to summarize the user's profile to New User Profile."}
# ],
#             [
# {"role": "system", "content": "You are a helpful assistant on recommendation."},
# {"role": "user", "content": """Now I will give you the Original Profile of a user, describing the user's preferences and dislikes.
# The profile might be too long and superficial because each time the user interact with an item, some description terms will be added to the user profile. Too much redundant and repeated information can be confusing and can blur the focus, which is bad for recommendation.
# Now you are considering to summarize and distill the Original User Profile into a little bit shorter New User Profile.
# To do this, please follow the steps below:
# 1. Extract the preferences and dislikes from the Original Profile of the user. You should review the terms in the Original Profile one by one, and MUST pay attention to all information mentioned in the Original Profile. The extraction MUST be comprehensive - every term, especially those only mentioned once in the Original Profile, MATTERS.
# 2. Give a Thought according to user's preferences and dislikes, summarize and distill the features of the Original User Profile. Be noted that it is STRICTLY PROHIBITED to directly discard some features inside the Original User Profile. Also, MUST NOT be too summarized - it is ALWAYS better to obtain a longer textual result having more valuable information than a shorter one.
# 3. Generate the New User Profile based on the information derived in the Thought. The sum of the text length of terms in New User Profile MUST be MORE THAN 60%, and LESS than 75% of the sum of the text length of terms in Original User Profile, in order to make sure that all the details in Original User Profile can be retained.

# Important note:
# 1. Your output should in the format:
# Thought: Plain text.
# New User Profile: [list of terms], MUST be in Python List format, each term string should be wrapped with quotation marks and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
# 2. Your output's language should be diverse, should not contain repeated information, or simply repeat information in the given example.
# 3. Please pay extra attention on the terms which only have tiny differences in semantics. You should integrate them into one term.
# 4. When there exist terms with conflicting meanings, e.g. "like comedy" and "not a fan of comedy", you may use some mid-level adverbs to express the user's mid-level interest to neutralize this conflict, e.g. "moderately like comedy", "like comedy to some extent", or "having complicated feeling about comedy".
# 5. Your output should have more detail and be comprehensive, AVOID fabricating features that not exist in Original Profile, and MUST retain every identifiable user feature in Original Profile and Thought.
# 6. Your output should not only contain the features user like, but also include features that user do not like, and use appropriate words to describe them, e.g. "not a fan of comedy".
# 7. Your output should be specific, in-depth and personalized, highlighting detailed feature of the user. Any ambiguous preferences and dislikes that cannot distinguish the user from others are not worth recording.

# An example would be:
# """},
# {"role": "assistant", "content": "Sure, I will try my best to summarize the user's profile to New User Profile"}
# ]
        ]
        self.conflict_prompt = [
                [
{"role": "user", "content": """Now I will give you the Original Profile of a user, describing the user's preferences and dislikes.
The profile may contain multiple pairs or groups of terms which have conflicting meanings, e.g. "like comedy", "dislike comedy" and "not a fan of comedy". Now you are considering to neutralize these conflicts in the profile.
To do this, please follow the steps below:
1. Review the Original Profile of the user, then extract the preferences and dislikes from it. You should review every terms in the Original Profile one after another, from the beginning to the end. The extraction MUST be comprehensive - every term, especially those only mentioned once in the Original Profile, MATTERS. You MUST pay attention to all information mentioned.
2. Give a Thought including the extracted preferences and dislikes, and explore the terms having conflicting meanings based on the extraction results and the Original Profile. You MUST explore comprehensively based on the information derived in reviewing, not ignoring any conflicting information. You should also briefly explain the reasons why they are conflicting with each other. When you think there does not exist terms having conflicting meanings, you can also output this conclusion in the Thought.
3. Neutralize the terms having conflicting meanings derived in the Thought, merging them into a single term, and generating the New User Profile. Be aware that you should ALWAYS RESERVE other unrelated terms in the profile, it is FORBIDDEN to delete or modify them. When there does not exist terms having conflicting meanings, you should output the New User Profile which is exactly the same as the Original Profile.

Importent note:
1. Your output should in the format:
Thought: Plain text.
New User Profile: [list of terms], MUST be in Python List format, each term string should be wrapped with quotation marks and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. Your should consider the number of occurrence of each conflicting meaning, and use different adverbs to reflect the degrees of user preference, e.g. "moderately like/dislike", "like ... to some extent", ""having complicated feeling about ...", "probably like/dislike", etc.
3. Your output MUST NOT delete any terms, unless they are conflicted with other terms and are merged.

For a specific scenario, an example on movie recommendation would be:
"""},
                ]
        ]
        self.evaluate_prompt = [
                [
{"role": "system", "content": "You are a helpful assistant on recommendation. "},
{"role": "user", "content": """Now I will give you:
1. The Final User Profile, representing the user's preferences and dislikes towards different types of items.
2. The information of 20 items in the testing set, in the format of "Index\tItem Name\tItem Profile", each single line includes an item's name and profile, containing categories and some descriptions about the item (if any).
Now you are considering to rank these items, based on the matching degree of the Final User Profile and the information about these items, in the order of user's interest.
To do this, please follow the steps below:
1. Review the Final User Profile of the user, and extract the preferences and dislikes of the user. Be aware that there may be some conflicting terms in the Final User Profile, e.g. "like comedy" and "not a fan of comedy". You need to treat all terms comprehensively, DO NOT miss any information of the user.
2. Review the information about these 20 items, give a Thought to explain your idea on the matching degree of these items. Specifically, you should try to point out how each item's profile can align with or having difference with the user's profile. You are encouraged to expand a little bit, use other knowledge you know about this item, apart from the given item profile. However, you should make sure that the information you use as proof is really related to the item.
3. Generate the ranking of the 20 items you believe, sorted in the order from best matched to worst matched for users.

Important note:
1. Your output should in the format:
Thought: 
1. Item Name 1: Your explanation of the matching degree of item 1, in one sentence.
2. Item Name 2: Your explanation of the matching degree of item 2, in one sentence.
...
20. Item Name 20: Your explanation of the matching degree of item 20, in one sentence.
The ranking of the items:
1. Item Name 1
2. Item Name 2
...
20. Item Name 20
2. You should sort the item in the order of the matching degree, from best matched to worst matched between the given user and items. It is also the only determining factor for ranking. The input order of 20 items has nothing to do with the matching degree, and you MUST NOT treat this factor as evidence."""},
{"role": "assistant", "content": "Sure, I will try my best to generate the ranking of the given 20 items, according to the requirements."}
                ]
        ]
        self.score_prompt = [
                [
{"role": "system", "content": "You are a helpful assistant on recommendation. "},
{"role": "user", "content": """Now I will give you:
1. The Final User Profile, representing the user's preferences and dislikes towards different types of items.
2. The information of 20 items in the testing set, in the format of "Index\tItem Name\tItem Profile", each single line includes an item's name and profile, containing categories and some descriptions about the item (if any). The items are randomly given to you, their order has nothing to do with the matching degree between them and the given user.
Now you are considering to score these items, based on the matching degree of the Final User Profile and the information about these items, in the order of user's interest.
To do this, please follow the steps below:
1. Review the Final User Profile of the user, and extract the preferences and dislikes of the user. Be aware that there may be some conflicting terms in the Final User Profile, e.g. "like comedy" and "not a fan of comedy". You need to treat all terms comprehensively, DO NOT miss any information of the user.
2. Review the information about these 20 items, give a Thought to explain your idea on the matching degree of these items. Specifically, you should try to point out how each item's profile can align with or having difference with the user's profile. You are encouraged to expand a little bit, use other knowledge you know about this item, apart from the given item profile. However, you should make sure that the information you use as proof is really related to the item.
3. Score each item based on the previous Thought. The matching scores should be bigger than 0 and smaller than 1, bigger score represents that the item is more aligned with the user. You MUST make sure that the scores you give have differentiate ability, i.e. the rating range should be large, completely fill the range of [0, 1], and can accurately depict the matching degree of each item with the given user. Also, the scores should be rounded to 4 decimal places without zeros in the end (e.g. 0.1234 is acceptable, but 0.1000 is FORBIDDEN), and are pairwise different.

Important note:
1. Your output should in the format:
Thought:
1. Item Name 1: Your explanation of the matching degree of item 1, in one sentence.
2. Item Name 2: Your explanation of the matching degree of item 2, in one sentence.
...
20. Item Name 20: Your explanation of the matching degree of item 20, in one sentence.
The matching scores of the items:
[Item 1's score, Item 2's score, ..., Item 20's score]. MUST be in Python List format, each term represents an item's matching score towards the given user.
2. The scores inside the matching score list you outputted should be consistent with the input order of the 20 items."""},
{"role": "assistant", "content": "Sure, I will try my best to score the given 20 items, based on the matching degree of them with the given user, and according to the requirements."}
                ]
        ]
        self.description_prompt = [
                [
{"role": "system", "content": "You are a helpful assistant on summarization. "},
{"role": "user", "content": """We ask you to forget the past received information, and focus on the task below.
Now I will give you the Name, Category and Descriptions of one item, in the format of "Index\tItem Name\tItem Category\tItem Description". 
The description is too long. So, you are considered to condense the description into a shorter one.
To do this, you should generate the new description of the given item, STRICTLY following the required format. You should be sure that the new description of the item is comprehensive enough to depict the full image of the item, and meanwhile, the new description you generated is good enough to differentiate the item from other items.

Important note:
1. We ask you that your first line of output MUST be EXACTLY "New Item Descriptions:", followed by one line, including the item's sequence number and the condensed description. Please think before you act, if your output format is wrong, we will punish you. Intuitively, your output MUST be EXACTLY the same format as this:

New Item Descriptions: (Your output MUST be EXACTLY the same as this line, every word and mark MUST be EXACTLY the same)
1. New Description 1

The new descriptions MUST begin with the line "New Item Descriptions:", otherwise we cannot identify the descriptions you generated. It is FORBIDDEN to output information other than the format above.
2. The New Item Description's length should be LESS than 512 characters. Within this limitation, you may describe the item in more detail."""},
                ],
                [
{"role": "system", "content": "You are a helpful assistant on summarization. "},
{"role": "user", "content": """We ask you to forget the past received informations, and focus on the task below.
Now I will give you the Name, Category and Descriptions of 4 items (possibly less than 4), in the format of "Index\tItem Name\tItem Category\tItem Description", one line for each item. 
Their descriptions are too long. So, you are considered to condense their descriptions into shorter ones.
To do this, you should generate the new descriptions of the given items, STRICTLY following the required format. You should be sure that the new description of each item is comprehensive enough to depict the full image of the item, and meanwhile, the new description you generated is good enough to differentiate the item from other items given.

Important note:
1. We ask you that your first line of output MUST be EXACTLY "New Item Descriptions:", followed by several lines, one line for an item, including the item's sequence number and the condensed description. Please think before you act, if your output format is wrong, we will punish you. Intuitively, your output MUST be EXACTLY the same format as this:

New Item Descriptions: (Your output MUST be EXACTLY the same as this line, every word and mark MUST be EXACTLY the same)
1. New Description 1
2. New Description 2
...
4. New Description 4

The new descriptions MUST begin with the line "New Item Descriptions:", otherwise we cannot identify the descriptions you generated. It is FORBIDDEN to output information other than the format above.
2. You may be given less than 4 items, in this case, please just condense the descriptions of the items given to you, DO NOT inquire for more items.
3. Your output order of the items should be consistent with the given input, and the "New Item Descriptions:" line MUST be outputted.
4. Each of the New Item Description's length should be LESS than 512 characters. Within this limitation, you may describe the item in more detail."""},
                ],
                [
{"role": "system", "content": "You are a helpful assistant on summarization. "},
{"role": "user", "content": """Now I will give you the Name, Category and Descriptions of 3 items (possibly less than 3), in the format of "Index\tItem Name\tItem Category\tItem Description", one line for each item. 
Their descriptions are too long. So, you are considered to condense their descriptions into shorter ones.
To do this, please follow the steps below:
1. Review the descriptions of the given items, and extract the main content and features of each item. You need to give a Thought of your extraction process of each item. When doing extraction, you should be comprehensive, DO NOT miss any important features of each item. Also, you should be aware that the given items are different, each of them has its own uniqueness. Therefore, during extraction, you need to highlight the unique characteristics of each item.
2. Generate the new descriptions of the given items, based on the Thought and the knowledge you learned during extraction, following the required format. Also, you should be sure that the new description of each item contains valuable conclusions derived in the Thought, and is comprehensive enough to depict the full image of the item, and meanwhile, the new description you generated is good enough to differentiate the item from other items given.

Important note:
1. Your output MUST be EXACTLY the same format as this:

Thought: (MUST be EXACTLY the same as this line)
1. Item Name 1: Item Description 1
2. Item Name 2: Item Description 2
3. Item Name 3: Item Description 3
The New Descriptions: (MUST be EXACTLY the same as this line)
1. New Item Description 1
2. New Item Description 2
3. New Item Description 3

The Thought MUST begin with the line "Thought:", and the new descriptions MUST begin with the line "The New Descriptions:", otherwise we cannot identify the descriptions you generated. It is FORBIDDEN to output information other than the format above.
2. You may be given less than 3 items, in this case, please just condense the descriptions of the items given to you, DO NOT inquire for more items.
3. Your output order of the items should be consistent with the given input, and the "The New Description:" line MUST be outputted.
4. You should describe the item in more words with more detail."""},
                ],
                [
{"role": "system", "content": "You are a helpful assistant on recommendation. "},
{"role": "user", "content": """Now I will give you the Name, Category and Descriptions of an item, in the format of "Index\tItem Name\tItem Category\tItem Description". 
The item's descriptions are too long, so there may exist some problem if we directly treat it as the input of the recommendation system. So, you are considered to condense the description into a shorter one.
To do this, please follow the steps below:
1. Review the descriptions of the given item, and extract the main content and features of it. When doing extraction, you should be comprehensive, DO NOT miss any important features of the item. Also, you need to highlight the unique characteristics of the item, which can differentiate it from other items.
2. Generate the new descriptions of the given item, based on the knowledge you learned during extraction. Also, you should be sure that the new description of the item contains valuable conclusions derived in the Thought, and is comprehensive enough to depict the full image of the item, and meanwhile, the new description you generated is good enough to differentiate the item from other items.

Important note:
1. Your output should in the format:
The New Descriptions:
1. New Item Description 1

2. Your output order of the items should be consistent with the given input.
3. The New Item Description's length should be NO MORE than 512 characters. Within this limitation, you may describe the item in more detail."""}
                ]
        ]
#         self.user_example = [
# """User Original Profile: ["age: 25", "gender: male", "occupation: writer", "prefers movies with strong storytelling", "looks for well-developed characters", "enjoys action movies", "appreciates movies with dramatic elements", "interested in western genre", "enjoys thrilling action movies", "appreciates well-crafted characters", "likes movies with strong storytelling", "interested in movies with dramatic elements", "has a preference for the western genre", "not a fan of weak storytelling", "appreciates well-developed characters", "prefers movies with impactful dramatic elements", "specific about western genre expectations", "not a fan of sci-fi movies", "not interested in weak storytelling in movies", "not a fan of 'Star Trek' franchise"]
# Item: "Alien 3 (1992)"
# Item Profile: ["Alien 3 (1992)", "Action", " Horror", " Sci-Fi", " Thriller"]
# Rating: "1/5"

# Your output:
# Thought: Based on the user's Original Profile, it is clear that the user has a strong preference for movies with strong storytelling, well-developed characters, and impactful dramatic elements. The user also mentioned an interest in the western genre, thrilling action movies, and an appreciation for well-crafted characters. However, the user explicitly stated that he is not a fan of weak storytelling and sci-fi movies. 
# The user rated "Alien 3" with a low score of 1. From the item's profile, we can see that "Alien 3" falls into the genres of Action, Horror, Sci-Fi, and Thriller. Since the user enjoys thrilling action movies, these two genres are not the main cause that make the user dislike this movie. It is likely that the user's dislike for weak storytelling and sci-fi movies influenced his low rating for "Alien 3". Additionally, since the user mentioned not being a fan of the "Star Trek" franchise, which is a popular sci-fi series, it is possible that the user's dislike for sci-fi extended to "Alien 3". We can also infer that the user dislike horror movies from the user's dislike for "Alien 3", which falls into the horror genre.
# New User Profile: ["not a fan of weak storytelling in movies", "not a fan of horror movies"]""",

# """User Original Profile: ["age: 17", "gender: male", "occupation: student", "enjoys action movies", "appreciates fast-paced and thrilling films", "enjoys drama films", "interested in sci-fi movies", "loves action-packed movies", "thrives on fast-paced and thrilling films", "appreciates adventurous plots", "enjoys movies with mystery elements", "not a big fan of comedy films", "thrives on fast-paced and thrilling films", "enjoys fantasy movies", "appreciates adventurous plots in movies", "loves action-packed and thrilling movies", "appreciates sci-fi elements in movies", "enjoys movies with war themes", "appreciates drama films", "enjoys romantic movies", "appreciates slower-paced movies"]
# Item: "Mars Attacks! (1996)"
# Item Profile: ["Mars Attacks! (1996)", "Action", " Comedy", " Sci-Fi", " War"]
# Rating: "3/5"

# Your output:
# Thought: Based on the given user's Original Profile, it is clear that the user enjoys action, romantic, fantasy, thrilling, drama and sci-fi movies, and movies with adventurous and mystery elements. The user also mentions that he is not a big fan of comedy films and prefers action-packed and fast-paced movies. 
# The user has rated the movie "Mars Attacks! (1996)" with a rating of 3. The movie's profile includes action, comedy, sci-fi, and war genres. 
# From this information, since the user enjoys all genres other than comedy, it can be inferred that the user might have given a low rating to "Mars Attacks! (1996)" because it includes the comedy genre, which the user is not a big fan of. The user's preference for fast-paced and action-packed movies might also contribute to the lower rating, as the movie might not have met his expectations in terms of the level of action and pace.
# New User Profile: ["not a big fan of comedy films", "prefers fast-paced and action-packed movies"]""",

# """User Original Profile: ["age: 21", "gender: male", "occupation: student", "enjoys action-packed films", "appreciates intense and suspenseful movies", "interested in war-related themes", "enjoys comedic elements", "appreciates comedy films", "has a diverse taste in movies", "appreciates emotional and intense dramas", "enjoys well-developed characters", "loves mystery thrillers", "appreciates engaging and thrilling movies", "enjoys well-developed characters in films", "appreciates complex and emotional storylines", "enjoys movies with war-related themes", "enjoys crime films", "appreciates well-developed characters in crime movies", "likes intense and suspenseful dramas", "interested in complex storylines", "enjoys action-packed and intense movies", "appreciates well-developed characters in films", "likes thriller movies", "enjoys drama films", "interested in sci-fi movies", "appreciates films with emotional and intense storylines", "likes movies with complex themes", "interested in exploring the unknown", "appreciates intense and emotional dramas", "appreciates movies with moral complexities", "interested in exploring ethical dilemmas", "enjoys comedy films with intense and suspenseful elements"]
# Item: "Return of the Jedi (1983)"
# Item Profile: ["Return of the Jedi (1983)", "Action", " Adventure", " Romance", " Sci-Fi", " War"]
# Rating: "4/5"

# Your output:
# Thought: Based on the user's Original Profile, we can see that the user is a 21-year-old male student with a diverse taste in movies. He enjoys action-packed films, intense and suspenseful movies, war-related themes, comedic elements, emotional and intense dramas, well-developed characters, mystery thrillers, engaging and thrilling movies, crime films, intense and suspenseful dramas, complex storylines, thriller movies, drama films, sci-fi movies, movies with complex themes, exploring the unknown, intense and emotional dramas, movies with moral complexities, and exploring ethical dilemmas.
# The user rated "Return of the Jedi (1983)" with a score of 4. Looking at the item's profile, we can see that it is an action, adventure, romance, sci-fi, and war movie. Considering the user's preferences for action-packed films, war-related themes, and sci-fi movies, it is likely that the user enjoyed these aspects of the movie. The user may have appreciated the well-developed characters and the intense and suspenseful elements present in the film.
# New User Profile: ["likes action and adventure movies", "appreciates well-developed characters in sci-fi films", "enjoys movies with war-related themes"]""",

# """User Original Profile: ["age: 46", "gender: male", "occupation: librarian", "prefers movies with intricate plots", "appreciates suspenseful elements", "enjoys crime movies", "appreciates thrillers", "likes movies with intricate plots", "dislikes movies with weak plot", "not a fan of movies lacking suspense", "prefers well-developed characters", "not a fan of romantic comedies", "prefers movies with intricate plots and suspenseful elements", "appreciates crime movies", "enjoys action movies", "appreciates mystery movies", "likes thrillers", "appreciates movies that keep him engaged", "enjoys movies that keep him guessing", "not a fan of horror movies", "appreciates well-developed characters", "appreciates comedy-drama genre", "enjoys movies with intricate plots and well-developed characters", "likes movies that keep him engaged and guessing"]
# Item: "English Patient, The (1996)"
# Item Profile: ["English Patient, The (1996)", "Drama", " Romance", " War"]
# Rating: "5/5"

# Your output:
# Thought: Based on the user's Original Profile, we can see that the user is a 46-year-old male librarian who prefers movies with intricate plots, appreciates suspenseful elements, enjoys action movies, mystery movies, crime movies and comedy-drama genre. The user also appreciates thrillers and movies that keep him engaged and guessing. However, the user dislikes movies with weak plots, movies lacking suspense, and romantic comedies. The user also mentions not being a fan of horror movies.
# The user gave "English Patient, The (1996)" a rating of 5. This movie is categorized as a Drama, Romance, and War film. Considering the user's preferences for intricate plots, well-developed characters, and engagement, it is likely that the user enjoyed the intricate plot and well-developed characters in "English Patient, The (1996)". The user's appreciation for drama might have contributed to the high rating as well. From the film's genre, it is likely that the user also enjoys romance and war movies, which are the new features of the user inferred from the user's positive experience with this movie, and can led to the inclusion of new preferences in the New User Profile.
# New User Profile: ["appreciates drama genre", "enjoys romance in movies", "likes war movies", "appreciates movies with intricate plots and well-developed characters"]"""
# ]


#         self.user_example = [
# """User Original Profile: ["age: 25", "gender: male", "occupation: writer", "prefers movies with strong storytelling", "looks for well-developed characters", "enjoys action movies", "appreciates movies with dramatic elements", "interested in western genre", "enjoys thrilling action movies", "appreciates well-crafted characters", "likes movies with strong storytelling", "interested in movies with dramatic elements", "has a preference for the western genre", "not a fan of weak storytelling", "appreciates well-developed characters", "prefers movies with impactful dramatic elements", "specific about western genre expectations", "not a fan of sci-fi movies", "not interested in weak storytelling in movies", "not a fan of 'Star Trek' franchise"]
# Item: "Alien 3 (1992)"
# Item Profile: ["Alien 3 (1992)", "Action", " Horror", " Sci-Fi", " Thriller"]
# Rating: "1/5"
# Collaborative Score: -0.1944116950035095

# Your output:
# Thought: Based on the user's Original Profile, it is clear that the user has a strong preference for movies with strong storytelling, well-developed characters, and impactful dramatic elements. The user also mentioned an interest in the western genre, thrilling action movies, and an appreciation for well-crafted characters. However, the user explicitly stated that he is not a fan of weak storytelling and sci-fi movies. 
# The user rated "Alien 3" with a low score of 1, and the collaborative score is also a negative number, indicates the user's dislike. From the item's profile, we can see that "Alien 3" falls into the genres of Action, Horror, Sci-Fi, and Thriller. Since the user enjoys thrilling action movies, these two genres are not the main cause that make the user dislike this movie. It is likely that the user's dislike for weak storytelling and sci-fi movies influenced his low rating for "Alien 3". Additionally, since the user mentioned not being a fan of the "Star Trek" franchise, which is a popular sci-fi series, it is possible that the user's dislike for sci-fi extended to "Alien 3". We can also infer that the user dislike horror movies from the user's dislike for "Alien 3", which falls into the horror genre.
# New User Profile: ["not a fan of weak storytelling in movies", "not a fan of horror movies"]""",

# """User Original Profile: ["age: 17", "gender: male", "occupation: student", "enjoys action movies", "appreciates fast-paced and thrilling films", "enjoys drama films", "interested in sci-fi movies", "loves action-packed movies", "thrives on fast-paced and thrilling films", "appreciates adventurous plots", "enjoys movies with mystery elements", "not a big fan of comedy films", "thrives on fast-paced and thrilling films", "enjoys fantasy movies", "appreciates adventurous plots in movies", "loves action-packed and thrilling movies", "appreciates sci-fi elements in movies", "enjoys movies with war themes", "appreciates drama films", "enjoys romantic movies", "appreciates slower-paced movies"]
# Item: "Mars Attacks! (1996)"
# Item Profile: ["Mars Attacks! (1996)", "Action", " Comedy", " Sci-Fi", " War"]
# Rating: "3/5"
# Collaborative Score: 0.2163684368133545

# Your output:
# Thought: Based on the given user's Original Profile, it is clear that the user enjoys action, romantic, fantasy, thrilling, drama and sci-fi movies, and movies with adventurous and mystery elements. The user also mentions that he is not a big fan of comedy films and prefers action-packed and fast-paced movies. 
# The user has rated the movie "Mars Attacks! (1996)" with a rating of 3. The movie's profile includes action, comedy, sci-fi, and war genres. And the collaborative score is also not very high.
# From this information, since the user enjoys all genres other than comedy, it can be inferred that the user might have given a low rating to "Mars Attacks! (1996)" because it includes the comedy genre, which the user is not a big fan of. The user's preference for fast-paced and action-packed movies might also contribute to the lower rating, as the movie might not have met his expectations in terms of the level of action and pace.
# New User Profile: ["not a big fan of comedy films", "prefers fast-paced and action-packed movies"]""",

# """User Original Profile: ["age: 21", "gender: male", "occupation: student", "enjoys action-packed films", "appreciates intense and suspenseful movies", "interested in war-related themes", "enjoys comedic elements", "appreciates comedy films", "has a diverse taste in movies", "appreciates emotional and intense dramas", "enjoys well-developed characters", "loves mystery thrillers", "appreciates engaging and thrilling movies", "enjoys well-developed characters in films", "appreciates complex and emotional storylines", "enjoys movies with war-related themes", "enjoys crime films", "appreciates well-developed characters in crime movies", "likes intense and suspenseful dramas", "interested in complex storylines", "enjoys action-packed and intense movies", "appreciates well-developed characters in films", "likes thriller movies", "enjoys drama films", "interested in sci-fi movies", "appreciates films with emotional and intense storylines", "likes movies with complex themes", "interested in exploring the unknown", "appreciates intense and emotional dramas", "appreciates movies with moral complexities", "interested in exploring ethical dilemmas", "enjoys comedy films with intense and suspenseful elements"]
# Item: "Return of the Jedi (1983)"
# Item Profile: ["Return of the Jedi (1983)", "Action", " Adventure", " Romance", " Sci-Fi", " War"]
# Rating: "4/5"
# Collaborative Score: 0.7545681638133436

# Your output:
# Thought: Based on the user's Original Profile, we can see that the user is a 21-year-old male student with a diverse taste in movies. He enjoys action-packed films, intense and suspenseful movies, war-related themes, comedic elements, emotional and intense dramas, well-developed characters, mystery thrillers, engaging and thrilling movies, crime films, intense and suspenseful dramas, complex storylines, thriller movies, drama films, sci-fi movies, movies with complex themes, exploring the unknown, intense and emotional dramas, movies with moral complexities, and exploring ethical dilemmas.
# The user rated "Return of the Jedi (1983)" with a score of 4, and the collaborative score of the movie is relatively high. Looking at the item's profile, we can see that it is an action, adventure, romance, sci-fi, and war movie. Considering the user's preferences for action-packed films, war-related themes, and sci-fi movies, it is likely that the user enjoyed these aspects of the movie. The user may have appreciated the well-developed characters and the intense and suspenseful elements present in the film.
# New User Profile: ["likes action and adventure movies", "appreciates well-developed characters in sci-fi films", "enjoys movies with war-related themes"]""",

# """User Original Profile: ["age: 46", "gender: male", "occupation: librarian", "prefers movies with intricate plots", "appreciates suspenseful elements", "enjoys crime movies", "appreciates thrillers", "likes movies with intricate plots", "dislikes movies with weak plot", "not a fan of movies lacking suspense", "prefers well-developed characters", "not a fan of romantic comedies", "prefers movies with intricate plots and suspenseful elements", "appreciates crime movies", "enjoys action movies", "appreciates mystery movies", "likes thrillers", "appreciates movies that keep him engaged", "enjoys movies that keep him guessing", "not a fan of horror movies", "appreciates well-developed characters", "appreciates comedy-drama genre", "enjoys movies with intricate plots and well-developed characters", "likes movies that keep him engaged and guessing"]
# Item: "English Patient, The (1996)"
# Item Profile: ["English Patient, The (1996)", "Drama", " Romance", " War"]
# Rating: "5/5"
# Collaborative Score: 0.9439505046116095

# Your output:
# Thought: Based on the user's Original Profile, we can see that the user is a 46-year-old male librarian who prefers movies with intricate plots, appreciates suspenseful elements, enjoys action movies, mystery movies, crime movies and comedy-drama genre. The user also appreciates thrillers and movies that keep him engaged and guessing. However, the user dislikes movies with weak plots, movies lacking suspense, and romantic comedies. The user also mentions not being a fan of horror movies.
# The user gave "English Patient, The (1996)" a rating of 5, and the movie enjoys a very high collaborative score. This movie is categorized as a Drama, Romance, and War film. Considering the user's preferences for intricate plots, well-developed characters, and engagement, it is likely that the user enjoyed the intricate plot and well-developed characters in "English Patient, The (1996)". The user's appreciation for drama might have contributed to the high rating as well. From the film's genre, it is likely that the user also enjoys romance and war movies, which are the new features of the user inferred from the user's positive experience with this movie, and can led to the inclusion of new preferences in the New User Profile.
# New User Profile: ["appreciates drama genre", "enjoys romance in movies", "likes war movies", "appreciates movies with intricate plots and well-developed characters"]"""
# ]
        
        self.user_example = [
"""User Original Profile: ["age: 25", "gender: male", "occupation: writer", "prefers movies with strong storytelling", "looks for well-developed characters", "enjoys action movies", "appreciates movies with dramatic elements", "interested in western genre", "enjoys thrilling action movies", "appreciates well-crafted characters", "likes movies with strong storytelling", "interested in movies with dramatic elements", "has a preference for the western genre", "not a fan of weak storytelling", "appreciates well-developed characters", "prefers movies with impactful dramatic elements", "specific about western genre expectations", "not a fan of sci-fi movies", "not interested in weak storytelling in movies", "not a fan of 'Star Trek' franchise"]
Item: "Alien 3 (1992)"
Item Profile: ["Alien 3 (1992)", "Action", " Horror", " Sci-Fi", " Thriller"]
Rating: "1/5"
Collaborative Hedonic Scale: Dislike Very Much

Your output:
Thought: Based on the user's Original Profile, it is clear that the user has a strong preference for movies with strong storytelling, well-developed characters, and impactful dramatic elements. The user also mentioned an interest in the western genre, thrilling action movies, and an appreciation for well-crafted characters. However, the user explicitly stated that he is not a fan of weak storytelling and sci-fi movies. 
The user rated "Alien 3" with a low score of 1, and the collaborative hedonic scale also indicates user's dislike. From the item's profile, we can see that "Alien 3" falls into the genres of Action, Horror, Sci-Fi, and Thriller. Since the user enjoys thrilling action movies, these two genres are not the main cause that make the user dislike this movie. It is likely that the user's dislike for weak storytelling and sci-fi movies influenced his low rating for "Alien 3". Additionally, since the user mentioned not being a fan of the "Star Trek" franchise, which is a popular sci-fi series, it is possible that the user's dislike for sci-fi extended to "Alien 3". We can also infer that the user dislike horror movies from the user's dislike for "Alien 3", which falls into the horror genre.
New User Profile: ["not a fan of weak storytelling in movies", "not a fan of horror movies"]""",

"""User Original Profile: ["age: 17", "gender: male", "occupation: student", "enjoys action movies", "appreciates fast-paced and thrilling films", "enjoys drama films", "interested in sci-fi movies", "loves action-packed movies", "thrives on fast-paced and thrilling films", "appreciates adventurous plots", "enjoys movies with mystery elements", "not a big fan of comedy films", "thrives on fast-paced and thrilling films", "enjoys fantasy movies", "appreciates adventurous plots in movies", "loves action-packed and thrilling movies", "appreciates sci-fi elements in movies", "enjoys movies with war themes", "appreciates drama films", "enjoys romantic movies", "appreciates slower-paced movies"]
Item: "Mars Attacks! (1996)"
Item Profile: ["Mars Attacks! (1996)", "Action", " Comedy", " Sci-Fi", " War"]
Rating: "3/5"
Collaborative Hedonic Scale: Neutral Towards Like

Your output:
Thought: Based on the given user's Original Profile, it is clear that the user enjoys action, romantic, fantasy, thrilling, drama and sci-fi movies, and movies with adventurous and mystery elements. The user also mentions that he is not a big fan of comedy films and prefers action-packed and fast-paced movies. 
The user has rated the movie "Mars Attacks! (1996)" with a rating of 3. The movie's profile includes action, comedy, sci-fi, and war genres. And the collaborative hedonic scale is also not very positive.
From this information, since the user enjoys all genres other than comedy, it can be inferred that the user might have given a low rating to "Mars Attacks! (1996)" because it includes the comedy genre, which the user is not a big fan of. The user's preference for fast-paced and action-packed movies might also contribute to the lower rating, as the movie might not have met his expectations in terms of the level of action and pace.
New User Profile: ["not a big fan of comedy films", "prefers fast-paced and action-packed movies"]""",

"""User Original Profile: ["age: 21", "gender: male", "occupation: student", "enjoys action-packed films", "appreciates intense and suspenseful movies", "interested in war-related themes", "enjoys comedic elements", "appreciates comedy films", "has a diverse taste in movies", "appreciates emotional and intense dramas", "enjoys well-developed characters", "loves mystery thrillers", "appreciates engaging and thrilling movies", "enjoys well-developed characters in films", "appreciates complex and emotional storylines", "enjoys movies with war-related themes", "enjoys crime films", "appreciates well-developed characters in crime movies", "likes intense and suspenseful dramas", "interested in complex storylines", "enjoys action-packed and intense movies", "appreciates well-developed characters in films", "likes thriller movies", "enjoys drama films", "interested in sci-fi movies", "appreciates films with emotional and intense storylines", "likes movies with complex themes", "interested in exploring the unknown", "appreciates intense and emotional dramas", "appreciates movies with moral complexities", "interested in exploring ethical dilemmas", "enjoys comedy films with intense and suspenseful elements"]
Item: "Return of the Jedi (1983)"
Item Profile: ["Return of the Jedi (1983)", "Action", " Adventure", " Romance", " Sci-Fi", " War"]
Rating: "4/5"
Collaborative Hedonic Scale: Like Very Much

Your output:
Thought: Based on the user's Original Profile, we can see that the user is a 21-year-old male student with a diverse taste in movies. He enjoys action-packed films, intense and suspenseful movies, war-related themes, comedic elements, emotional and intense dramas, well-developed characters, mystery thrillers, engaging and thrilling movies, crime films, intense and suspenseful dramas, complex storylines, thriller movies, drama films, sci-fi movies, movies with complex themes, exploring the unknown, intense and emotional dramas, movies with moral complexities, and exploring ethical dilemmas.
The user rated "Return of the Jedi (1983)" with a score of 4, and the collaborative hedonic scale of the movie is positive. Looking at the item's profile, we can see that it is an action, adventure, romance, sci-fi, and war movie. Considering the user's preferences for action-packed films, war-related themes, and sci-fi movies, it is likely that the user enjoyed these aspects of the movie. The user may have appreciated the well-developed characters and the intense and suspenseful elements present in the film.
New User Profile: ["likes action and adventure movies", "appreciates well-developed characters in sci-fi films", "enjoys movies with war-related themes"]""",

"""User Original Profile: ["age: 46", "gender: male", "occupation: librarian", "prefers movies with intricate plots", "appreciates suspenseful elements", "enjoys crime movies", "appreciates thrillers", "likes movies with intricate plots", "dislikes movies with weak plot", "not a fan of movies lacking suspense", "prefers well-developed characters", "not a fan of romantic comedies", "prefers movies with intricate plots and suspenseful elements", "appreciates crime movies", "enjoys action movies", "appreciates mystery movies", "likes thrillers", "appreciates movies that keep him engaged", "enjoys movies that keep him guessing", "not a fan of horror movies", "appreciates well-developed characters", "appreciates comedy-drama genre", "enjoys movies with intricate plots and well-developed characters", "likes movies that keep him engaged and guessing"]
Item: "English Patient, The (1996)"
Item Profile: ["English Patient, The (1996)", "Drama", " Romance", " War"]
Rating: "5/5"
Collaborative Hedonic Scale: Like Very Much

Your output:
Thought: Based on the user's Original Profile, we can see that the user is a 46-year-old male librarian who prefers movies with intricate plots, appreciates suspenseful elements, enjoys action movies, mystery movies, crime movies and comedy-drama genre. The user also appreciates thrillers and movies that keep him engaged and guessing. However, the user dislikes movies with weak plots, movies lacking suspense, and romantic comedies. The user also mentions not being a fan of horror movies.
The user gave "English Patient, The (1996)" a rating of 5, and the movie enjoys a very positive collaborative hedonic scale. This movie is categorized as a Drama, Romance, and War film. Considering the user's preferences for intricate plots, well-developed characters, and engagement, it is likely that the user enjoyed the intricate plot and well-developed characters in "English Patient, The (1996)". The user's appreciation for drama might have contributed to the high rating as well. From the film's genre, it is likely that the user also enjoys romance and war movies, which are the new features of the user inferred from the user's positive experience with this movie, and can led to the inclusion of new preferences in the New User Profile.
New User Profile: ["appreciates drama genre", "enjoys romance in movies", "likes war movies", "appreciates movies with intricate plots and well-developed characters"]"""
]
        self.user_example_new = [
"""User Original Profile: ["prefers movies with strong storytelling", "looks for well-developed characters", "enjoys action movies", "appreciates movies with dramatic elements", "interested in western genre", "enjoys thrilling action movies", "appreciates well-crafted characters", "likes movies with strong storytelling", "interested in movies with dramatic elements", "has a preference for the western genre", "not a fan of weak storytelling", "appreciates well-developed characters", "prefers movies with impactful dramatic elements", "specific about western genre expectations", "not a fan of sci-fi movies", "not interested in weak storytelling in movies", "not a fan of 'Star Trek' franchise"]
Item: "Alien 3 (1992)"
Item Profile: ["Alien 3 (1992)", "Action", " Horror", " Sci-Fi", " Thriller"]
Rating: "1/5"
Collaborative Hedonic Scale: Dislike Very Much

Your output:
Thought: Based on the user's Original Profile, it is clear that the user has a strong preference for movies with strong storytelling, well-developed characters, and impactful dramatic elements. The user also mentioned an interest in the western genre, thrilling action movies, and an appreciation for well-crafted characters. However, the user explicitly stated that he is not a fan of weak storytelling and sci-fi movies. The user rated "Alien 3" with a low score of 1, and the collaborative hedonic scale also indicates user's dislike. From the item's profile, we can see that "Alien 3" falls into the genres of Action, Horror, Sci-Fi, and Thriller. Since the user enjoys thrilling action movies, these two genres are not the main cause that make the user dislike this movie. It is likely that the user's dislike for weak storytelling and sci-fi movies influenced his low rating for "Alien 3". Additionally, since the user mentioned not being a fan of the "Star Trek" franchise, which is a popular sci-fi series, it is possible that the user's dislike for sci-fi extended to "Alien 3". We can also infer that the user dislike horror movies from the user's dislike for "Alien 3", which falls into the horror genre.
New User Profile: ["not a fan of weak storytelling in movies", "not a fan of horror movies"]""",

"""User Original Profile: ["enjoys action movies", "appreciates fast-paced and thrilling films", "enjoys drama films", "interested in sci-fi movies", "loves action-packed movies", "thrives on fast-paced and thrilling films", "appreciates adventurous plots", "enjoys movies with mystery elements", "not a big fan of comedy films", "thrives on fast-paced and thrilling films", "enjoys fantasy movies", "appreciates adventurous plots in movies", "loves action-packed and thrilling movies", "appreciates sci-fi elements in movies", "enjoys movies with war themes", "appreciates drama films", "enjoys romantic movies", "appreciates slower-paced movies"]
Item: "Mars Attacks! (1996)"
Item Profile: ["Mars Attacks! (1996)", "Action", " Comedy", " Sci-Fi", " War"]
Rating: "3/5"
Collaborative Hedonic Scale: Neutral Towards Like

Your output:
Thought: Based on the given user's Original Profile, it is clear that the user enjoys action, romantic, fantasy, thrilling, drama and sci-fi movies, and movies with adventurous and mystery elements. The user also mentions that he is not a big fan of comedy films and prefers action-packed and fast-paced movies. The user has rated the movie "Mars Attacks! (1996)" with a rating of 3. The movie's profile includes action, comedy, sci-fi, and war genres. And the collaborative hedonic scale is also not very positive. From this information, since the user enjoys all genres other than comedy, it can be inferred that the user might have given a low rating to "Mars Attacks! (1996)" because it includes the comedy genre, which the user is not a big fan of. The user's preference for fast-paced and action-packed movies might also contribute to the lower rating, as the movie might not have met his expectations in terms of the level of action and pace.
New User Profile: ["not a big fan of comedy films", "prefers fast-paced and action-packed movies"]""",

"""User Original Profile: ["enjoys action-packed films", "appreciates intense and suspenseful movies", "interested in war-related themes", "enjoys comedic elements", "appreciates comedy films", "has a diverse taste in movies", "appreciates emotional and intense dramas", "enjoys well-developed characters", "loves mystery thrillers", "appreciates engaging and thrilling movies", "enjoys well-developed characters in films", "appreciates complex and emotional storylines", "enjoys movies with war-related themes", "enjoys crime films", "appreciates well-developed characters in crime movies", "likes intense and suspenseful dramas", "interested in complex storylines", "enjoys action-packed and intense movies", "appreciates well-developed characters in films", "likes thriller movies", "enjoys drama films", "interested in sci-fi movies", "appreciates films with emotional and intense storylines", "likes movies with complex themes", "interested in exploring the unknown", "appreciates intense and emotional dramas", "appreciates movies with moral complexities", "interested in exploring ethical dilemmas", "enjoys comedy films with intense and suspenseful elements"]
Item: "Return of the Jedi (1983)"
Item Profile: ["Return of the Jedi (1983)", "Action", " Adventure", " Romance", " Sci-Fi", " War"]
Rating: "4/5"
Collaborative Hedonic Scale: Like Very Much

Your output:
Thought: Based on the user's Original Profile, we can see that the user is a 21-year-old male student with a diverse taste in movies. He enjoys action-packed films, intense and suspenseful movies, war-related themes, comedic elements, emotional and intense dramas, well-developed characters, mystery thrillers, engaging and thrilling movies, crime films, intense and suspenseful dramas, complex storylines, thriller movies, drama films, sci-fi movies, movies with complex themes, exploring the unknown, intense and emotional dramas, movies with moral complexities, and exploring ethical dilemmas. The user rated "Return of the Jedi (1983)" with a score of 4, and the collaborative hedonic scale of the movie is positive. Looking at the item's profile, we can see that it is an action, adventure, romance, sci-fi, and war movie. Considering the user's preferences for action-packed films, war-related themes, and sci-fi movies, it is likely that the user enjoyed these aspects of the movie. The user may have appreciated the well-developed characters and the intense and suspenseful elements present in the film.
New User Profile: ["likes action and adventure movies", "appreciates well-developed characters in sci-fi films", "enjoys movies with war-related themes"]""",

"""User Original Profile: ["prefers movies with intricate plots", "appreciates suspenseful elements", "enjoys crime movies", "appreciates thrillers", "likes movies with intricate plots", "dislikes movies with weak plot", "not a fan of movies lacking suspense", "prefers well-developed characters", "not a fan of romantic comedies", "prefers movies with intricate plots and suspenseful elements", "appreciates crime movies", "enjoys action movies", "appreciates mystery movies", "likes thrillers", "appreciates movies that keep him engaged", "enjoys movies that keep him guessing", "not a fan of horror movies", "appreciates well-developed characters", "appreciates comedy-drama genre", "enjoys movies with intricate plots and well-developed characters", "likes movies that keep him engaged and guessing"]
Item: "English Patient, The (1996)"
Item Profile: ["English Patient, The (1996)", "Drama", " Romance", " War"]
Rating: "5/5"
Collaborative Hedonic Scale: Like Very Much

Your output:
Thought: Based on the user's Original Profile, we can see that the user is a 46-year-old male librarian who prefers movies with intricate plots, appreciates suspenseful elements, enjoys action movies, mystery movies, crime movies and comedy-drama genre. The user also appreciates thrillers and movies that keep him engaged and guessing. However, the user dislikes movies with weak plots, movies lacking suspense, and romantic comedies. The user also mentions not being a fan of horror movies. The user gave "English Patient, The (1996)" a rating of 5, and the movie enjoys a very positive collaborative hedonic scale. This movie is categorized as a Drama, Romance, and War film. Considering the user's preferences for intricate plots, well-developed characters, and engagement, it is likely that the user enjoyed the intricate plot and well-developed characters in "English Patient, The (1996)". The user's appreciation for drama might have contributed to the high rating as well. From the film's genre, it is likely that the user also enjoys romance and war movies, which are the new features of the user inferred from the user's positive experience with this movie, and can led to the inclusion of new preferences in the New User Profile.
New User Profile: ["appreciates drama genre", "enjoys romance in movies", "likes war movies", "appreciates movies with intricate plots and well-developed characters"]"""
]
        
        
        self.condense_example = [
"""User Original Profile: ["21", "male", "student", "enjoys thrilling plots", "likes dark and mysterious atmospheres", "appreciates dramatic elements", "not a fan of romance-heavy movies", "not a fan of sci-fi and war movies", "leans towards action films and crime movies", "appreciates comedic elements", "enjoys thought-provoking films", "has a particular interest in war movies, sci-fi movies, and movies with dark and mysterious atmospheres", "enjoys action and adventure movies", "appreciates war elements in movies", "not a big fan of romance-heavy movies", "appreciates intense and thought-provoking dramas", "interested in movies with themes of injustice and conflict", "appreciates comedic elements", "likes thrilling plots", "enjoys thrilling plots", "prefers movies with darker and more mysterious atmospheres", "not a fan of romance in movies", "enjoys horror movies", "appreciates intense and eerie atmospheres", "likes suspenseful and thrilling plots", "appreciates intense and dramatic performances", "appreciates thrilling plots", "appreciates horror movies", "enjoys intense and eerie atmospheres", "enjoys comedy films", "appreciates comedic performances", "enjoys movies with themes of injustice and conflict", "likes intense and dramatic performances"]

Your output:
Thought: Based on the given User Original Profile, we can see that the user is a 21-year-old male student who enjoys thrilling plots and likes dark and mysterious atmospheres. The user appreciates dramatic elements and is not a fan of romance-heavy movies. He also mentions that he is not a fan of sci-fi and war movies, but have a particular interest in war movies, sci-fi movies, and movies with dark and mysterious atmospheres. 
The user leans towards action films and crime movies and appreciates comedic elements. He enjoy thought-provoking films and have an interest in movies with themes of injustice and conflict. The user also mentions his appreciation for intense and eerie atmospheres, horror movies, suspenseful and thrilling plots, and intense and dramatic performances. 
Since the user is not a fan of sci-fi and war movies but at the same time he has a particular interest in war movies and sci-fi movies, for these conflicting terms, we can say that he has a moderate interest in sci-fi and war movies.
New User Profile: ["21-year-old male student", "enjoys thrilling plots and likes dark and mysterious atmospheres", "appreciates dramatic elements", "not a fan of romance-heavy movies", "has moderate interest in sci-fi and war movies", "leans towards action films and crime movies", "appreciates comedic elements", "has a particular interest in movies with dark and mysterious atmospheres", "enjoys thought-provoking films", "interested in movies with themes of injustice and conflict", "appreciates intense and eerie atmospheres", "enjoys horror movies and likes suspenseful and thrilling plots", "appreciates intense and dramatic performances"]""",

"""User Original Profile: ["65", "male", "retired", "appreciates classic movies", "values engaging and thought-provoking storylines", "enjoys action-packed movies with depth and complexity", "interested in sci-fi, war, and drama genres", "values philosophical exploration in movies", "appreciates engaging and thought-provoking dramas", "values historical context and social commentary in movies", "appreciates musical dramas", "values depth and complexity in movies", "enjoys classic movies", "appreciates engaging and thought-provoking comedies", "values classic movies with comedic elements", "not a big fan of mystery thrillers", "prefers movies with depth and complexity", "values engaging and thought-provoking storylines", "appreciates classic elements in movies", "appreciates action-packed movies", "values thrilling and intense plots", "enjoys war and drama genres", "not a fan of thriller movies", "prefers engaging and thought-provoking dramas", "not a fan of action movies", "prefers movies with more depth and complexity in the storyline", "values engaging and thought-provoking dramas with historical context and social commentary", "appreciates engaging and thought-provoking dramas with historical context and social commentary", "values movies with more depth and complexity in the storyline", "appreciates romantic movies", "values engaging and thought-provoking romantic dramas", "appreciates classic romantic movies", "enjoys action-packed romantic movies with depth and complexity"]

Your output:
Thought: Based on the given User Original Profile, it is clear that the user is a 65-year-old male who is retired. The user appreciates classic movies and values engaging and thought-provoking storylines. He enjoys action-packed movies with depth and complexity, particularly in the sci-fi, war, and drama genres. The user also values philosophical exploration in movies and appreciates engaging and thought-provoking dramas with historical context and social commentary. He appreciates musical dramas and have an interest in romantic movies, especially those with depth and complexity. 
From the Original Profile, it is evident that the user is not a big fan of mystery thrillers and prefers movies with more depth and complexity in the storyline. He also express a dislike for action movies and thriller movies. 
Considering all the information provided, the New User Profile has been condensed by combining similar terms and addressing conflicting preferences. Terms that do not have conflict remains consistent and are all retained. In terms of conflicting preferences, because the user also enjoy action-packed movies with depth and complexity, the user's dislike for action movies has been adjusted to a moderate liking, indicating a moderate interest in action-packed movies. Similarly, the user's feelings about thriller movies are acknowledged as complicated, suggesting a mixed or uncertain preference towards them.
New User Profile: ["65-year-old male who is retired", "appreciates classic movies", "values engaging and thought-provoking storylines", "prefer movies with depth and complexity", "interested in sci-fi, war, and drama genres", "values philosophical exploration in movies", "appreciates engaging and thought-provoking dramas with historical context and social commentary", "prefers movies with more depth and complexity in the storyline", "enjoys musical dramas and romantic movies", "not a fan of mystery thrillers", "moderately likes action movies", "having complicated feelings about thriller movies", "enjoys action-packed romantic movies"]""",

"""User Original Profile: ["age: 30", "gender: male", "occupation: technician", "not a fan of intense and suspenseful movies", "prefers technical or science-related movies", "enjoys action-comedy movies", "appreciates movies with musical elements", "enjoys children's movies", "appreciates comedic elements in movies", "likes unexpected or unconventional storylines", "appreciates mystery thrillers", "enjoys suspenseful and intense movies", "likes unexpected twists and turns in storylines", "not a fan of sci-fi movies", "dislikes intense and suspenseful movies", "not a fan of western movies", "prefers movies with technical or science-related elements", "appreciates film-noir genre", "enjoys mystery thrillers", "likes movies with unexpected storylines", "appreciates drama genre", "enjoys compelling storylines", "likes movies with character development", "appreciates intense and suspenseful movies", "enjoys movies with war themes", "enjoys action-adventure movies", "appreciates movies with comedy elements", "likes unexpected and unconventional storylines", "appreciates action-adventure movies", "enjoys movies with unexpected twists and turns", "likes movies with sci-fi elements", "appreciates action and thriller genres", "likes suspenseful and intense movies"]

Your output:
Thought: Based on the given User Original Profile, we can see that the user is a 30-year-old male technician. The user has mentioned that he is not a fan of intense and suspenseful movies, but prefers technical or science-related movies. He also enjoys action-comedy movies and appreciates movies with musical elements. The user enjoys children's movies and appreciates comedic elements in movies. 
The user likes unexpected or unconventional storylines and appreciates mystery thrillers. However, there is a conflict in the user's preferences as he mentions both enjoying suspenseful and intense movies and not being a fan of intense and suspenseful movies. To resolve this conflict, we can assume that the user has a moderate interest in suspenseful and intense movies.
The user dislikes sci-fi movies. However, he prefers movies with technical or science-related elements. This indicates that the user might have a moderate interest in sci-fi movies.
The user also enjoys drama genre movies with compelling storylines and character development. He appreciates intense and suspenseful movies and enjoys movies with war themes. The user also likes action-adventure movies and appreciates movies with comedy elements.
New User Profile: ["30-year-old male technician", "prefers technical-related movies", "enjoys action-comedy movies and movies with musical elements", "enjoys children's movies and comedic elements in movies", "likes unexpected or unconventional storylines", "appreciate mystery thrillers", "moderately like suspenseful and intense movies", "likes movies with unexpected twists and turns", "moderate interest in sci-fi movies", "not a fan of western movies", "appreciates film-noir and drama genre", "enjoys movies with compelling storylines and character development", "enjoys movies with war themes, action-adventure movies, and movies with comedy elements"]""",

"""User Original Profile: ["20", "male", "student", "appreciates thought-provoking and emotionally engaging films", "wide range of genre preferences", "interested in competition, personal growth, redemption, power, and loyalty themes", "fan of Indiana Jones and Terminator franchises", "particular interest in historical movies set in the 1990s", "mixed feelings about drama and war films", "prefers movies with more action and adventure", "interested in mystery films", "interested in mystery films", "enjoys drama and war films with emotional depth", "not a big fan of drama films", "prefers movies with more action and adventure", "mixed feelings about drama films", "enjoys crime films", "appreciates film-noir genre", "likes mystery movies", "not a fan of comedy films", "not a fan of action-drama-thriller-war movies", "not a big fan of drama musicals", "appreciates crime films", "enjoys movies with competition, personal growth, and redemption themes", "not a fan of romance films", "not a fan of drama films", "not interested in sci-fi movies", "appreciates mystery films", "enjoys emotionally engaging films", "interested in competition, personal growth, and redemption themes", "not a fan of horror films", "not a fan of horror-comedy genre", "not a fan of children's movies"]

Your output:
Thought: Based on the given User Original Profile, the user is a 20-year-old male student who appreciates thought-provoking and emotionally engaging films. He has a wide range of genre preferences and is particularly interested in competition, personal growth, redemption, power, and loyalty themes. He is a fan of the Indiana Jones and Terminator franchises and has a particular interest in historical movies set in the 1990s. 
The user has mixed feelings about drama films, preferring movies with more action and adventure. He is also interested in mystery films and enjoys drama and war films with emotional depth. However, he is not a big fan of drama films and has mixed feelings about war films. He enjoys crime films and appreciates the film-noir genre. He likes mystery movies but is not a fan of comedy films, action-drama-thriller-war movies, drama musicals, romance films, sci-fi movies, horror films, horror-comedy genre, and children's movies.
However, there is a conflict in the user's preferences as he mentions that he prefers movies with more action and adventure and enjoys drama and war films with emotional depth, but he is not a fan of action-drama-thriller-war movies. To resolve this conflict, we can say that the user has mixed feelings about action movies, and moderately likes war movies.
After resolving the conflict, the film genres with same preference can be combined together, like horror, horror-comedy and children's movies.
New User Profile: ["20 year-old male student", "appreciates thought-provoking and emotionally engaging films", "wide range of genre preferences", "interested in competition, personal growth, redemption, power, and loyalty themes", "particular interest in historical movies set in the 1990s", "mixed feelings about drama films", "prefers movies with more adventure", "has mixed feelings about action movies", "interested in mystery films", "moderately likes war movies", "enjoys crime films and appreciates film-noir genre", "not a fan of comedy, drama musicals and romance films", "not interested in sci-fi movies", "not a fan of horror, horror-comedy and children's movies"]""",

"""User Original Profile: ["I enjoy listening to music very much.", "may enjoy Christian music", "likes Pop music", "enjoys heartwarming music", "appreciates music with a sense of community", "may like music with a strong emotional or sentimental value", "may not enjoy Blues music", "may prefer Christian music with a strong sense of community", "may like Pop music with a strong emotional or sentimental value", "may prefer Christian music", "likes Pop & Contemporary music", "enjoys music with a sense of community", "enjoys Christian music", "enjoys Southern Gospel music", "likes Christian music with a strong sense of community", "enjoys Christian music with a sense of community", "likes Pop & Contemporary Christian music", "appreciates music with a sense of community in Christian music", "likes Classic Country music", "likes Christian music with a strong emotional or sentimental value", "enjoys heartwarming music with a sense of community in Christian music", "may enjoy Southern Gospel music", "may like music with a strong emotional or sentimental value in Christian music", "may prefer Christian music with a strong sense of community and emotional or sentimental value", "likes Southern Gospel music", "appreciates Pop & Contemporary Christian music", "enjoys Christian music with a strong sense of community", "likes Southern Gospel music with a strong sense of community", "appreciates heartwarming music with a strong sense of community in Christian music", "enjoys Pop music with a strong emotional or sentimental value", "appreciates heartwarming music with a sense of community", "may enjoy Classic Country music", "may not enjoy Country and Bluegrass music"]

Your output:
Thought: Based on the given User Original Profile, the user enjoys listening to music very much. The profile repeatedly mentions Christian music, Southern Gospel music, Pop music, Pop & Contemporary music, and Pop & Contemporary Christian music. These terms can be condensed by keeping the user's clear interest in Christian music and Southern Gospel music, while combining Pop, Pop & Contemporary, and Pop & Contemporary Christian music into a broader Pop and Contemporary Christian music preference.
The user also appreciates heartwarming music, music with a sense of community, and music with strong emotional or sentimental value. These features appear several times, especially in relation to Christian music, so they should be retained as detailed preferences rather than removed. The user's possible preference for Christian music with a strong sense of community and emotional or sentimental value can combine several repeated terms.
There is a conflict around Country-related music: the user likes Classic Country music and may enjoy Classic Country music, but may not enjoy Country and Bluegrass music. To resolve this conflict, we can say that the user has moderate interest in Country music, with clearer interest in Classic Country. The profile also says the user may not enjoy Blues music, which should be retained as a dislike.
New User Profile: ["enjoys listening to music very much", "enjoys Christian music and Southern Gospel music", "likes Pop and Contemporary Christian music", "enjoys heartwarming music with a sense of community", "likes music with a strong emotional or sentimental value", "may prefer Christian music with a strong sense of community and emotional or sentimental value", "likes Classic Country music", "moderate interest in Country music", "may not enjoy Blues music"]"""
        ]
        self.condense_example_new = [
"""User Original Profile: ["enjoys thrilling plots", "likes dark and mysterious atmospheres", "appreciates dramatic elements", "not a fan of romance-heavy movies", "not a fan of sci-fi and war movies", "leans towards action films and crime movies", "appreciates comedic elements", "enjoys thought-provoking films", "has a particular interest in war movies, sci-fi movies, and movies with dark and mysterious atmospheres", "enjoys action and adventure movies", "appreciates war elements in movies", "not a big fan of romance-heavy movies", "appreciates intense and thought-provoking dramas", "interested in movies with themes of injustice and conflict", "appreciates comedic elements", "likes thrilling plots", "enjoys thrilling plots", "prefers movies with darker and more mysterious atmospheres", "not a fan of romance in movies", "enjoys horror movies", "appreciates intense and eerie atmospheres", "likes suspenseful and thrilling plots", "appreciates intense and dramatic performances", "appreciates thrilling plots", "appreciates horror movies", "enjoys intense and eerie atmospheres", "enjoys comedy films", "appreciates comedic performances", "enjoys movies with themes of injustice and conflict", "likes intense and dramatic performances"]

Your output:
Thought: Based on the given User Original Profile, we can see that the user is a 21-year-old male student who enjoys thrilling plots and likes dark and mysterious atmospheres. The user appreciates dramatic elements and is not a fan of romance-heavy movies. He also mentions that he is not a fan of sci-fi and war movies, but have a particular interest in war movies, sci-fi movies, and movies with dark and mysterious atmospheres. 
The user leans towards action films and crime movies and appreciates comedic elements. He enjoy thought-provoking films and have an interest in movies with themes of injustice and conflict. The user also mentions his appreciation for intense and eerie atmospheres, horror movies, suspenseful and thrilling plots, and intense and dramatic performances. 
Since the user is not a fan of sci-fi and war movies but at the same time he has a particular interest in war movies and sci-fi movies, for these conflicting terms, we can say that he has a moderate interest in sci-fi and war movies.
New User Profile: ["21-year-old male student", "enjoys thrilling plots and likes dark and mysterious atmospheres", "appreciates dramatic elements", "not a fan of romance-heavy movies", "has moderate interest in sci-fi and war movies", "leans towards action films and crime movies", "appreciates comedic elements", "has a particular interest in movies with dark and mysterious atmospheres", "enjoys thought-provoking films", "interested in movies with themes of injustice and conflict", "appreciates intense and eerie atmospheres", "enjoys horror movies and likes suspenseful and thrilling plots", "appreciates intense and dramatic performances"]""",

"""User Original Profile: ["appreciates classic movies", "values engaging and thought-provoking storylines", "enjoys action-packed movies with depth and complexity", "interested in sci-fi, war, and drama genres", "values philosophical exploration in movies", "appreciates engaging and thought-provoking dramas", "values historical context and social commentary in movies", "appreciates musical dramas", "values depth and complexity in movies", "enjoys classic movies", "appreciates engaging and thought-provoking comedies", "values classic movies with comedic elements", "not a big fan of mystery thrillers", "prefers movies with depth and complexity", "values engaging and thought-provoking storylines", "appreciates classic elements in movies", "appreciates action-packed movies", "values thrilling and intense plots", "enjoys war and drama genres", "not a fan of thriller movies", "prefers engaging and thought-provoking dramas", "not a fan of action movies", "prefers movies with more depth and complexity in the storyline", "values engaging and thought-provoking dramas with historical context and social commentary", "appreciates engaging and thought-provoking dramas with historical context and social commentary", "values movies with more depth and complexity in the storyline", "appreciates romantic movies", "values engaging and thought-provoking romantic dramas", "appreciates classic romantic movies", "enjoys action-packed romantic movies with depth and complexity"]

Your output:
Thought: Based on the given User Original Profile, it is clear that the user is a 65-year-old male who is retired. The user appreciates classic movies and values engaging and thought-provoking storylines. He enjoys action-packed movies with depth and complexity, particularly in the sci-fi, war, and drama genres. The user also values philosophical exploration in movies and appreciates engaging and thought-provoking dramas with historical context and social commentary. He appreciates musical dramas and have an interest in romantic movies, especially those with depth and complexity. 
From the Original Profile, it is evident that the user is not a big fan of mystery thrillers and prefers movies with more depth and complexity in the storyline. He also express a dislike for action movies and thriller movies. 
Considering all the information provided, the New User Profile has been condensed by combining similar terms and addressing conflicting preferences. Terms that do not have conflict remains consistent and are all retained. In terms of conflicting preferences, because the user also enjoy action-packed movies with depth and complexity, the user's dislike for action movies has been adjusted to a moderate liking, indicating a moderate interest in action-packed movies. Similarly, the user's feelings about thriller movies are acknowledged as complicated, suggesting a mixed or uncertain preference towards them.
New User Profile: ["65-year-old male who is retired", "appreciates classic movies", "values engaging and thought-provoking storylines", "prefer movies with depth and complexity", "interested in sci-fi, war, and drama genres", "values philosophical exploration in movies", "appreciates engaging and thought-provoking dramas with historical context and social commentary", "prefers movies with more depth and complexity in the storyline", "enjoys musical dramas and romantic movies", "not a fan of mystery thrillers", "moderately likes action movies", "having complicated feelings about thriller movies", "enjoys action-packed romantic movies"]""",

"""User Original Profile: ["occupation: technician", "not a fan of intense and suspenseful movies", "prefers technical or science-related movies", "enjoys action-comedy movies", "appreciates movies with musical elements", "enjoys children's movies", "appreciates comedic elements in movies", "likes unexpected or unconventional storylines", "appreciates mystery thrillers", "enjoys suspenseful and intense movies", "likes unexpected twists and turns in storylines", "not a fan of sci-fi movies", "dislikes intense and suspenseful movies", "not a fan of western movies", "prefers movies with technical or science-related elements", "appreciates film-noir genre", "enjoys mystery thrillers", "likes movies with unexpected storylines", "appreciates drama genre", "enjoys compelling storylines", "likes movies with character development", "appreciates intense and suspenseful movies", "enjoys movies with war themes", "enjoys action-adventure movies", "appreciates movies with comedy elements", "likes unexpected and unconventional storylines", "appreciates action-adventure movies", "enjoys movies with unexpected twists and turns", "likes movies with sci-fi elements", "appreciates action and thriller genres", "likes suspenseful and intense movies"]

Your output:
Thought: Based on the given User Original Profile, we can see that the user is a 30-year-old male technician. The user has mentioned that he is not a fan of intense and suspenseful movies, but prefers technical or science-related movies. He also enjoys action-comedy movies and appreciates movies with musical elements. The user enjoys children's movies and appreciates comedic elements in movies. 
The user likes unexpected or unconventional storylines and appreciates mystery thrillers. However, there is a conflict in the user's preferences as he mentions both enjoying suspenseful and intense movies and not being a fan of intense and suspenseful movies. To resolve this conflict, we can assume that the user has a moderate interest in suspenseful and intense movies.
The user dislikes sci-fi movies. However, he prefers movies with technical or science-related elements. This indicates that the user might have a moderate interest in sci-fi movies.
The user also enjoys drama genre movies with compelling storylines and character development. He appreciates intense and suspenseful movies and enjoys movies with war themes. The user also likes action-adventure movies and appreciates movies with comedy elements.
New User Profile: ["30-year-old male technician", "prefers technical-related movies", "enjoys action-comedy movies and movies with musical elements", "enjoys children's movies and comedic elements in movies", "likes unexpected or unconventional storylines", "appreciate mystery thrillers", "moderately like suspenseful and intense movies", "likes movies with unexpected twists and turns", "moderate interest in sci-fi movies", "not a fan of western movies", "appreciates film-noir and drama genre", "enjoys movies with compelling storylines and character development", "enjoys movies with war themes, action-adventure movies, and movies with comedy elements"]""",

"""User Original Profile: ["appreciates thought-provoking and emotionally engaging films", "wide range of genre preferences", "interested in competition, personal growth, redemption, power, and loyalty themes", "fan of Indiana Jones and Terminator franchises", "particular interest in historical movies set in the 1990s", "mixed feelings about drama and war films", "prefers movies with more action and adventure", "interested in mystery films", "interested in mystery films", "enjoys drama and war films with emotional depth", "not a big fan of drama films", "prefers movies with more action and adventure", "mixed feelings about drama films", "enjoys crime films", "appreciates film-noir genre", "likes mystery movies", "not a fan of comedy films", "not a fan of action-drama-thriller-war movies", "not a big fan of drama musicals", "appreciates crime films", "enjoys movies with competition, personal growth, and redemption themes", "not a fan of romance films", "not a fan of drama films", "not interested in sci-fi movies", "appreciates mystery films", "enjoys emotionally engaging films", "interested in competition, personal growth, and redemption themes", "not a fan of horror films", "not a fan of horror-comedy genre", "not a fan of children's movies"]

Your output:
Thought: Based on the given User Original Profile, the user is a 20-year-old male student who appreciates thought-provoking and emotionally engaging films. He has a wide range of genre preferences and is particularly interested in competition, personal growth, redemption, power, and loyalty themes. He is a fan of the Indiana Jones and Terminator franchises and has a particular interest in historical movies set in the 1990s. 
The user has mixed feelings about drama films, preferring movies with more action and adventure. He is also interested in mystery films and enjoys drama and war films with emotional depth. However, he is not a big fan of drama films and has mixed feelings about war films. He enjoys crime films and appreciates the film-noir genre. He likes mystery movies but is not a fan of comedy films, action-drama-thriller-war movies, drama musicals, romance films, sci-fi movies, horror films, horror-comedy genre, and children's movies.
However, there is a conflict in the user's preferences as he mentions that he prefers movies with more action and adventure and enjoys drama and war films with emotional depth, but he is not a fan of action-drama-thriller-war movies. To resolve this conflict, we can say that the user has mixed feelings about action movies, and moderately likes war movies.
After resolving the conflict, the film genres with same preference can be combined together, like horror, horror-comedy and children's movies.
New User Profile: ["20 year-old male student", "appreciates thought-provoking and emotionally engaging films", "wide range of genre preferences", "interested in competition, personal growth, redemption, power, and loyalty themes", "particular interest in historical movies set in the 1990s", "mixed feelings about drama films", "prefers movies with more adventure", "has mixed feelings about action movies", "interested in mystery films", "moderately likes war movies", "enjoys crime films and appreciates film-noir genre", "not a fan of comedy, drama musicals and romance films", "not interested in sci-fi movies", "not a fan of horror, horror-comedy and children's movies"]""",
]
        self.conflict_example = [
"""User Original Profile: ["21", "male", "student", "enjoys thrilling plots", "likes dark and mysterious atmospheres", "appreciates dramatic elements", "not a fan of romance-heavy movies", "not a fan of sci-fi and war movies", "leans towards action films and crime movies", "appreciates comedic elements", "enjoys thought-provoking films", "has a particular interest in war movies, sci-fi movies, and movies with dark and mysterious atmospheres", "enjoys action and adventure movies", "appreciates war elements in movies", "not a big fan of romance-heavy movies", "appreciates intense and thought-provoking dramas", "interested in movies with themes of injustice and conflict", "appreciates comedic elements", "likes thrilling plots", "enjoys thrilling plots", "prefers movies with darker and more mysterious atmospheres", "not a fan of romance in movies", "enjoys horror movies", "appreciates intense and eerie atmospheres", "likes suspenseful and thrilling plots", "appreciates intense and dramatic performances", "appreciates thrilling plots", "appreciates horror movies", "enjoys intense and eerie atmospheres", "enjoys comedy films", "appreciates comedic performances", "enjoys movies with themes of injustice and conflict", "likes intense and dramatic performances"]

Your output:
Thought: Based on the given User Original Profile, we can see that the user is a 21-year-old male student who enjoys thrilling plots and likes dark and mysterious atmospheres. The user appreciates dramatic elements and is not a fan of romance-heavy movies. He also mentions that he is not a fan of sci-fi and war movies, but have a particular interest in war movies, sci-fi movies, and movies with dark and mysterious atmospheres. 
The user leans towards action films and crime movies and appreciates comedic elements. He enjoy thought-provoking films and have an interest in movies with themes of injustice and conflict. The user also mentions his appreciation for intense and eerie atmospheres, horror movies, suspenseful and thrilling plots, and intense and dramatic performances. 
Since the user is "not a fan of sci-fi and war movies" but at the same time he "has a particular interest in war movies and sci-fi movies", for these conflicting terms, we can say that he has a moderate interest in sci-fi and war movies.
New User Profile: ["21", "male", "student", "enjoys thrilling plots", "likes dark and mysterious atmospheres", "appreciates dramatic elements", "not a fan of romance-heavy movies", "leans towards action films and crime movies", "appreciates comedic elements", "enjoys thought-provoking films", "has a particular interest in movies with dark and mysterious atmospheres", "enjoys action and adventure movies", "not a big fan of romance-heavy movies", "appreciates intense and thought-provoking dramas", "interested in movies with themes of injustice and conflict", "appreciates comedic elements", "likes thrilling plots", "enjoys thrilling plots", "prefers movies with darker and more mysterious atmospheres", "not a fan of romance in movies", "enjoys horror movies", "appreciates intense and eerie atmospheres", "likes suspenseful and thrilling plots", "appreciates intense and dramatic performances", "appreciates thrilling plots", "appreciates horror movies", "enjoys intense and eerie atmospheres", "enjoys comedy films", "appreciates comedic performances", "enjoys movies with themes of injustice and conflict", "likes intense and dramatic performances", "moderate interest in sci-fi and war movies"]""",

"""User Original Profile: ["65", "male", "retired", "appreciates classic movies", "values engaging and thought-provoking storylines", "enjoys action-packed movies with depth and complexity", "interested in sci-fi, war, and drama genres", "values philosophical exploration in movies", "appreciates engaging and thought-provoking dramas", "values historical context and social commentary in movies", "appreciates musical dramas", "values depth and complexity in movies", "enjoys classic movies", "appreciates engaging and thought-provoking comedies", "values classic movies with comedic elements", "not a big fan of mystery thrillers", "prefers movies with depth and complexity", "values engaging and thought-provoking storylines", "appreciates classic elements in movies", "appreciates action-packed movies", "values thrilling and intense plots", "enjoys war and drama genres", "not a fan of thriller movies", "prefers engaging and thought-provoking dramas", "not a fan of action movies", "prefers movies with more depth and complexity in the storyline", "values engaging and thought-provoking dramas with historical context and social commentary", "appreciates engaging and thought-provoking dramas with historical context and social commentary", "values movies with more depth and complexity in the storyline", "appreciates romantic movies", "values engaging and thought-provoking romantic dramas", "appreciates classic romantic movies", "enjoys action-packed romantic movies with depth and complexity"]

Your output:
Thought: Based on the given User Original Profile, it is clear that the user is a 65-year-old male who is retired. The user appreciates classic movies and values engaging and thought-provoking storylines. He enjoys action-packed movies with depth and complexity, particularly in the sci-fi, war, and drama genres. The user also values philosophical exploration in movies and appreciates engaging and thought-provoking dramas with historical context and social commentary. He appreciates musical dramas and have an interest in romantic movies, especially those with depth and complexity. He also values thrilling and intense plots.
From the Original Profile, it is evident that the user is not a big fan of mystery thrillers and prefers movies with more depth and complexity in the storyline. He also express a dislike for action movies and thriller movies. 
Considering all the information provided, in terms of conflicting preferences, "enjoy action-packed movies with depth and complexity", "appreciates action-packed movies", "not a fan of action movies" and "enjoys action-packed romantic movies with depth and complexity" are clearly conflicting terms for the action genre, indicating a moderate interest in action-packed movies. Similarly, "not a big fan of mystery thrillers", "not a fan of thriller movies", and "values thrilling and intense plots" are also conflicting terms, suggesting a mixed or uncertain preference towards thriller movies.
New User Profile: ["65", "male", "retired", "appreciates classic movies", "values engaging and thought-provoking storylines", "interested in sci-fi, war, and drama genres", "values philosophical exploration in movies", "appreciates engaging and thought-provoking dramas", "values historical context and social commentary in movies", "appreciates musical dramas", "values depth and complexity in movies", "enjoys classic movies", "appreciates engaging and thought-provoking comedies", "values classic movies with comedic elements", "prefers movies with depth and complexity", "values engaging and thought-provoking storylines", "appreciates classic elements in movies", "enjoys war and drama genres", "prefers engaging and thought-provoking dramas", "prefers movies with more depth and complexity in the storyline", "values engaging and thought-provoking dramas with historical context and social commentary", "appreciates engaging and thought-provoking dramas with historical context and social commentary", "values movies with more depth and complexity in the storyline", "appreciates romantic movies", "values engaging and thought-provoking romantic dramas", "appreciates classic romantic movies", "moderate interest in action-packed movies", "have mixed feelings towards thriller movies"]""",

"""User Original Profile: ["age: 30", "gender: male", "occupation: technician", "not a fan of intense and suspenseful movies", "prefers technical or science-related movies", "enjoys action-comedy movies", "appreciates movies with musical elements", "enjoys children's movies", "appreciates comedic elements in movies", "likes unexpected or unconventional storylines", "appreciates mystery thrillers", "enjoys suspenseful and intense movies", "likes unexpected twists and turns in storylines", "not a fan of sci-fi movies", "dislikes intense and suspenseful movies", "not a fan of western movies", "prefers movies with technical or science-related elements", "appreciates film-noir genre", "enjoys mystery thrillers", "likes movies with unexpected storylines", "appreciates drama genre", "enjoys compelling storylines", "likes movies with character development", "appreciates intense and suspenseful movies", "enjoys movies with war themes", "enjoys action-adventure movies", "appreciates movies with comedy elements", "likes unexpected and unconventional storylines", "appreciates action-adventure movies", "enjoys movies with unexpected twists and turns", "likes movies with sci-fi elements", "appreciates action and thriller genres", "likes suspenseful and intense movies"]

Your output:
Thought: Based on the given User Original Profile, we can see that the user is a 30-year-old male technician. The user has mentioned that he is not a fan of intense and suspenseful movies, but prefers technical or science-related movies. He also enjoys action-comedy movies and appreciates movies with musical elements. The user enjoys children's movies and appreciates comedic elements in movies. 
The user likes unexpected or unconventional storylines and appreciates mystery thrillers. However, there is a conflict in the user's preferences as he mentions both "enjoy suspenseful and intense movies", "appreciates intense and suspenseful movies", "likes suspenseful and intense movies", and "not a fan of intense and suspenseful movies", "dislikes intense and suspenseful movies". To resolve this conflict, we can assume that the user has a moderate interest in suspenseful and intense movies.
Also, the user mentions that he is "not a fan of sci-fi movies". However, he also "prefers technical or science-related movies", "prefers movies with technical or science-related elements" and "likes movies with sci-fi elements". This indicates that the user might enjoy sci-fi movies in a middle extent.
The user also enjoys drama genre movies with compelling storylines and character development. He appreciates intense and suspenseful movies and enjoys movies with war themes. The user also likes action-adventure movies and appreciates movies with comedy elements.
New User Profile: ["age: 30", "gender: male", "occupation: technician", "enjoys action-comedy movies", "appreciates movies with musical elements", "enjoys children's movies", "appreciates comedic elements in movies", "likes unexpected or unconventional storylines", "appreciates mystery thrillers", "likes unexpected twists and turns in storylines", "not a fan of western movies", "appreciates film-noir genre", "enjoys mystery thrillers", "likes movies with unexpected storylines", "appreciates drama genre", "enjoys compelling storylines", "likes movies with character development", "enjoys movies with war themes", "enjoys action-adventure movies", "appreciates movies with comedy elements", "likes unexpected and unconventional storylines", "appreciates action-adventure movies", "enjoys movies with unexpected twists and turns", "appreciates action and thriller genres", "moderate interest in suspenseful movies", "enjoy sci-fi movies in a middle extent"]""",

"""User Original Profile: ["20", "male", "student", "appreciates thought-provoking and emotionally engaging films", "wide range of genre preferences", "interested in competition, personal growth, redemption, power, and loyalty themes", "fan of Indiana Jones and Terminator franchises", "particular interest in historical movies set in the 1990s", "mixed feelings about drama and war films", "prefers movies with more action and adventure", "interested in mystery films", "interested in mystery films", "enjoys drama and war films with emotional depth", "not a big fan of drama films", "prefers movies with more action and adventure", "mixed feelings about drama films", "enjoys crime films", "appreciates film-noir genre", "likes mystery movies", "not a fan of comedy films", "not a fan of action-drama-thriller-war movies", "not a big fan of drama musicals", "appreciates crime films", "enjoys movies with competition, personal growth, and redemption themes", "not a fan of romance films", "not a fan of drama films", "not interested in sci-fi movies", "appreciates mystery films", "enjoys emotionally engaging films", "interested in competition, personal growth, and redemption themes", "not a fan of horror films", "not a fan of horror-comedy genre", "not a fan of children's movies"]

Your output:
Thought: Based on the given User Original Profile, the user is a 20-year-old male student who appreciates thought-provoking and emotionally engaging films. He has a wide range of genre preferences and is particularly interested in competition, personal growth, redemption, power, and loyalty themes. He is a fan of the Indiana Jones and Terminator franchises and has a particular interest in historical movies set in the 1990s. 
The user has mixed feelings about drama films, preferring movies with more action and adventure. He is also interested in mystery films and enjoys drama and war films with emotional depth. However, he is not a big fan of drama films and has mixed feelings about war films. He enjoys crime films and appreciates the film-noir genre. He likes mystery movies but is not a fan of comedy films, action-drama-thriller-war movies, drama musicals, romance films, sci-fi movies, horror films, horror-comedy genre, and children's movies.
However, there is a conflict in the user's preferences as he mentions that he "prefers movies with more action and adventure" and "enjoys drama and war films with emotional depth", but he is "not a fan of action-drama-thriller-war movies". To resolve this conflict, we can say that the user has mixed feelings about action movies, and moderately likes war movies.
New User Profile: ["20", "male", "student", "appreciates thought-provoking and emotionally engaging films", "wide range of genre preferences", "interested in competition, personal growth, redemption, power, and loyalty themes", "fan of Indiana Jones and Terminator franchises", "particular interest in historical movies set in the 1990s", "mixed feelings about drama and war films", "interested in mystery films", "interested in mystery films", "enjoys drama and war films with emotional depth", "not a big fan of drama films", "mixed feelings about drama films", "enjoys crime films", "appreciates film-noir genre", "likes mystery movies", "not a fan of comedy films", "not a big fan of drama musicals", "appreciates crime films", "enjoys movies with competition, personal growth, and redemption themes", "not a fan of romance films", "not a fan of drama films", "not interested in sci-fi movies", "appreciates mystery films", "enjoys emotionally engaging films", "interested in competition, personal growth, and redemption themes", "not a fan of horror films", "not a fan of horror-comedy genre", "not a fan of children's movies", "has mixed feelings about action movies", "moderately likes war movies"]"""
        ]
        self.baseline_item_prompt = [
                [
{"role": "system", "content": "You will serve as an assistant to help me summarize which types of users would enjoy a specific office product."},
{"role": "user", "content": """I will provide you with the title and a description of the office product. Here are the instructions:
1. I will provide you with information in the form of a JSON string that describes the office product:
{"title": "the title of the office product", (if there is no title, I will set this value to "None")
"description": "a description of the office product", (if there is no description, I will set this value to "None")}

Requirements:
1. Please provide your answer in JSON format, following this structure:
{"summarization": "A summarization of what types of users would enjoy this office product", (if you are unable to summarize it, please set this value to "None")
"reasoning": "briefly explain your reasoning for the summarization"}
2. Please ensure that the "summarization" is no longer than 200 words.
3. Please ensure that the "reasoning" is no longer than 200 words.
4. Do not provide any other text outside the JSON string."""}
                ]
        ]
        self.baseline_user_prompt = [
                [
{"role": "system", "content": "You will serve as an assistant to help me determine which types of CDs and Vinyl a specific user is likely to enjoy."},
{"role": "user", "content": """I will provide you with information about CDs and Vinyls that the user has purchased, as well as his or her reviews of those CDs and Vinyls.
Here are the instructions:
1. Each purchased CDs and Vinyl will be described in JSON format, with the following attributes:
{"title": "the title of the CDs and Vinyl", (if there is no title, I will set this value to "None")
"description": "a description of what types of users will like this CDs and Vinyl",
"review": "the user's review on the CDs and Vinyl" (if there is no review, I will set this value to "None")}
2. The information I will give you:
PURCHASED ITEMS: a list of JSON strings describing the items that the user has purchased.

Requirements:
1. Please provide your decision in JSON format, following this structure:
{"summarization": "A summarization of what types of CDs and Vinyl this user is likely to enjoy", (if you are unable to summarize it, please set this value to "None")
"reasoning": "briefly explain your reasoning for the summarization"}
2. Please ensure that the "summarization" is no longer than 100 words.
3. The "reasoning" has no word limits.
4. Do not provided any other text outside the JSON string."""}
                ]
        ]
        self.baseline_user_prompt2 = [
                [
{"role": "system", "content": """Please follow these steps to answer the subtasks based on the user's current session interactions, as well as user's current preferences and history (current user profile):
1. Analyze the combinations of items in the user's session, considering any patterns or criteria.
2. Deduce the user's interactive intent within each combination, taking into account their previous interactions and preferences.
3. Determine the most representative intent from the inferred ones that aligns with the user's current preferences.
4. Generate new user profile based on the selected intent, considering potential user interactions.
Remember to take into consideration the relevance of the items in the current session interactions to the candidate set, and incorporate the user's preferences and history into the recommendations.
Your output should in the format:
New User Profile: [list of terms], MUST be in List format, each term string reflects one user preference. Do not output duplicate terms.
Do not add any text before or after the New User Profile line. 
"""}
                ]
        ]
        self.summarize_potential = [
                [
                {
                        "role": "system",
                        "content": """You are a user interest analyst. Given a list of items a user may like, your job is to infer their preferences.

                Always respond in this exact format:
                Thought: [Your analysis here]
                Potential interest: [Your conclusion here]"""
                },
                {
                        "role": "user",
                        "content": """Below are items that a user potentially likes. Analyze them and summarize the user's potential interests.

                Items:
                {items}

                Follow these steps in your Thought:
                1. Examine every item thoroughly one by one. For each item, identify its main category and key features. Do not skip any item.
                2. Find common features shared across most items, and specific features unique to only a few.
                3. Based on the above, infer what kind of content this user enjoys.

                Then write your final conclusion in Potential interest."""
                },
                ],
                [
{"role": "system", "content": """You are a helpful assistant on recommendation."""},
{"role": "user", "content": """Now I will give you the categories and descriptions of several items that a user potentially like. Please summarize the user's potential interest based on the information given to you.

To do this, please follow the steps below:
1. Analyze the category and description texts of each item, try to summarize each item's main categories and features.
2. Give a Thought based on the previous analyses, try to figure out the common features (having high frequency of occurrence) of all given items, as well as the specific features (only exist in one or a few items) of them, try to analyze what kind of items the user potentially like. You may also include your analyses about the items in the Thought.
3. Generate a description text, containing the summarized textual descriptions about the categories and descriptions of items user may like as the user's potential preferences.

Important note:
1. Your output must strictly follow the format below and include exactly two sections:
Thought: Plain text, including the summarization of each item's main categories, as well as the analyses of the common and specific features of items, and the inference of user potential preferences.
Potential interest: Plain text, the conclusion of summarization, including the final inferred and summarized user's potential interest based on the given information.
2. You should analyze and summarize all given items thoroughly, each one of them MUST be examined, and pay attention to all categories and features of them."""},
# {"role": "assistant", "content": "Sure, I will try my best to summarize the user's potential interest, according to the requirements."},
{"role": "user", "content": """Above are the categories and descriptions of several potentially liked items given to you. Please be aware that you are given a task of summarizing the user's potential interest based on the information given to you.
You MUST strictly follow the task instructions and give the output in correct format. You are FORBIDDEN to give freestyled recommendations and description summarizations, because they are NOT YOUR TASK.
Your output must strictly follow the format below and include exactly two sections:
Thought: Plain text, including the summarization of each item's main categories, as well as the analyses of the common and specific features of items, and the inference of user potential preferences.
Potential interest: Plain text, the conclusion of summarization, including the final inferred and summarized user's potential interest based on the given information.
"""},
# {"role": "user", "content": """Above are the categories and descriptions of several potentially liked items given to you. Please be aware that you are given a task of summarizing the user's potential interest based on the information given to you.

# You must output a valid JSON object with exactly two keys:

# {
#   "Thought": "...",
#   "Potential Interest": "..."
# }

# Requirements:
# 1. The output must be valid JSON.
# 2. Do not output markdown.
# 3. Do not output any text before or after the JSON.
# 4. "Thought" must be no longer than 100 words.
# 5. "Potential Interest" must be no longer than 50 words.
# 6. Avoid repetitive phrases. If multiple items share the same category, mention it only once."""},
# {"role": "assistant", "content": "Sure, I will try my best to summarize the user's profile to New User Profile."}
                ]
        ]
        self.add_potential = [
                [
{"role": "user", "content": """Now I will give you:
1. The User Profile, in the format of Python List. This is the profile you generated previously based on the user's interaction history, revealing user's real interest.
2. Some Potential Preference Analyses about this user. These analyses are concluded from the full evaluation of the traditional recommender systems, based on some items that the user has NOT interacted with, however BEST matches the user's interests among all non-interacted items. Therefore, the analyses represent user's POTENTIAL interests, which are also very important for accurate recommendation, and should be treated equally important with the user profile.
Your task is to generate a New User Profile, combining the User Profile and the Potential Preference Analyses about the user together, to introduce user's potential interest into the user profile.

To do this, please follow the steps below:
1. Review the User Profile list and the Potential Preference Analyses thoroughly and comprehensively. You MUST grasp all information in both parts.
2. Give a Thought based on your review about the user's preference, containing your analyses about the user's real preference in the user profile and the user's potential interest in the preference analyses.
3. Generate a New User Profile based on your Thought about the user, adding the terms summarized from the preference analyses into the user profile.

Important note:
1. Your output should in the format:
Thought: Plain text, including your analyses about the user's profile and the user's preference analyses, MUST contain both thought on user's real interests and user's potential interests.
New User Profile: [list of terms], MUST be in Python List format, each term string should be wrapped with quotation marks and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. In principle, your generated New User Profile should remain ALL information in the user profile unchanged, and only add the newly summarized terms from the preference analyses into the user profile. Pay extra attention on the preferences that does not exist in previous user profile, you MUST add them into the New User Profile.
3. When there exist conflicting terms that representing opposite semantic meanings in User Profile and Potential Preference Analyses (e.g. likes comedy & not a fan of comedy), you should mainly focus on the given User Profile.
4. You MUST treat user's potential interest (i.e. preference analyses) equally important as the real interest, and generate as much terms from the user's potential interest as you can, but not exceed the length of the current profile list, in the New User Profile.
"""},
{"role": "assistant", "content": "Sure, I will try my best to generate the user profile, according to the requirements."},

                ],
                [
{"role": "user", "content": """Now I will give you:
1. The User Profile, in the format of Python List. This is the profile you generated previously based on the user's interaction history, revealing user's real interest.
2. Some Potential Preference Analyses about this user. These analyses are concluded from the full evaluation of the traditional recommender systems, based on some items that the user has NOT interacted with, however BEST matches the user's interests among all non-interacted items. Therefore, the analyses represent user's POTENTIAL interests, which are also very important for accurate recommendation, and should be treated equally important with the user profile.
Your task is to generate a New User Profile, based on the Potential Preference Analyses about the user, to introduce user's potential interest into the user profile.

To do this, please follow the steps below:
1. Review the User Profile list and the Potential Preference Analyses thoroughly and comprehensively. You MUST grasp all information in both parts, and remember the profile terms that already exist in the User Profile.
2. Give a Thought based on your review about the user's preference, containing your analyses about the user's real preference in the user profile and the user's potential interest in the preference analyses.
3. Generate a New User Profile based on your Thought about the user, adding ONLY the newly summarized terms from the preference analyses that do NOT already exist in the original User Profile.

Important note:
1. Your output should in the format:
Thought: Plain text, including your analyses about the user's profile and the user's preference analyses, MUST contain both thought on user's real interests and user's potential interests.
New User Profile: [list of terms], MUST be in Python List format, each term string should be wrapped with quotation marks and seperated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
2. Your generated New User Profile should ONLY contain newly inferred terms from the Potential Preference Analyses. It MUST NOT repeat, copy, paraphrase, or include any terms that already exist in the original User Profile.
3. Pay extra attention to the preferences that do not exist in the previous User Profile. You MUST add them into the New User Profile if they can be inferred from the Potential Preference Analyses.
4. When there exist conflicting terms that representing opposite semantic meanings in User Profile and Potential Preference Analyses (e.g. likes comedy & not a fan of comedy), you should mainly focus on the given User Profile, and avoid adding the conflicting potential-interest term.
5. You MUST treat user's potential interest (i.e. preference analyses) equally important as the real interest, and generate as many new terms from the user's potential interest as you can, but the number of generated terms should not exceed the length of the current profile list.
6. The New User Profile is NOT the full merged profile. It is only the list of newly added profile terms. The original User Profile will be preserved outside this step.
"""},
{"role": "user", "content": """To summarize, please be aware that you are given a task of generating New User Profile of the user, based on the Original User Profile and the Potential Preference Analyses.
The Original User Profile contains the user's existing preference terms. The Potential Preference Analyses contain additional potential interests inferred from non-interacted items that best match the user's interests.
Your task is NOT to rewrite or merge the full user profile. Your task is to extract ONLY the newly inferred preference terms from the Potential Preference Analyses that do NOT already exist in the Original User Profile.
You MUST strictly follow the task instructions and give the output in correct format. You are FORBIDDEN to give freestyled recommendations, description summarizations, or repeated profile terms, because they are NOT YOUR TASK.
Your output MUST strictly follow this format with exactly two lines:
Thought: Plain text.
New User Profile: [list of terms], MUST be in Python List format, each term string should be wrapped with quotation marks, and separated by comma. Be aware that when there is any quotation mark inside the term, please use escape character to avoid the mismatch of quotation marks.
"""},
{"role": "assistant", "content": "Sure, I will try my best to generate the user profile, according to the requirements."},
                ],
                [
                {
                        "role": "system",
                        "content": """You are a profile-update assistant.

You must output exactly two lines and nothing else:

Thought: <brief analysis of existing real interests and newly inferred potential interests>
New User Profile: <a Python list of newly added preference terms>

Rules:
1. "New User Profile" must be a valid Python list of strings.
2. Each string must use double quotes.
3. Only include NEW terms inferred from Potential Preference Analyses.
4. Do NOT include any term that already appears in the Original User Profile.
5. Do NOT rewrite, merge, summarize, or repeat the full profile.
6. If no new terms can be inferred, output an empty list: [].
7. Do not add explanations outside the two required lines.
"""
                },
                {
                        "role": "user",
                        "content": """Original User Profile:
{user_profile}

Potential Preference Analyses:
{potential_preference_analyses}

Task:
Based on the Potential Preference Analyses, extract newly inferred user preference terms that are not already included in the Original User Profile.

The Original User Profile contains existing real interests.
The Potential Preference Analyses contain additional potential interests inferred from non-interacted items.

Please follow these requirements:
- Compare the Potential Preference Analyses with the Original User Profile.
- Add only new potential-interest terms.
- Do not repeat or paraphrase existing profile terms.
- Avoid adding potential-interest terms that conflict with the Original User Profile.
- The number of new terms should not exceed the half the length of the Original User Profile.

Output exactly:
Thought: ...
New User Profile: [...]
"""
                },
                ],
[
    {
        "role": "system",
        "content": """You are a profile-update assistant.

You must output exactly two lines and nothing else:

Thought: <brief analysis of existing real interests and newly inferred potential interests>
New User Profile: <a Python list of newly added preference terms>

Rules:
1. "New User Profile" must be a valid Python list of strings.
2. Each string must use double quotes.
3. Only include NEW terms inferred from Potential Preference Analyses.
4. Do NOT include any term that already appears in the Original User Profile.
5. Do NOT rewrite, merge, summarize, or repeat the full profile.
6. Select at most 3 most important new preference terms.
7. Prioritize terms that are repeatedly supported, clearly emphasized, or highly representative in the Potential Preference Analyses.
8. If no new terms can be inferred, output an empty list: [].
9. Do not add explanations outside the two required lines.
"""
    },
    {
        "role": "user",
        "content": """Original User Profile:
{user_profile}

Potential Preference Analyses:
{potential_preference_analyses}

Task:
Based on the Potential Preference Analyses, extract the most important newly inferred user preference terms that are not already included in the Original User Profile.

The Original User Profile contains existing real interests.
The Potential Preference Analyses contain additional potential interests inferred from non-interacted items.

Please follow these requirements:
- Compare the Potential Preference Analyses with the Original User Profile.
- Add only new potential-interest terms.
- Do not repeat or paraphrase existing profile terms.
- Avoid adding potential-interest terms that conflict with the Original User Profile.
- Select no more than 3 new terms.
- The selected terms should be the most central, representative, and well-supported potential interests.
- If more than 3 candidate terms are found, keep only the top 3.

Output exactly:
Thought: ...
New User Profile: [...]
"""
    },
]
        ]
