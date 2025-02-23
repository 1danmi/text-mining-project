SPAM_EXAMPLE = """For example:
Text: "{text}"

SPAM or HAM: {label}
"""

SPAM_SIMPLE_PROMPT_TEMPLATE = """This is an overall spam classifier for input emails and messages.
Categorize the input as a "spam" or "ham".
{examples}
Text: "{dataset_text}"

SPAM or HAM:"""


SPAM_PROMPT_TEMPLATE = """This is an overall spam classifier for input emails and messages.
Present CLUES (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references) that support the spam or ham determination of the input.
Next, deduce the diagnostic REASONING process from premises (i.e., clues, input) that support the sentiment determination.
Finally, based on clues, reasoning and the input, categorize the overall SENTIMENT of input as a "spam" or "ham".
Limit your answer 100 tokens or less.
{examples}
Text: "{dataset_text}"

SPAM or HAM:"""

IMDB_EXAMPLE = """For example:
Text: "{text}"

SENTIMENT: {label}
"""

IMDB_SIMPLE_PROMPT_TEMPLATE = """This is an overall sentiment classifier for movie reviews. Classify the overall SENTIMENT of the INPUT as Positive or
Negative.
{examples}
Text: "{dataset_text}"

SENTIMENT:"""

IMDB_PROMPT_TEMPLATE = """This is an overall sentiment classifier for movie reviews
Present CLUES (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references) that support the spam or ham determination of the input.
Next, deduce the diagnostic REASONING process from premises (i.e., clues, input) that support the sentiment determination.
Finally, based on clues, reasoning and the input, categorize the overall SENTIMENT of input as a "negative" or "positive".
Limit your answer 100 tokens or less.
{examples}
Text: "{dataset_text}"

SENTIMENT:"""

AMAZON_EXAMPLE = """For example:
Text: {text}

SENTIMENT: {label}
"""


AMAZON_SIMPLE_PROMPT_TEMPLATE = """This is an overall sentiment classifier for amazon products reviews. Classify the overall SENTIMENT of the INPUT as Positive or
Negative.
{examples}
Text: "{dataset_text}"
SENTIMENT:"""

AMAZON_PROMPT_TEMPLATE = """This is an overall sentiment classifier for amazon products reviews
Present CLUES (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references) that support the spam or ham determination of the input.
Next, deduce the diagnostic REASONING process from premises (i.e., clues, input) that support the sentiment determination.
Finally, based on clues, reasoning and the input, categorize the overall SENTIMENT of input as a "negative" or "positive".
Limit your answer 100 tokens or less.
{examples}
Text: "{dataset_text}"

SENTIMENT:"""


R8_EXAMPLE = """For example:
Text: {text}

TOPIC: {label}
"""


R8_SIMPLE_PROMPT_TEMPLATE = """This is an overall text classifier for news article topics. 
Categorize topic of the article to one of the following topics: "earnings", "acquisitions", "money", "oil", "grain", "trade", "monetary", "wheat", "shipping". 
{examples}
Text: "{dataset_text}"

TOPIC:"""


R8_PROMPT_TEMPLATE = """This is an overall text classifier for news article topics. 
Present CLUES (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references) that support the topic classfication of the input.
Next, deduce the diagnostic REASONING process from premises (i.e., clues, input) that support the sentiment determination.
Finally, based on clues, reasoning and the input, categorize topic of the article to one of the following topics: "earnings", "acquisitions", "money", "oil", "grain", "trade", "monetary", "wheat", "shipping". 
Limit your answer 100 tokens or less.
{examples}
Text: "{dataset_text}"

TOPIC:"""

