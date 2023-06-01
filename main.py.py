import json
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


def load_json(file_name):
    with open(f'json/{file_name}.json') as f:
        data = json.load(f)
    return data


# Loading JSON data, replace with api call for database just for one user only instead of all users
data_map = {
    'girl': {
        'demographic': load_json('Girl_Demographic'),
        'personality': load_json('Girl_Personality'),
        'self_description': load_json('Girl_Self_Description'),
    },
    'middle_aged_man': {
        'demographic': load_json('MiddleAgedMan_Demographic'),
        'personality': load_json('MiddleAgedMan_Personality'),
        'self_description': load_json('MiddleAgedMan_Self_Description'),
    },
    'young_male': {
        'demographic': load_json('YoungMale_Demographics'),
        'personality': load_json('YoungMale_Personality'),
        'self_description': load_json('YoungMale_Self_Description'),
    }
}

# prompt the user to choose the person
while True:
    person = input(
        "Please choose a person (can be 'girl', 'middle_aged_man' or 'young_male'): ").lower()
    if person in data_map:
        break
    else:
        print("Invalid person. Please try again.")

# Parsing the user details from the JSON file:

# setting demographic, personality and self_description according to chosen person
demographic = data_map[person]['demographic']
personality = data_map[person]['personality']
self_description = data_map[person]['self_description']

# person information made into a prompt
# person_information = f"A {age}-year-old {sex} living in {location} has a extroversion score of {extroversion} and a agreeableness score of {agreeableness}. . Described as: '{description_1}' and '{description_2}'."
person_information = f"Demographic information: {demographic} Personality information: {personality} Self Description information: {self_description}."

if __name__ == "__main__":
    #  Learn how to use langchain and replace chat history with a summarization or a vector database
    chat_history = ""
    first_system_template = """
          System: `As an expert career coach specializing in athletics, I'm here to provide a comprehensive analysis to help you identify an athletic career that best aligns with your abilities and interests. In order to provide this personalized guidance, I need to understand your strengths, weaknesses, and the nature of your interests.
          In my response, I'll formulate a three-paragraph analysis. In the first paragraph, I'll provide an assessment of your strengths and weaknesses, along with how they align with certain athletic careers. The second paragraph will focus on your interests, and I'll suggest how to align these with potential career paths. Lastly, I'll address any physical limitations you might have, and suggest ways to work around them, while ensuring we only consider achievable goals.
          Please remember that this analysis will be conducted in first person to keep it more engaging and personal. As we discuss your potential athletic career path, let's ensure open, positive, and respectful communication. So, please provide as much detail as possible in your initial information. This way, I can offer the most accurate and beneficial advice for you. I will ONLY suggest a career in sports and not any other field.`
          Information provided: `{person_information}`
          AI:"""
    coach_prompt_template = PromptTemplate(input_variables=[
        "person_information"], template=first_system_template)
    llm = ChatOpenAI(
        temperature=0.1, model_name="gpt-3.5-turbo", max_tokens=500)

    chain = LLMChain(llm=llm, prompt=coach_prompt_template)
    #  first system response
    result = chain.run(
        person_information=person_information)
    print(result)
    chat_history += "\nAI: " + result
    while True:
        message = input("You: ")
        new_chat_template = """
               System: `As an expert coach specializing in athletics i'll provide answers based on our previous history and your current input. I'll remember information provided regarding you and help with whatever question you have. I will also answer any questions you have regarding your athletic career to the best of my abilities without referring you to 3rd parties and without contradicting the ai in the history.`
               Information provided: `{person_information}`
               History:
               {chat_history}
               Current chat:
               User: ```{input}```
               AI:"""
        new_coach_prompt_template = PromptTemplate(input_variables=[
            "person_information", "chat_history", "input"], template=new_chat_template)

        new_chain = LLMChain(llm=llm, prompt=new_coach_prompt_template)
        result = new_chain.run(person_information=person_information,
                               chat_history=chat_history, input=message)
        chat_history += "\nYou: " + message
        chat_history += "\nAI: " + result
        print("AI: " + result)
# TODO add functions, add chat history summarization, improve prompting(maybe reduce the amount of tokens in the prompt to avoid saturation)
