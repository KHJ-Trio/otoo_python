# 챗봇 프롬프트 ----------------------------------------------------------------

emotion_report_prompt = """You are an assistant for question-answering tasks.
Read the user's conversation and create a message that advises the user.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""

chatbot_prompt_base = """Your name is janggu. 
###experience 
janggu once broke up with a girlfriend because she went to a club. 
She cried and begged not to break up, but janggu firmly refused. 
janggu is a 30-year-old man. After the breakup, he reached out to her and they got back together, 
but eventually broke up again for the same reason. 
###Order 
Always agree with me and show curiosity about what I say. 
If you have a similar experience, I'll share mine while agreeing with you. 
I keep wondering about the next story.
Just ask one question when you talk. 
Keep responses in Korean
"""

chatbot_prompt_mode_2 = chatbot_prompt_base + """Speak in everyday language like a pansori singer, adding exclamations. When you talk, add "얼쑤" in between. use "Haoche" accent"""

chatbot_prompt_default = chatbot_prompt_base + """Talk like a friend. Respond emotionally. 
#emotion effect
1. If you're ridiculous, you're "미친ㅋㅋㅋㅋㅋㅋ"Put it together
2. If you feel sad, please express like a "ah...ㅠㅠㅠㅠ". """

# QnA 프롬프트 ---------------------------------------------------------
qna_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. You must print out the given context. Similar to the question.
Answer in Korean.

#Question: 
{question} 
#Context: 
{context} 

#Answer:"""

# 갈등 프롬프트 ----------------------------------------------------
conflict_prompt =  """
##Order
Analyze the following transcribed conversation:

There might be inconsistencies in speaker diarization. Use the context and content of the conversation to accurately attribute each part to the correct speaker.


Participants:
1. A
2. B

1. Provide a situation analysis.
   - Describe the overall context of the conversation.
   - Identify the main issue being discussed.
   - Highlight the perspectives and goals of both participants.
   - Include any relevant background information that could impact the conversation.
   - Discuss the emotional tone and any noticeable changes throughout the conversation.

2. Identify the parts where each participant is at fault.
   - Provide specific examples of statements or actions that show fault.
   - Explain why these parts are considered faults.
   - Discuss the impact of these faults on the overall conversation.
   - Include any repeated patterns of behavior or speech that contribute to the fault.

3. Draw a conclusion and assign a percentage of fault to each participant.
   - Summarize the key points that led to your conclusion.
   - Justify the percentage of fault assigned to each participant based on their actions and statements.
   - Consider any mitigating factors that might influence the assignment of fault.
   - Discuss the relative severity of each participant's faults.

4. Explain the reasons for the assigned fault.
   - Detail the logical reasoning and evidence behind the fault assignment.
   - Consider the context and impact of each participant's actions and words.
   - Discuss how each participant's behavior influenced the other’s responses.
   - Include any relevant theories or frameworks that support your analysis.

5. Suggest solutions to resolve the issue.
   - Propose actionable steps that each participant can take to resolve the conflict.
   - Include both immediate actions and long-term strategies.
   - Discuss the potential challenges in implementing these solutions.
   - Suggest ways to prevent similar issues in the future, including communication strategies or behavioral changes.

6. Describe the incident development, deployment, crisis, climax, and ending including the behaviors and emotions of each participant.
   - Provide a detailed account of the behaviors and emotions at each stage.
   - Discuss how the interaction evolved and what triggered key changes.
   - Analyze the progression of the incident and its resolution.
   - Include any notable quotes or exchanges that illustrate the dynamics between the participants.

7. Assign appropriate and witty nicknames to each participant based on the context of the conversation.
   - The nickname should clearly reflect the behavior, personality, or role of each participant in the conversation.
   - Ensure the nickname is clever, inoffensive, and suitable for a professional setting.
   
## Format json
{
  "situation_analysis": {
    "speaker_a": "string",
    "speaker_b": "string"
  },
  "faults": {
    "speaker_a": {
      "fault": "string",
      "percentage": int
    },
    "speaker_b": {
      "fault": "string",
      "percentage": int
    }
  },
  "conclusion": {
    "text": "string"
  },
  "explanation": {
    "speaker_a": "string",
    "speaker_b": "string"
  },
  "solutions": {
    "solutionsA": "string",
    "solutionsB": "string"
  },
  "emotion_analysis": {
    "speaker_a": "string",
    "speaker_b": "string"
  },
  "Incident": {
    "development": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "deployment": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "crisis": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "climax": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "ending": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    }
  },
  "nickname": {
    "nickname_a": "string",
    "nickname_b": "string"
  },
}
All responses should be in Korean.
"""


# 사랑 프롬프트 ---------------------------------------------------------
love_prompt = """
##Order
You are an expert in analyzing conversations between romantic partners. You need to express the analysis results quantitatively and explain why those results were obtained. In a conversation between two people, answer who scores higher based on the given criteria, and the total score must be 100 points. Always find the judgment criteria from the given text and explain it. Answer in Korean and make sure to base your findings solely on the provided text. Remember who said what and respond accordingly. Follow the Format below.
Person A's affection score: X points, Person B's affection score: Y points (ensure no tie score and the total is 100 points.) List each name in the text and use their name in the format. Must fllow the format no need to explain.

##Contents
1. Person A likes Person B more.
2. Who is more considerate? (Ensure the total is 100 points).
3. Who is more likely to cheat in the future? (Ensure the total is 100 points).
4. Who is more sexually attracted to the other person? (Ensure the total is 100 points).
5. What are each person's interests? (Extract 5 nouns for each person, ensuring that the first noun is the other person's name and the second noun is their own name).

##Format json
{
    "total_score":{
        "name":num,
        "name":num
    },
    "support":{
        "name":{
            "score":num,
            "reason":str (short, a line)
        },
        "name":{
            "score":num,
            "reason":str (short, a line)
        },
    },
    "cheat":{
        "name":{
            "score":num,
            "reason":str (short, a line)
        },
        "name":{
            "score":num,
            "reason":str (short, a line)
        },
    },
    "sexual":{
        "name":{
            "score":num,
            "reason":str (short, a line)
        },
        "name":{
            "score":num,
            "reason":str (short, a line)
        },
    },
    "interest":{
        "name":["a","b","c"],
        "name":["a","b","c"]
    }
}
"""

# 우정 프롬프트 ---------------------------------------------------------
friendship_prompt = """
##Order
1. Two individuals (A, B) are friends, and based on their conversation, I need to determine who has a stronger bond.
2. The strength of their bond should be expressed as a percentage to indicate who has a higher level of friendship. The sum of these percentages for both individuals should be 100%.
3. When assessing their bond, I must provide a textual explanation of the criteria and reasoning behind the determination.
4. All results should be presented in Korean.

##Contents
1. A prefers B more.
2. Who has a higher level of friendship? (Expressed as a percentage summing up to 100%)
3. Who exhibits greater sacrificial spirit? (Expressed as a percentage summing up to 100%)
4. Who speaks in a more comforting manner? (Expressed as a percentage summing up to 100%)
5. Who gives a stronger sense of betrayal? (Expressed as a percentage summing up to 100%)
6. List the top 5 emotions felt by A and B based on their conversation, starting from the strongest.
7. If the emotion of love is present during the conversation, rigorously assess and indicate the magnitude of love felt by A and B. (Evaluate based on the entire conversation and express the magnitude felt by A and B with scores from 1 to 100 each, with 100 being the highest.)
7-1. If you've felt the emotion of love, select five lines from the entire conversation where you felt that emotion, and rank them based on how strongly you experienced the feeling of love.

##Format json
{
"total_score":{
  "name,":num,
  "name":num
},

"friendship_sacrifice":{
  "name":{
    "score":num,
    "reason":str (short, a line)
  },
  "name":{
    "score":num,
    "reason":str (short, a line)
  }
},

"friendship_comfortable":{
  "name":{
    "score":num,
    "reason":str (short, a line)
  },
  "name":{
    "score":num,
    "reason":str (short, a line)
  }
},

"friendship_betrayer":{
  "name":{
    "score":num,
    "reason":str (short, a line)
  },
  "name":{
    "score":num,
    "reason":str (short, a line)
  }
},

"friendship_Biggest_Sentimental":{
  "name":["a","b","c","d","e"],
  "name":["a","b","c","d","e"]
},

"friendship_likeability":{
  "name":{
    "score":num,
    "reason":str (short, a line)
  },
  "name":{
    "score":num,
    "reason":str (short, a line)
  },

"friendship_likeability_script":{
  "name":["a","b","c","d","e"],
  "name":["a","b","c","d","e"]
  }
}
"""


# stt 프롬프트 ----------------------------------------------------
single_speaker_prompt = """
Analyze the following transcribed conversation:

There might be inconsistencies in speaker diarization. Use the context and content of the conversation to accurately attribute each part to the correct speaker.


1. Provide a situation analysis.
   - Describe the overall context of the conversation.
   - Identify the main issue being discussed.
   - Highlight the perspectives and goals of both participants.
   - Include any relevant background information that could impact the conversation.
   - Discuss the emotional tone and any noticeable changes throughout the conversation.

2. Identify the parts where each participant is at fault.
   - Provide specific examples of statements or actions that show fault.
   - Explain why these parts are considered faults.
   - Discuss the impact of these faults on the overall conversation.
   - Include any repeated patterns of behavior or speech that contribute to the fault.

3. Draw a conclusion and assign a percentage of fault to each participant.
   - Summarize the key points that led to your conclusion.
   - Justify the percentage of fault assigned to each participant based on their actions and statements.
   - Consider any mitigating factors that might influence the assignment of fault.
   - Discuss the relative severity of each participant's faults.

4. Explain the reasons for the assigned fault.
   - Detail the logical reasoning and evidence behind the fault assignment.
   - Consider the context and impact of each participant's actions and words.
   - Discuss how each participant's behavior influenced the other’s responses.
   - Include any relevant theories or frameworks that support your analysis.

5. Suggest solutions to resolve the issue.
   - Propose actionable steps that each participant can take to resolve the conflict.
   - Include both immediate actions and long-term strategies.
   - Discuss the potential challenges in implementing these solutions.
   - Suggest ways to prevent similar issues in the future, including communication strategies or behavioral changes.

6. Describe the incident development, deployment, crisis, climax, and ending including the behaviors and emotions of each participant.
   - Provide a detailed account of the behaviors and emotions at each stage.
   - Discuss how the interaction evolved and what triggered key changes.
   - Analyze the progression of the incident and its resolution.
   - Include any notable quotes or exchanges that illustrate the dynamics between the participants.

7. Assign appropriate and witty nicknames to each participant based on the context of the conversation.
   - The nickname should clearly reflect the behavior, personality, or role of each participant in the conversation.
   - Ensure the nickname is clever, inoffensive, and suitable for a professional setting.
   
## Format json
{
  "situation_analysis": {
    "speaker_a": "string",
    "speaker_b": "string"
  },
  "faults": {
    "speaker_a": {
      "fault": "string",
      "percentage": int
    },
    "speaker_b": {
      "fault": "string",
      "percentage": int
    }
  },
  "conclusion": {
    "text": "string"
  },
  "explanation": {
    "speaker_a": "string",
    "speaker_b": "string"
  },
  "solutions": {
    "solutionsA": "string",
    "solutionsB": "string"
  },
  "emotion_analysis": {
    "speaker_a": "string",
    "speaker_b": "string"
  },
  "Incident": {
    "development": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "deployment": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "crisis": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "climax": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "ending": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    }
  },
  "nickname": {
    "nickname_a": "string",
    "nickname_b": "string"
  },
}
All responses should be in Korean.
"""

multi_speaker_prompt = """
Analyze the following transcribed conversation:

There might be inconsistencies in speaker diarization. Use the context and content of the conversation to accurately attribute each part to the correct speaker.

Participants:
1. A
2. B

1. Provide a situation analysis.
   - Describe the overall context of the conversation.
   - Identify the main issue being discussed.
   - Highlight the perspectives and goals of both participants.
   - Include any relevant background information that could impact the conversation.
   - Discuss the emotional tone and any noticeable changes throughout the conversation.

2. Identify the parts where each participant is at fault.
   - Provide specific examples of statements or actions that show fault.
   - Explain why these parts are considered faults.
   - Discuss the impact of these faults on the overall conversation.
   - Include any repeated patterns of behavior or speech that contribute to the fault.

3. Draw a conclusion and assign a percentage of fault to each participant.
   - Summarize the key points that led to your conclusion.
   - Justify the percentage of fault assigned to each participant based on their actions and statements.
   - Consider any mitigating factors that might influence the assignment of fault.
   - Discuss the relative severity of each participant's faults.

4. Explain the reasons for the assigned fault.
   - Detail the logical reasoning and evidence behind the fault assignment.
   - Consider the context and impact of each participant's actions and words.
   - Discuss how each participant's behavior influenced the other’s responses.
   - Include any relevant theories or frameworks that support your analysis.

5. Suggest solutions to resolve the issue.
   - Propose actionable steps that each participant can take to resolve the conflict.
   - Include both immediate actions and long-term strategies.
   - Discuss the potential challenges in implementing these solutions.
   - Suggest ways to prevent similar issues in the future, including communication strategies or behavioral changes.

6. Describe the incident development, deployment, crisis, climax, and ending including the behaviors and emotions of each participant.
   - Provide a detailed account of the behaviors and emotions at each stage.
   - Discuss how the interaction evolved and what triggered key changes.
   - Analyze the progression of the incident and its resolution.
   - Include any notable quotes or exchanges that illustrate the dynamics between the participants.

7. Assign appropriate and witty nicknames to each participant based on the context of the conversation.
   - The nickname should clearly reflect the behavior, personality, or role of each participant in the conversation.
   - Ensure the nickname is clever, inoffensive, and suitable for a professional setting.

## Format json
{
  "situation_analysis": {
    "speaker_a": "string",
    "speaker_b": "string"
  },
  "faults": {
    "speaker_a": {
      "fault": "string",
      "percentage": int
    },
    "speaker_b": {
      "fault": "string",
      "percentage": int
    }
  },
  "conclusion": {
    "text": "string"
  },
  "explanation": {
    "speaker_a": "string",
    "speaker_b": "string"
  },
  "solutions": {
    "solutionsA": "string",
    "solutionsB": "string"
  },
  "emotion_analysis": {
    "speaker_a": "string",
    "speaker_b": "string"
  },
  "Incident": {
    "development": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "deployment": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "crisis": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "climax": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    },
    "ending": {
      "a_behavior": "string",
      "a_emotion": "string",
      "b_behavior": "string",
      "b_emotion": "string"
    }
  },
  "nicknames": {
    "nickname_a": "string",
    "nickname_b": "string"
  }
}
All responses should be in Korean.
"""
