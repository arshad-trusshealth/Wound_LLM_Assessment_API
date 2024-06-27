from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import openai
import ollama
import base64
import time
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# import torch
from dotenv import load_dotenv
# from groq import Groq test commit
from flask_cors import CORS
load_dotenv()

api_key = os.getenv('OPEN_AI_API_KEY')

if not api_key:
    raise ValueError("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")

openai.api_key = api_key
# groq_api = ""

app = Flask(__name__)
CORS(app)
device_cuda = "cuda"
dev_cpu = "cpu"

def encode_image(image_bytes):
    # image_bytes = uploaded_file.read()
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    return encoded

def get_wound_characteristics_local(image_bytes, model_llm='phi3:3.8b'):
    ollama.pull(model_llm)
    llm = ChatOllama(model=model_llm, temperature=0, device = dev_cpu)
    base64_image = encode_image(image_bytes)
    template = """
                You are a helpful clinical assistant.
                You will be provided with an image of a wound.
                The image is {image}.
                Analyze and identify the specific type of wound, and potential risk of infection.
                Additionally, segment the image and based on it have an estimate of wound area, Erythema Redness of the wound and risk score of wound (scale of 1 - 10).
                Provide a response in a clear and concise manner which explains the specific type of wound, the Erythema Redness, the risk of infection and the risk score of wound overall.
                No need to give any disclaimer and recommendations.   
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain_setup = {"image": base64_image}
    chain = prompt | llm | StrOutputParser()
    resp_wound_char = chain.invoke(chain_setup)
    return resp_wound_char
    # client = Groq()
    # completion = client.chat.completions.create(
    #     model="gemma-7b-it",
    #     messages=[
    #         {
    #             "role": "system",
    #             "content": 
    #                 """
    #                     You are a helpful clinical assistant.
    #                     You will be provided with an image of a wound.
    #                     Analyze and identify the specific type of wound, and potential risk of infection.
    #                     Additionally, segment the image and based on it have an estimate of wound area, Erythema Redness of the wound and risk score of wound (scale of 1 - 10).
    #                     Provide a response in a clear and concise manner which explains the specific type of wound, the Erythema Redness, the risk of infection and the risk score of wound overall.
    #                     No need to give any disclaimer and recommendations.
    #                 """
    #         },
    #         {
    #             "role": "user",
    #             "content": f"![wound image](data:image/jpeg;base64,{base64_image})"
    #         }
    #     ],
    #     temperature=1,
    #     max_tokens=1024,
    #     top_p=1,
    #     stream=True,
    #     stop=None,
    # )

    # # Print the response
    # for chunk in completion:
    #     print(chunk.choices[0].delta.content or "", end="")


def get_wound_characteristics_gpt(image_bytes, MODEL="gpt-4-turbo"):
    client = openai.OpenAI(api_key = api_key)
    base64_image = encode_image(image_bytes)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content":
                """
                You are a helpful clinical assistant.
                You will be provided with an image of a wound.
                Analyze and identify the specific type of wound, and potential risk of infection.
                Additionally, segment the image to and based on it have an estimate of wound area, Erythema Redness of the wound and risk score of wound (scale of 1 - 10).
                Provide a response in a clear and concise manner which explains the specific type of wound, the Erythema Redness, the risk of infection and the risk score of wound overall.
                No need to give any disclaimer and any recommendations.
                """},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{base64_image}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    resp_wound_char = response.choices[0].message.content
    return resp_wound_char

# @app.route("/")
# def index():
#     return render_template("index.html")

@app.route('/analyze_results', methods=['GET', 'POST'])
def analyze_wound():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        image_bytes = file.read()
    llm_type = request.form.get('llm_type', 'API')
    if llm_type == 'API':
        result = get_wound_characteristics_gpt(image_bytes)
    elif llm_type == 'Local':
        result = get_wound_characteristics_local(image_bytes)
    else:
        return jsonify({"error": "Invalid LLM type"}), 400
    print(result)
    return jsonify({"image_analysis": result})

if __name__ == '__main__':
    app.run(debug = True , host='0.0.0.0' , port=8081)
