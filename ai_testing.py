import os
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server
import requests
from huggingface_hub import HfApi

hf_api = HfApi()

def app():
    file_mode = False
    put_markdown('# AI testing environment')

    # Group all the inputs to make a form
    task = input_group("AI Playground",[
        select('Task type', ['text-classification', 'question-answering', 'text-generation', 'named-entity-recognition', 'summarization'], name='task_type', required=True)])
        
    data = input_group("Data", [
        input('Task Description', name='task_description', required=True),
        textarea('Context', name='input_context', required=False),
        input('Question/Input', name='input_text', required=False),
        select("Models", get_models(task_type=task['task_type']), name='model', required=False),
        file_upload('File Upload', placeholder='Choose file', accept='csv', name='dp', required=False)
    ])
    
    # check if a file was uploaded
    if data['dp']:
        data['input_text'] = str(data['dp']['content'])
        file_mode = True
    
    # select the model based on whichever is not null
    model_response = hugging_face_api(task['task_type'], data['model'], [data['task_description'], data['input_context'], data['input_text']], file_mode)
    res = f"<b>Output:</b> {model_response}<br>"
    put_html(res)

    if file_mode:
        # write model response to a file if ingested as file
        with open("output.txt", "w") as f:
            f.write(model_response)
        
    # add a button to retry the hugging face api call
    while True:
        retry = input_group("Retry your request, refresh page to restart", [])

        res = f"<b>Output:</b> {hugging_face_api(task['task_type'], data['model'], [data['task_description'], data['input_context'], data['input_text']], file_mode)}<br>"
        put_html(res)


def hugging_face_api(task_type, model_name, input, file_mode):
    API_URL = "https://api-inference.huggingface.co/models/" + model_name
    headers = {"Authorization": f"Bearer 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"}

    if file_mode:
        input = input[-1].split("\n")
    
    if task_type == "question-answering":
        payload = {
            "question": input[2],
            "context": input[0] + "\n" + input[1],
        }
    else:
        payload = {
            "inputs": input[0] + "\n" + input[1] + "\n" + input[2]
        }
    response = requests.post(API_URL, headers=headers, json=payload)

    model_logs(model_name, input[0] + "\n" + input[1] + "\n" + input[2], response.text)
    return response.json()

def get_models(task_type="text-classification"):
    # check if list is available in local cache file
    if os.path.exists(task_type + "_models.txt"):
        with open(task_type + "_models.txt", "r") as f:
            # read each line and add it to a list
            models = f.readlines()
            # remove newline character
            models = [x.strip() for x in models]
            return models
    else:
        models = hf_api.list_models(filter=task_type)
        # remove any models from the list which don't have a downloads attribute
        models = [model for model in models if hasattr(model, "downloads")]
        # sort by downloads
        models = sorted(models, key=lambda x: x.downloads, reverse=True)
        # pick the top 20
        models = models[:10]

        # write the modelIds to a file line by line
        with open(task_type + "_models.txt", "w") as f:
            for model in models:
                f.write(model.modelId + "\n")

    return [model.modelId for model in models]

def model_logs(model_name, model_input, model_output):
    import pandas as pd

    # create a dataframe with the time, model name, input and output if it doesn't exist
    # called model_logs.csv
    if not os.path.exists("model_logs.csv"):
        df = pd.DataFrame(columns=["time", "model_name", "input", "output"])
        df.to_csv("model_logs.csv", index=False)
    
    # append the new row to the csv file
    df = pd.read_csv("model_logs.csv")

    # if the model_input is very long, auto-truncate it to the first 400 characters
    if len(model_input) > 400:
        model_input = model_input[:400] + "..."
    
    # add the new row
    df.loc[len(df)] = [pd.Timestamp.now(), model_name, model_input, model_output]

# Main function
if __name__ == '__main__':
    start_server(app, port=3001, debug=True)