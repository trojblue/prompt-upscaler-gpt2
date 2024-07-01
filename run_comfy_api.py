
# WORK_DIR = ""

# %cd $WORK_DIR

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
from PIL import Image
import io

import unibox as ub

import random
from PIL import Image

class WorkflowExecutor:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, prompt):
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"http://{self.server_address}/prompt", data=data, method="POST")
        return json.loads(urllib.request.urlopen(req).read())

    def get_image(self, filename, subfolder, folder_type):
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(f"http://{self.server_address}/view?{url_values}") as response:
            return response.read()

    def get_history(self, prompt_id):
        with urllib.request.urlopen(f"http://{self.server_address}/history/{prompt_id}") as response:
            return json.loads(response.read())

    def get_images(self, ws, prompt):
        prompt_id = self.queue_prompt(prompt)['prompt_id']
        output_images = {}
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executing':
                    data = message['data']
                    if data['node'] is None and data['prompt_id'] == prompt_id:
                        break  # Execution is done
            else:
                continue  # Previews are binary data

        history = self.get_history(prompt_id)[prompt_id]
        for o in history['outputs']:
            for node_id in history['outputs']:
                node_output = history['outputs'][node_id]
                if 'images' in node_output:
                    images_output = []
                    for image in node_output['images']:
                        image_data = self.get_image(image['filename'], image['subfolder'], image['type'])
                        pil_image = Image.open(io.BytesIO(image_data))
                        images_output.append(pil_image)
                    output_images[node_id] = images_output

        return output_images

    def run_workflow(self, workflow_json:dict):
        ws = websocket.WebSocket()
        ws.connect(f"ws://{self.server_address}/ws?clientId={self.client_id}")
        images = self.get_images(ws, workflow_json)
        return images


def modify_workflow_keys(
    workflow:dict, 
    pos:str, 
    neg:str="lowres, worst quality, displeasing, bad quality, bad anatomy, text, error, extra digit, cropped, average quality",
    model:str="bxl-v4c-stepfix/checkpoint-e4_s19000.safetensors",
    seed:int=-1,
    batch_size:int=4
):
    if seed == -1:
        seed = random.randint(0, 9000000)

    if '6' in workflow and workflow['6']['class_type'] == 'CLIPTextEncode':
        workflow['6']['inputs']['text'] = pos
    
    if '7' in workflow and workflow['7']['class_type'] == 'CLIPTextEncode':
        workflow['7']['inputs']['text'] = neg

    if '101' in workflow and workflow['101']['class_type'] == 'CheckpointLoaderSimple':
        workflow['101']['inputs']['ckpt_name'] = model

    if '34:3' in workflow and workflow['34:3']['class_type'] == 'KSampler':
        workflow['34:3']['inputs']['seed'] = seed

    if '34:0' in workflow and workflow['34:0']['class_type'] == 'EmptyLatentImage':
        workflow['34:0']['inputs']['batch_size'] = batch_size

    return workflow


def concatenate_images_horizontally(images, max_height=1024):
    """
    Concatenates a list of PIL.Image objects horizontally, ensuring no image exceeds a specified maximum height.

    :param images: List of PIL.Image objects.
    :param max_height: Maximum height for any image in the list. Images taller than this will be resized proportionally.
    :return: A single PIL.Image object resulting from the horizontal concatenation.
    """
    if not images:
        print("No images to concatenate.")
        return None

    resized_images = []

    # Resize images if necessary to ensure no image exceeds the max height
    for image in images:
        if image.height > max_height:
            aspect_ratio = image.width / image.height
            new_width = int(aspect_ratio * max_height)
            resized_image = image.resize((new_width, max_height))
            resized_images.append(resized_image)
        else:
            resized_images.append(image)

    # Determine the total width and the maximum height of resized images
    total_width = sum(image.width for image in resized_images)
    max_height = max(image.height for image in resized_images)

    # Create a new image with the appropriate dimensions
    concatenated_image = Image.new('RGB', (total_width, max_height))

    # Paste each resized image into the new image
    x_offset = 0
    for image in resized_images:
        concatenated_image.paste(image, (x_offset, 0))
        x_offset += image.width

    return concatenated_image


DEFAULT_NEG = "lowres, worst quality, displeasing, bad quality, bad anatomy, text, error, extra digit, cropped, average quality"
DEFAULT_MODEL = "bxl-v4c-stepfix/checkpoint-e5_s19000.safetensors"

def run_workflow(pos:str, neg:str="", model:str="", seed:int=-1, batch_size:int=4):

    if not neg:
        neg = DEFAULT_NEG
    if not model:
        model = DEFAULT_MODEL

    executor = WorkflowExecutor()
    workflow = ub.loads("lworkflow_api.json", debug_print=False)
    workflow = modify_workflow_keys(workflow, pos=pos, neg=neg, model=model, seed=seed, batch_size=batch_size)

    # run the workflow
    workflow_result = executor.run_workflow(workflow)
    images_list = workflow_result["103"]

    # return images
    return images_list