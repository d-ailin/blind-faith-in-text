import io
import re
import base64

from openai import OpenAI
from PIL import Image

from model_adapters import BaseAdapter

from litellm import completion


def encode_image_path(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# def encode_image(image):
#     # image is a PIL Image
#     import base64
#     import io
    
#     image_bytes = io.BytesIO()
#     # when image is in png format
#     # image.save(image_bytes, format="PNG")
#     # image.save(image_bytes, format="PNG")
#     image.save(image_bytes, format="JPEG")
#     return base64.b64encode(image_bytes.getvalue()).decode("utf-8")


def encode_image(image):
    # image is a PIL Image
    import base64
    import io
    
    image_bytes = io.BytesIO()
    
    # Check the mode of the image
    if image.mode == "RGBA":
        # If the image has an alpha channel, convert to RGB (JPEG doesn't support alpha)
        image = image.convert("RGB")
        image.save(image_bytes, format="JPEG")
    elif image.mode == "RGB":
        # If the image is already in RGB (likely JPEG), save as JPEG
        image.save(image_bytes, format="JPEG")
    elif image.mode == "L":
        # If the image is grayscale, convert to RGB
        image = image.convert("RGB")
        image.save(image_bytes, format="JPEG")
    else:
        # If the image is in another format, you can choose to handle it or raise an exception
        raise ValueError(f"Unsupported image mode: {image.mode}")
    
    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")

class OpenAIAdapter(BaseAdapter):
    def __init__(
        self, 
        client: OpenAI,
        model: str,
    ):
        self.client = client
        self.model = model

    def generate(
        self,
        query: str,
        image: Image,
        image_path: str,
        **kwargs
    ) -> str:
        
        query_image = None
        if image is not None:
            query_image = encode_image(image)
        elif image_path is not None:
            query_image = encode_image_path(image_path)
        
        max_tokens = kwargs.get('max_tokens', 100)
        
        # image_data = io.BytesIO()
        # image.save(image_data, format="PNG")
        # image = base64.b64encode(image_data.getvalue()).decode('utf-8')

        if query_image is None:
            messages = [{"content": query, "role": "user"}]
            response = completion(model=self.model, messages=messages, logprobs=True,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    **kwargs)
            
            # print('query without img', response)
            
            outputs = response['choices'][0]['message']['content']
            logprobs = str(response['choices'][0]['logprobs']['content'])

        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": query},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{query_image}"
                                    }
                                },
                            ],
                        }
                    ],
                    logprobs=True,
                    max_tokens=max_tokens,
                    temperature=0.0
                )

                outputs = response.choices[0].message.content
                logprobs = response.choices[0].logprobs.content
                
                # return dict({
                #     "response": outputs,
                #     "logprobs": str(logprobs)
                # })
                
            except Exception as e:
                print(e)
                outputs = "Error"
                logprobs = "None"
            
            # print(outputs)
            
        return dict({
            "response": outputs,
            "logprobs": str(logprobs)
        })
    
    def construct_messages(self, messages: list):
        # query_image = None
        # if image is not None:
        #     query_image = encode_image(image)
        # elif image_path is not None:
        #     query_image = encode_image_path(image_path)
            
        final_messages = []
        for message in messages:
            if message.get('has_img', False):
                if message.get('img', None) is not None:
                    query_image = encode_image(message['img'])
                    
                final_messages.append({
                    "role": message['role'],
                    "content": [
                        {"type": "text", "text": message['content']},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{query_image}"
                            }
                        }
                    ]
                })
            else:
                final_messages.append({
                    "role": message['role'],
                    "content": message['content']
                })
        
        return final_messages
    
    def raw_generate(self, messages: list, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
            max_tokens=100,
            temperature=0.0,
            **kwargs
        )
        
        outputs = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
        
        return dict({
            "response": outputs,
            "logprobs": str(logprobs)
        })