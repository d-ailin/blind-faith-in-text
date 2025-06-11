import io
import re
import base64

# from openai import OpenAI
from PIL import Image
import anthropic

from model_adapters import BaseAdapter

from litellm import completion


def encode_image_path(image_path):
    import base64

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image(image):
    # image is a PIL Image
    import base64
    import io
    
    image_bytes = io.BytesIO()
    # when image is in png format
    # image.save(image_bytes, format="PNG")
    # image.save(image_bytes, format="PNG")
    image.save(image_bytes, format="JPEG")
    return base64.b64encode(image_bytes.getvalue()).decode("utf-8")


class ClaudeAdapter(BaseAdapter):
    def __init__(
        self, 
        model: str,
    ):
        self.client = anthropic.Anthropic()
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
        
        max_tokens = kwargs.get('max_tokens', 200)
        
        # image_data = io.BytesIO()
        # image.save(image_data, format="PNG")
        # image = base64.b64encode(image_data.getvalue()).decode('utf-8')

        if query_image is None:
            messages = [{"content": query, "role": "user"}]
            response = completion(model=self.model, messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    **kwargs)
            
            # print('query without img', response)
            
            outputs = response['choices'][0]['message']['content']
            # outputs = response['content'][0]['text']
            # logprobs = str(response['choices'][0]['logprobs']['content'])
            logprobs = 'None'

        else:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": query},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": query_image,
                                    },
                                }
                            ],
                        }
                    ],
                    # logprobs=True,
                    max_tokens=max_tokens,
                    temperature=0.0
                )
                # print(response)
                # outputs = response.choices[0].message.content
                outputs = response.content[0].text


                # logprobs = response.choices[0].logprobs.content
                logprobs = 'None'
                
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
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": query_image,
                            },
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