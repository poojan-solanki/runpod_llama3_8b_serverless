import runpod
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

prompt = """
You will analyze data from two sources to provide a final verdict on the scene:  
                1. **Llama-Vision Data**: High-level descriptive overview.  
                2. **YOLO11x-Pose Detection Data**: Detailed pose and bounding box data for individuals in the scene. \n
                "I will give you context of your previous responce (in case of first iteration it might be empty). In available you should strictly mention the changes that have happened from your previous responce **If none end the responce**." 

                ### Your Tasks:  
                1. Compare both datasets and resolve discrepancies, highlighting key findings.  
                2. For each person detected, determine:  
                - Activity (sitting, standing, walking).  
                - Proximity to others (in a crowd or not).  
                3. Provide a **final integrated verdict** of the scene:  
                - Number of people, their activities, interactions, and any notable details.  
                - Environmental context and any unusual behaviors.  
                - Summary of key insights prioritized by importance. 
                #  \n\n#### Comparison and Discrepancy Resolution ** and strictly focus on giving final verdict
"""

def video_description(input):
    llama_vision_data = input.get('llama_vision_data')
    yolo_data = input.get('yolo_data')

    # Placeholder for a task; replace with image or text generation logic as needed
    messages = [
        {"role": "system", "content": "You are a accurate video scene analyser who likes to conbine multiple source of information and give final verdict!"},
        {"role": "user", "content": f"{prompt}"
        f"\n\nHere is the data for your analysis: Take file name into consideration too (especially elapsed time) it might be helpful. (elapsed means elapsed time of video)\n\n"
            f"### llama-vision data:\n{llama_vision_data}\n\n"
            f"### yolo11x-pose detection data:\n{yolo_data}\n\n"
            "Now, combine these datasets and provide the best possible output."
            "## at last summarise the output based on the importance you think is best"
        },
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    result = outputs[0]["generated_text"][-1]
    return result.get("content")

def generate_smart_name(data):
    messages = [
        {"role": "system", "content": "You are an AI assistant skilled in generating smart and creative video titles based on the content of the video. Your task is to create a concise and catchy title, no longer than 7 words, that summarizes the key theme of the video."},
        {"role": "user", "content": "Please generate a single, smart, and engaging video title based on the following content. The title should capture the main idea and be no more than 7 words. Here is the video content:\n\n" 
                                    f"{data}"
        },
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=20,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    result = outputs[0]["generated_text"][-1]
    return result.get("content")


def answer_query(user_query, best_matched_embeddings):
    messages = [
        {"role": "system", "content": "You are a video chatbot that answers only sensible user queries based on provided textual data extracted from video frames. Ignore garbage or nonsensical input. Extract and deliver clear, necessary information without technical terms (e.g., avoid jargon like 'YOLO model'). If relevant, include activity duration with elapsed time from the video, keeping responses short and to the point."},
        {"role": "user", "content": f"User query: {user_query}"
                            f"Find information from this daat:\n{best_matched_embeddings}"
        },
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    result = outputs[0]["generated_text"][-1]
    return result.get("content")

def handler(event):
    input = event['input']

    if input.get('task') == "combine_data":
        return video_description(input)
    elif input.get('task') == "smart_name":
        return generate_smart_name(input.get('llama_res'))
    elif input.get('task') == "query":
        return answer_query(input.get('user_query'),input.get("best_matched_embeddings"))
    else:
        return "No task defined for this purpose"

    

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})