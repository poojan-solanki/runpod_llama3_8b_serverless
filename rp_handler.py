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


def handler(event):
    input = event['input']
    llama_vision_data = input.get('llama_vision_data')
    yolo_data = input.get('yolo_data')

    # Placeholder for a task; replace with image or text generation logic as needed
    messages = [
        {"role": "system", "content": "You are a accurate scene analyser who likes to conbine multiple source of information and give final verdict!"},
        {"role": "user", "content": f"{prompt}"
        f"\n\nHere is the data for your analysis:\n\n"
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

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})