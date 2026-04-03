from openai import OpenAI
import base64

client = OpenAI(
    api_key="fill in here",
    base_url="https://api.featherless.ai/v1"
)

def analyzeImage(imagePath):
    with open(imagePath, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()


    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "You are a meeting summarizing agent tasked with understanding a photo from the meeting and summarizing it to help create a video enabled meeting transcript. Output everythigng in raw text"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }]
    )
    return response.model_dump()['choices'][0]['message']['content']

def generateMeetingSummary(slide_summaries, full_transcript):
    response = client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        messages=[
            {
                "role": "system",
                "content": "You are an expert meeting analyst. Write detailed, well-structured meeting summaries that are comprehensive but only include relevant information from video frame summaries and transcript. Output in plain text."
            },
            {
                "role": "user",
                "content": f"""
    Here are the per-slide summaries and full transcript from a meeting.

    SLIDE SUMMARIES:
    {slide_summaries}

    FULL TRANSCRIPT:
    {full_transcript}

    Write a detailed meeting summary that:
    1. Gives a brief overview of the meeting's purpose
    2. Covers each slide/topic in order with key points
    3. Highlights any data, metrics, or decisions mentioned
    4. Lists all action items and who owns them
    5. Notes any open questions or follow-ups
    """
            }
        ],
        max_tokens=4000,
        temperature=0.3,   # lower = more factual, less creative
    )

    return response.choices[0].message.content