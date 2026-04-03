from groq import Groq
groqAPI = ""
client = Groq(api_key=groqAPI)

def makeTranscript (fileName):
    with open(fileName, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            response_format="verbose_json",  # gives you timestamps
            timestamp_granularities=["word"]
        )
        return result

if __name__ == "__main__":
    with open("presentation.mov", "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            response_format="verbose_json",  # gives you timestamps
            timestamp_granularities=["word"]
        )
        print(result.words)