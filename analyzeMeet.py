from videoSegmentation import extract_slide_frames
from whisperTest import makeTranscript
from LLM import analyzeImage, generateMeetingSummary
import json

# video processing params
videoPath = "presentation.mp4"
outputDir = "slides"
threshold = 0.92
sampleFps = 1
audioPath = "presentation.wav"

#process video

extract_slide_frames(videoPath,outputDir,threshold,sampleFps)


# transcription
transcript = makeTranscript(audioPath)
print(transcript.text)


f = open('slides/timeline.json')
data = json.load(f)


explanations =[]
i = 1
for slide in data:
    explanations.append("Frame "+str(i)+" "+ analyzeImage(slide['filepath'])+" ")
    break

print(explanations)

output = generateMeetingSummary(str(explanations),transcript.text)
print(output)