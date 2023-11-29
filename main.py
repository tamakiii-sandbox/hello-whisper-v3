import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path>")
        sys.exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(sys.argv[1])

    # Process and print the chunks with timestamps
    if "chunks" in result:
        for chunk in result["chunks"]:
            start_time = format_timestamp(chunk["timestamp"][0])
            end_time = format_timestamp(chunk["timestamp"][1])
            print(f"{start_time} --> {end_time}: {chunk['text']}")
    else:
        print("No chunks found or 'chunks' key missing.")
        sys.exit(2)

if __name__ == "__main__":
    main()
