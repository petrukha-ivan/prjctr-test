from fastapi import FastAPI
from pydantic import BaseModel
from transformers import HfArgumentParser as ArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass, field

@dataclass
class ServerArguments:

    model_checkpoint: str = field(
        default='distilbert/distilbert-base-cased',
        metadata={'help': 'Trained model path.'},
    )


class TextRequest(BaseModel):
    text: str


app = FastAPI()

@app.post('/score')
async def get_score(request: TextRequest):
    inputs = tokenizer(request.text, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    score = outputs.logits[0][0].item()
    return {'score': score}


if __name__ == '__main__':
    # Parse arguments
    parser = ArgumentParser(ServerArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Load model from checkpoint
    tokenizer = AutoTokenizer.from_pretrained(f'runs/{args.model_checkpoint}/checkpoint-best')
    model = AutoModelForSequenceClassification.from_pretrained(f'runs/{args.model_checkpoint}/checkpoint-best').eval()

    # Start server
    import uvicorn
    uvicorn.run(app)