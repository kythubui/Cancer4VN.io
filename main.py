from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "Qwen/Qwen2-0.5B"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LoRA adapters
LORA_PATHS = {
    "cancer": "/home/bobby/Cancer4VN/qwen2-0.5B-lora-finetuned",
    "cancer2": "/home/bobby/Cancer4VN/qwen2-0.5B-lora-finetuned-cancergov",
    "cancer3": "/home/bobby/Cancer4VN/qwen2-0.5B-lora-finetuned-breastcancer",
}

# Full SFT models
SFT_MODELS = {
    "sft": "./qwen-sft-final"
}

# New base Hugging Face models
BASE_MODELS = {
    "qwen3": "Qwen/Qwen3-0.6B"
}

# Load Hugging Face models (not LoRA or SFT, just raw pretrained)
base_models = {}
for name, model_id in BASE_MODELS.items():
    base_models[name] = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device).eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model for LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16
).to(device)

# Load all LoRA adapters
adapters = {}
for name, path in LORA_PATHS.items():
    adapters[name] = PeftModel.from_pretrained(base_model, path).to(device).eval()

# Load SFT models
sft_models = {}
for name, path in SFT_MODELS.items():
    sft_models[name] = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16
    ).to(device).eval()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "output": None})

@app.post("/generate", response_class=HTMLResponse)
async def generate_text(
    request: Request,
    text: str = Form(...),
    max_length: int = Form(200),
    model_choice: str = Form("cancer")
):
    # Select model
    if model_choice in adapters:
        model = adapters[model_choice]
    elif model_choice in sft_models:
        model = sft_models[model_choice]
    elif model_choice in base_models:
        model = base_models[model_choice]
    else:
        return templates.TemplateResponse(
            "home.html",
            {"request": request, "output": f"Model '{model_choice}' not found.", "text": text}
        )

    model.to(device)
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1024
    ).to(device)

    # Always generate in FP32 to avoid NaNs
    with torch.no_grad():
        outputs = model.to(torch.float32).generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=0.5,
            pad_token_id=tokenizer.eos_token_id  # required to prevent invalid probabilities
        )
    # Revert model to FP16 if using LoRA
    if model_choice in adapters:
        model.to(torch.float16)

    # Decode output and remove prompt
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = full_text[len(text):].strip() if full_text.startswith(text) else full_text.strip()

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "output": result,
            "text": text,
            "max_length": max_length,
            "model_choice": model_choice
        }
    )
