import modal
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Define the Modal app
app = modal.App("qa-model-finetuning")

# Define the image with all required dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install([
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "datasets>=2.12.0",
    "huggingface-hub>=0.15.0",
    "accelerate>=0.20.0",
    "evaluate>=0.4.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "tensorboard>=2.12.0",
])

# Hardcoded configuration
HUGGINGFACE_TOKEN = "your_huggingface_token_here"  # Replace with your actual token
MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = "qa_model"
NUM_SAMPLES = 5000
NUM_EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
PUSH_TO_HUB = True

# Test data for inference
TEST_QUESTION = "How many programming languages does BLOOM support?"
TEST_CONTEXT = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

# Create a shared volume for model storage
volume = modal.Volume.from_name("qa-model-storage", create_if_missing=True)

# Pydantic models for API requests
class TrainingRequest(BaseModel):
    model_name: Optional[str] = MODEL_NAME
    num_samples: Optional[int] = NUM_SAMPLES
    num_epochs: Optional[int] = NUM_EPOCHS
    batch_size: Optional[int] = BATCH_SIZE
    learning_rate: Optional[float] = LEARNING_RATE
    push_to_hub: Optional[bool] = PUSH_TO_HUB

class InferenceRequest(BaseModel):
    question: str
    context: str
    model_path: Optional[str] = None

class TrainingResponse(BaseModel):
    status: str
    message: str
    model_path: Optional[str] = None

class InferenceResponse(BaseModel):
    answer: str
    confidence_score: float
    question: str
    context: str

@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,  # 2 hours timeout for training
    volumes={"/models": volume},
)
def train_qa_model(
    model_name: str = MODEL_NAME,
    num_samples: int = NUM_SAMPLES,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    push_to_hub: bool = PUSH_TO_HUB
):
    """
    Fine-tune a question answering model on Modal's GPU infrastructure.
    """
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForQuestionAnswering,
        TrainingArguments,
        Trainer,
        DefaultDataCollator,
    )
    from huggingface_hub import login
    
    # Login to Hugging Face Hub using hardcoded token
    if HUGGINGFACE_TOKEN and HUGGINGFACE_TOKEN != "your_huggingface_token_here":
        login(token=HUGGINGFACE_TOKEN)
        print("Successfully logged in to Hugging Face Hub")
    else:
        print("Warning: No valid Hugging Face token found. Model won't be pushed to Hub.")
        push_to_hub = False

    output_dir = OUTPUT_DIR
    
    print(f"Using device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load tokenizer and model
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    # Load and preprocess dataset
    print("Loading SQuAD dataset...")
    squad_train = load_dataset("squad", split="train")
    squad = squad_train.select(range(num_samples))
    squad = squad.train_test_split(test_size=0.2)
    print(f"Number of training samples: {len(squad['train'])}")
    print(f"Number of validation samples: {len(squad['test'])}")

    # Tokenize dataset
    print("Tokenizing and preprocessing dataset...")
    def preprocess_with_tokenizer(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_squad = squad.map(
        preprocess_with_tokenizer,
        batched=True,
        remove_columns=squad["train"].column_names
    )
    print("Dataset preprocessing complete.")

    # Setup training arguments with correct parameter names
    training_args = TrainingArguments(
        output_dir=f"/models/{output_dir}",
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=500,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"/models/{output_dir}/logs",
        logging_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=push_to_hub,
        hub_model_id=output_dir if push_to_hub else None,
        report_to="tensorboard",
        logging_first_step=True,
    )

    # Add evaluation metrics
    import evaluate
    metric = evaluate.load("squad")
    
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        start_logits, end_logits = predictions
        start_predictions = start_logits.argmax(axis=-1)
        end_predictions = end_logits.argmax(axis=-1)
        
        print("\nEvaluation metrics:")
        print(f"Average start position accuracy: {(start_predictions == labels[0]).mean():.4f}")
        print(f"Average end position accuracy: {(end_predictions == labels[1]).mean():.4f}")
        
        return {
            "start_accuracy": (start_predictions == labels[0]).mean(),
            "end_accuracy": (end_predictions == labels[1]).mean(),
        }

    # Initialize trainer with metrics
    data_collator = DefaultDataCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["test"],
        processing_class=tokenizer,  # Changed from tokenizer to processing_class
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training with progress monitoring
    print("\nStarting model fine-tuning...")
    print(f"Training on {len(tokenized_squad['train'])} examples")
    print(f"Evaluating on {len(tokenized_squad['test'])} examples")
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    train_result = trainer.train()
    
    # Print training results
    print("\nTraining completed!")
    print(f"Total training steps: {train_result.global_step}")
    print(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Evaluate the model
    print("\nEvaluating final model...")
    eval_result = trainer.evaluate()
    print(f"Final evaluation loss: {eval_result['eval_loss']:.4f}")
    print(f"Start position accuracy: {eval_result['eval_start_accuracy']:.4f}")
    print(f"End position accuracy: {eval_result['eval_end_accuracy']:.4f}")
    
    # Save the model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(f"/models/{output_dir}")
    
    # Push to hub if requested
    if push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()
        print(f"Model pushed to Hub: {output_dir}")
    
    # Commit volume changes
    volume.commit()
    
    print("Training completed successfully!")
    return {
        "status": "success",
        "model_path": f"/models/{output_dir}",
        "training_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "start_accuracy": eval_result["eval_start_accuracy"],
        "end_accuracy": eval_result["eval_end_accuracy"]
    }

@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    volumes={"/models": volume},
)
def perform_inference(
    model_path: str = None,
    question: str = None,
    context: str = None
):
    """
    Perform inference using the fine-tuned model.
    """
    from transformers import pipeline
    
    # Use hardcoded values if not provided
    if model_path is None:
        model_path = f"/models/{OUTPUT_DIR}"
    if question is None:
        question = TEST_QUESTION
    if context is None:
        context = TEST_CONTEXT
    
    print(f"Loading model from: {model_path}")
    
    try:
        qa_pipeline = pipeline("question-answering", model=model_path)
        result = qa_pipeline(question=question, context=context)
        
        print("\n--- Inference Results ---")
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence Score: {result['score']:.4f}")
        
        return {
            "answer": result["answer"],
            "confidence_score": result["score"],
            "question": question,
            "context": context
        }
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise e

# Create FastAPI app
web_app = FastAPI(title="QA Model Training API", version="1.0.0")

@web_app.get("/")
async def root():
    return {
        "message": "QA Model API is running",
        "endpoints": {
            "POST /train": "Start model training",
            "POST /inference": "Perform inference with the model",
            "GET /health": "Health check"
        }
    }

@web_app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "qa-model-api"}

@web_app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Start model training with the provided parameters.
    """
    try:
        print(f"Starting training with parameters: {request.dict()}")
        
        # Call the training function
        result = train_qa_model.remote(
            model_name=request.model_name,
            num_samples=request.num_samples,
            num_epochs=request.num_epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            push_to_hub=request.push_to_hub
        )
        
        return TrainingResponse(
            status="success",
            message="Training completed successfully",
            model_path=result["model_path"]
        )
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@web_app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    Perform inference using the trained model.
    """
    try:
        result = perform_inference.remote(
            model_path=request.model_path,
            question=request.question,
            context=request.context
        )
        
        return InferenceResponse(
            answer=result["answer"],
            confidence_score=result["confidence_score"],
            question=result["question"],
            context=result["context"]
        )
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Deploy the FastAPI app
@app.function(
    image=image,
    timeout=300
)
@modal.asgi_app()
def fastapi_app():
    return web_app

# CLI functions for local testing
@app.local_entrypoint()
def train_locally():
    """Train the model locally via CLI"""
    result = train_qa_model.remote()
    print(f"Training result: {result}")

@app.local_entrypoint()
def inference_locally():
    """Run inference locally via CLI"""
    result = perform_inference.remote()
    print(f"Inference result: {result}")

if __name__ == "__main__":
    # For local development, you can run:
    # modal serve modal_qa_app.py
    pass