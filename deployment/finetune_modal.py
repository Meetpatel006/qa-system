import modal
import os
from typing import Dict, Any

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
])

# Hardcoded configuration
HUGGINGFACE_TOKEN = "your_huggingface_token_here"  # Replace with your actual token
MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = "my_awesome_qa_model"
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

@app.function(
    image=image,
    gpu=modal.gpu.A10G(),  # Use A10G GPU (you can change to A100 or H100 for faster training)
    timeout=3600,  # 1 hour timeout
    volumes={"/models": volume},
)
def preprocess_function(examples: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Preprocesses the SQuAD examples by tokenizing questions and contexts,
    and mapping answer start/end positions to token indices.
    """
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
            while idx <= context_start and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

@app.function(
    image=image,
    gpu=modal.gpu.A10G(),
    timeout=7200,  # 2 hours timeout for training
    volumes={"/models": volume},
)
def train_qa_model():
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

    # Use hardcoded configuration
    model_name = MODEL_NAME
    output_dir = OUTPUT_DIR
    num_samples = NUM_SAMPLES
    num_epochs = NUM_EPOCHS
    batch_size = BATCH_SIZE
    learning_rate = LEARNING_RATE
    push_to_hub = PUSH_TO_HUB
    
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

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=f"/models/{output_dir}",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir=f"/models/{output_dir}/logs",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=push_to_hub,
        hub_model_id=output_dir if push_to_hub else None,
        report_to=None,  # Disable wandb/tensorboard for simplicity
    )

    # Initialize trainer
    data_collator = DefaultDataCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    print("Starting model training...")
    trainer.train()
    
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
    return f"/models/{output_dir}"

@app.function(
    image=image,
    gpu=modal.gpu.A10G(),
    timeout=600,  # 10 minutes timeout for inference
    volumes={"/models": volume},
)
def perform_inference(
    model_path: str = None,
    question: str = None,
    context: str = None,
    framework: str = "pipeline"
):
    """
    Perform inference using the fine-tuned model.
    """
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    
    # Use hardcoded values if not provided
    if model_path is None:
        model_path = f"/models/{OUTPUT_DIR}"
    if question is None:
        question = TEST_QUESTION
    if context is None:
        context = TEST_CONTEXT
    
    print(f"Loading model from: {model_path}")
    
    if framework == "pipeline":
        qa_pipeline = pipeline("question-answering", model=model_path)
        result = qa_pipeline(question=question, context=context)
        print("\n--- Inference Results ---")
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence Score: {result['score']:.4f}")
        return result
    
    elif framework == "manual":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        
        inputs = tokenizer(question, context, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        
        print("\n--- Manual Inference Results ---")
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Answer: {answer}")
        return {"answer": answer}
    
    else:
        raise ValueError("Framework must be 'pipeline' or 'manual'")

@app.local_entrypoint()
def main():
    """
    Main function to orchestrate training or inference.
    
    Set MODE to "train" or "inference" to control what happens.
    """
    MODE = "train"  # Change this to "inference" to run inference instead
    
    if MODE == "train":
        print("Starting training on Modal...")
        model_path = train_qa_model.remote()
        print(f"Training completed! Model saved at: {model_path}")
        
    elif MODE == "inference":
        print("Running inference on Modal...")
        result = perform_inference.remote()
        print("Inference completed!")
        return result
        
    else:
        raise ValueError("MODE must be either 'train' or 'inference'")

if __name__ == "__main__":
    main()