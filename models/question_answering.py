import os
import torch
import tensorflow as tf
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    create_optimizer,
    TFAutoModelForQuestionAnswering,
    PushToHubCallback,
)
from huggingface_hub import notebook_login

def preprocess_function(examples, tokenizer):
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

def train_pytorch_model(output_dir, model_name, train_dataset, eval_dataset, tokenizer, data_collator):
    """
    Trains a PyTorch question answering model using the Hugging Face Trainer.
    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,  # Use tokenizer here for logging/saving purposes
        data_collator=data_collator,
    )

    print("Starting PyTorch model training...")
    trainer.train()
    print("PyTorch model training complete. Pushing to Hugging Face Hub...")
    trainer.push_to_hub()
    print(f"Model pushed to Hub at: {output_dir}")

def train_tensorflow_model(output_dir, model_name, train_dataset, eval_dataset, tokenizer, data_collator):
    """
    Trains a TensorFlow question answering model.
    """
    batch_size = 16
    num_epochs = 3
    total_train_steps = (len(train_dataset) // batch_size) * num_epochs
    optimizer, schedule = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=0,
        num_train_steps=total_train_steps,
    )

    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)

    tf_train_set = model.prepare_tf_dataset(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        eval_dataset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=data_collator,
    )

    model.compile(optimizer=optimizer)

    callback = PushToHubCallback(
        output_dir=output_dir,
        tokenizer=tokenizer,
    )

    print("Starting TensorFlow model training...")
    model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=num_epochs, callbacks=[callback])
    print(f"TensorFlow model training complete. Pushed to Hugging Face Hub at: {output_dir}")


def perform_inference(model_path, question, context, framework):
    """
    Performs inference using a finetuned question answering model.
    """
    if framework == "pipeline":
        from transformers import pipeline
        Youtubeer = pipeline("question-answering", model=model_path)
        result = Youtubeer(question=question, context=context)
        print("\n--- Inference using Hugging Face Pipeline ---")
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Answer: {result['answer']} (Score: {result['score']:.4f})")
    elif framework == "pytorch":
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        inputs = tokenizer(question, context, return_tensors="pt")

        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        with torch.no_grad():
            outputs = model(**inputs)

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens)
        print("\n--- Inference using PyTorch (manual) ---")
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Answer: {answer}")
    elif framework == "tensorflow":
        from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
        import tensorflow as tf

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        inputs = tokenizer(question, context, return_tensors="tf")

        model = TFAutoModelForQuestionAnswering.from_pretrained(model_path)
        outputs = model(**inputs)

        answer_start_index = int(tf.math.argmax(outputs.start_logits, axis=-1)[0])
        answer_end_index = int(tf.math.argmax(outputs.end_logits, axis=-1)[0])

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        answer = tokenizer.decode(predict_answer_tokens)
        print("\n--- Inference using TensorFlow (manual) ---")
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Answer: {answer}")
    else:
        print("Invalid framework for inference. Choose 'pipeline', 'pytorch', or 'tensorflow'.")


def main():
    # Define parameters directly
    mode = "train_pytorch"  # or "train_tensorflow" or "inference"
    model_name = "distilbert/distilbert-base-uncased"
    output_dir = "my_awesome_qa_model"
    question = "How many programming languages does BLOOM support?"
    context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
    inference_framework = "pipeline" # or "pytorch" or "tensorflow"

    # Log in to Hugging Face Hub if needed for pushing models
    if mode.startswith("train"):
        try:
            notebook_login()
        except Exception as e:
            print(f"Could not log in to Hugging Face Hub. Make sure you have a token configured. Error: {e}")
            print("Training will proceed, but model might not be pushed to the Hub without successful login.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DefaultDataCollator()

    if mode.startswith("train"):
        # Load and preprocess dataset
        print("Loading SQuAD dataset...")
        squad_train = load_dataset("squad", split="train", download_mode="force_redownload")
        squad = squad_train.select(range(5000))
        squad = squad.train_test_split(test_size=0.2)
        print(f"Number of training samples: {len(squad['train'])}")
        print(f"Number of test samples: {len(squad['test'])}")

        print("Tokenizing and preprocessing dataset...")
        # Pass tokenizer to preprocess_function
        tokenized_squad = squad.map(lambda examples: preprocess_function(examples, tokenizer),
                                    batched=True,
                                    remove_columns=squad["train"].column_names)
        print("Dataset preprocessing complete.")

        if mode == "train_pytorch":
            train_pytorch_model(output_dir, model_name,
                                tokenized_squad["train"], tokenized_squad["test"],
                                tokenizer, data_collator)
        elif mode == "train_tensorflow":
            train_tensorflow_model(output_dir, model_name,
                                   tokenized_squad["train"], tokenized_squad["test"],
                                   tokenizer, data_collator)
    elif mode == "inference":
        # Ensure the model directory exists for inference
        if not os.path.exists(output_dir):
            print(f"Error: Model directory '{output_dir}' not found. Please train the model first or specify a valid model path.")
            return

        perform_inference(output_dir, question, context, inference_framework)

if __name__ == "__main__":
    main()