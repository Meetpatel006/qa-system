import gradio as gr
from transformers import pipeline

# Initialize the pipeline
qa_pipeline = None

def load_model():
    """Load the question answering model from Hugging Face."""
    global qa_pipeline
    try:
        model_name = "RedRepter/my_awesome_qa_model"
        qa_pipeline = pipeline("question-answering", model=model_name)
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def answer_question(question: str, context: str):
    """
    Get answer for a question based on the provided context using HuggingFace pipeline.
    """
    if qa_pipeline is None:
        return "Error: Model not loaded", 0.0
    
    try:
        result = qa_pipeline(
            question=question,
            context=context,
            max_length=512,
            truncation=True
        )
        return result['answer'], float(result['score'])
    except Exception as e:
        return f"Error: {str(e)}", 0.0

def create_gradio_interface():
    # Load the model
    load_model()
    
    # Create the Gradio interface
    iface = gr.Interface(
        fn=answer_question,
        inputs=[
            gr.Textbox(label="Question", placeholder="Enter your question here..."),
            gr.Textbox(label="Context", placeholder="Enter the context here...", lines=5)
        ],
        outputs=[
            gr.Textbox(label="Answer"),
            gr.Number(label="Confidence Score")
        ],
        title="Question Answering System",
        description="This system answers questions based on the provided context using a transformer model.",
        examples=[
            [
                "How many programming languages does BLOOM support?",
                "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
            ],
            [
                "What is the capital of France?",
                "Paris is the capital and largest city of France. It is situated on the river Seine, in northern France."
            ]
        ],
        article="""
        ## How to use
        1. Enter your question in the first text box
        2. Provide the context (text passage) in the second text box
        3. Click submit to get the answer and confidence score
        
        The model will extract the answer from the context based on your question.
        The confidence score indicates how certain the model is about its answer.
        """
    )
    return iface

if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
