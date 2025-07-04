import modal

# Create the Modal image with all required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy<2.0.0",  # Fix NumPy compatibility issue
    "torch==2.1.2",  # Update to PyTorch 2.1+ for transformers compatibility
    "torchvision==0.16.2",  # Add torchvision for complete PyTorch ecosystem
    "transformers>=4.36.0",  # Use newer version for better compatibility
    "gradio==4.44.1",
    "accelerate>=0.25.0",  # Use newer compatible version
    "fastapi[standard]==0.115.4",
    "pydantic==2.9.2",
    "starlette==0.41.2",
)

# Create the Modal app
app = modal.App("qa-system", image=image)

# Define the QA model class
@app.cls(
    gpu="T4",  # Use T4 GPU for cost-effective inference
    scaledown_window=300,  # Keep container alive for 5 minutes after last request
    min_containers=1,  # Keep at least one container warm
)
class QuestionAnsweringModel:
    # Remove custom __init__ to fix deprecation warning
    qa_pipeline: object = None
    
    @modal.enter()
    def load_model(self):
        """Load the question answering model on container startup."""
        from transformers import pipeline
        
        try:
            model_name = "RedRepter/my_awesome_qa_model"
            self.qa_pipeline = pipeline("question-answering", model=model_name)
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a default model if the custom model fails
            print("Falling back to default DistilBERT model...")
            self.qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    @modal.method()
    def answer_question(self, question: str, context: str):
        """
        Get answer for a question based on the provided context.
        """
        if self.qa_pipeline is None:
            return "Error: Model not loaded", 0.0
        
        try:
            result = self.qa_pipeline(
                question=question,
                context=context,
                max_length=512,
                truncation=True
            )
            return result['answer'], float(result['score'])
        except Exception as e:
            return f"Error: {str(e)}", 0.0

# Create the Gradio web interface
@app.function(
    image=image,
    # Gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 1000 concurrent inputs
    max_containers=1,
)
@modal.concurrent(max_inputs=1000)
@modal.asgi_app()
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    # Create FastAPI app
    web_app = FastAPI()

    # Initialize the model
    model = QuestionAnsweringModel()

    def process_question(question: str, context: str):
        """Process the question and return answer with confidence score."""
        if not question.strip() or not context.strip():
            return "Please provide both a question and context.", 0.0
        
        answer, confidence = model.answer_question.remote(question, context)
        return answer, confidence

    # Create the Gradio interface
    with gr.Blocks(title="Question Answering System") as demo:
        gr.Markdown(
            """
            # Question Answering System
            
            This system answers questions based on the provided context using a transformer model.
            Enter your question and provide relevant context to get accurate answers.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Enter your question here...",
                    lines=2
                )
                context_input = gr.Textbox(
                    label="Context",
                    placeholder="Enter the context here...",
                    lines=8
                )
                submit_btn = gr.Button("Get Answer", variant="primary")
            
            with gr.Column(scale=1):
                answer_output = gr.Textbox(
                    label="Answer",
                    lines=3,
                    interactive=False
                )
                confidence_output = gr.Number(
                    label="Confidence Score",
                    interactive=False
                )
        
        # Add example inputs
        gr.Examples(
            examples=[
                [
                    "How many programming languages does BLOOM support?",
                    "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
                ],
                [
                    "What is the capital of France?",
                    "Paris is the capital and largest city of France. It is situated on the river Seine, in northern France."
                ],
                [
                    "When was Modal founded?",
                    "Modal was founded in 2021 by Erik Bernhardsson and Akshat Bubna. The company provides serverless cloud computing infrastructure specifically designed for AI and machine learning workloads."
                ]
            ],
            inputs=[question_input, context_input],
            label="Try these examples:"
        )
        
        # Connect the button to the processing function
        submit_btn.click(
            fn=process_question,
            inputs=[question_input, context_input],
            outputs=[answer_output, confidence_output]
        )
        
        # Also trigger on Enter key press
        question_input.submit(
            fn=process_question,
            inputs=[question_input, context_input],
            outputs=[answer_output, confidence_output]
        )
        context_input.submit(
            fn=process_question,
            inputs=[question_input, context_input],
            outputs=[answer_output, confidence_output]
        )

        gr.Markdown(
            """
            ## How to use
            1. Enter your question in the first text box
            2. Provide the context (text passage) in the second text box
            3. Click "Get Answer" or press Enter to get the answer and confidence score
            
            The model will extract the answer from the context based on your question.
            The confidence score indicates how certain the model is about its answer.
            
            ---
            **Powered by [Modal](https://modal.com) 🚀**
            """
        )

    # Mount the Gradio app on FastAPI
    return mount_gradio_app(app=web_app, blocks=demo, path="/")

# Optional: Add a local entrypoint for testing
@app.local_entrypoint()
def test():
    """Test the model locally before deployment."""
    model = QuestionAnsweringModel()
    
    # Test question and context
    question = "What is Modal?"
    context = "Modal is a serverless cloud platform designed specifically for AI and machine learning workloads. It allows developers to run code in the cloud without managing infrastructure."
    
    answer, confidence = model.answer_question.remote(question, context)
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer}")
    print(f"Confidence: {confidence:.3f}")

# Add a simple health check endpoint
@app.function()
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "qa-system"}

if __name__ == "__main__":
    # This allows running the script locally for testing
    print("Use 'modal serve modal_deploy.py' to run the Gradio interface")
    print("Use 'modal deploy modal_deploy.py' to deploy permanently")
