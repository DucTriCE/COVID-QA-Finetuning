# Fine-tuning Question-Answering Model for COVID-QA dataset

Full-finetuning Question-Answering model using the COVID-QA dataset related to COVID-19.

## Dataset

The dataset used for this project is the [COVID-QA Question Answering Dataset](https://huggingface.co/datasets/minh21/COVID-QA-question-answering-biencoder-data-75_25).

## Project Structure
```markdown
src/
├── config/
│   └── cfg.yaml
├── data/
│   └── dataset.py
├── models/
│   └── model.py
├── scripts/
│   ├── test.py
│   └── train.py
├── utils/
│   └── utils.py
└── app.py
requirements.txt
```


## Model Information

RoBERTa (Robustly optimized BERT approach) - `roberta-base` 
## Fine-tuning the Model

To fine-tune the model, follow these steps:

1. **Install Dependencies**: Install the required dependencies using the following command:
    ```sh
    pip install -r requirements.txt
    ```

2. **Prepare the Dataset**: The dataset is automatically loaded and preprocessed by the scripts.

3. **Train the Model**: Run the training script to fine-tune the model:
    ```sh
    python src/scripts/train.py
    ```

4. **Evaluate the Model**: Run the testing script to evaluate the model:
    ```sh
    python src/scripts/test.py
    ```

5. **Run the App**: Run the Streamlit app for real-time question-answering with the model:
    ```sh
    streamlit run src/app.py
    ```

The configuration for the model, training, and evaluation is specified in the [cfg.yaml](http://_vscodecontentref_/1) file. Adjust the configuration as needed for your specific requirements.