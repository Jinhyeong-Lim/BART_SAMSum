from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

samsum = load_dataset("samsum")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")


def preprocess_function(examples):
    inputs = [doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=80, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_samsum = samsum.map(preprocess_function, batched=True)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


training_args = Seq2SeqTrainingArguments(
    output_dir="samsum_save",
    per_device_train_batch_size= 8,
    per_device_eval_batch_size= 8,
    save_total_limit=3,
    evaluation_strategy="epoch",
    gradient_accumulation_steps= 1,
    learning_rate= 5e-5,
    num_train_epochs= 3.0,
    weight_decay= 0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    fp16=True,
    seed=1
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_samsum["train"],
    eval_dataset=tokenized_samsum["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
predict_results = trainer.predict(
            tokenized_samsum["test"],
            metric_key_prefix=" ",
            max_length=80,
            num_beams=6,
            length_penalty=1.0,
            no_repeat_ngram_size=3
        )
metrics = predict_results.metrics

trainer.log_metrics("predict", metrics)
trainer.save_metrics("predict", metrics)
