from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import random
import torch
from collections import defaultdict


samsum = load_dataset("samsum")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")


class collator(DataCollatorForSeq2Seq):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, features, return_tensors=None):

        contrastive_label = []
        rang = []
        id = []
        dialogue = []
        summary = []

        if "contrastive_labels" in features[0]:
            for i in range(len(features)):
                contrastive_label.append(features[i].pop("contrastive_labels"))
                rang.append(features[i].pop("rang"))
                id.append(features[i].pop("id"))
                dialogue.append(features[i].pop("dialogue"))
                summary.append(features[i].pop("summary"))

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        
        if len(contrastive_label) > 0:
            features["contrastive_labels"] = contrastive_label
            features["rang"] = rang

        return features


def preprocess_function(examples):
    inputs = [doc for doc in examples["dialogue"]]
    total_rang = []
    total_contrastive_label = []
    splitted = [x.split("\r\n") for x in inputs]


    # dialogue_preprocess : [[[idx, speaker, utterance]....[idx, speaker, utterance]] ] => data * dialogue * utterance
    dialogue_preprocess = []
    for dia in splitted:
        k = []
        for idx, spe in enumerate(dia):
            tmp_utter = spe.split(":")
            if tmp_utter[0] == '':
                # utterance 안 ":" str이 존재하는 경우 있음
                # 근데 해당 데이터 셋에서는 utterance 마지막 부분에 위치 
                break
            k.append([idx, tmp_utter[0], tmp_utter[1]])
        dialogue_preprocess.append(k)


    for ww in dialogue_preprocess:
        idx = []
        tok = []
        cnt = 1
        rang = []
        spe = []
        
        # cnt = cursor, rang = dialog_length * [cnt, speaker_token, utterance_token]
        # rang 리스트의 index 정보를 통해 sampling한 utterance vector mean pooling에 사용    
        for j, i in enumerate(ww):
            spe.append(len(tokenizer.tokenize(i[1] + ":")))
            if j==len(ww)-1:
                idx.append(len(tokenizer.tokenize(i[1] + ":" + i[2])))
                tok.extend(tokenizer.tokenize(i[1] + ":" + i[2]))
            else:
                idx.append(len(tokenizer.tokenize(i[1] + ":" + i[2] + "\r\n")))
                tok.extend(tokenizer.tokenize(i[1] + ":" + i[2] + "\r\n"))
            rang.append([cnt, cnt + spe[j], cnt+idx[j]])
            cnt += idx[j]


        cnt = 2
        if len(rang) >= cnt:
            assert len(rang) >= cnt, print(len(rang), cnt, len(dialogue_preprocess))
            sample_rang = random.sample(rang, cnt)

            s_i = ww[rang.index(sample_rang[0])][1]
            s_j = ww[rang.index(sample_rang[1])][1]
            label = 1 if s_i==s_j else 0
            total_rang.append(sample_rang)
            total_contrastive_label.append(label)
        else:
            total_rang.append([])
            total_contrastive_label.append(0)

    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=80, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["rang"] = total_rang
    model_inputs["contrastive_labels"] = total_contrastive_label

    return model_inputs


# dataset과 preprocess function mapping, definition data collator, metric definition
tokenized_samsum = samsum.map(preprocess_function, batched=True)
data_collator = collator(tokenizer=tokenizer, model=model)
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


def mean_pooling(model_output, attention_mask, rang, i=True, j=True):
    # model_output : bsz * input * hidden, model_output[:, 0:1, :] : bsz * 1 * hidden
    # attention_mask : bsz * input
    # input_mask_expanded : bsz * input * hidden
    # rang : bsz * 1 * [[sampled_1 start, sampled_1 end], [sampled_2 start, sampled_2 end]]

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    result = torch.tensor([]).to(model_output.device)
    if j == None:
        for w in range(model_output.size(0)):
            if len(rang[w]) == 0:
                # 하나의 utterance로만 구성된 dialogue는 샘플링 불가
                # 빈 list 형태의 데이터 
                # zero vector 처리를 통해 loss 계산 x
                k = torch.zeros(1024).to(model_output.device)
                result = torch.cat((result, k), dim=0)
                continue    
            start = rang[w][0][1]
            end = rang[w][0][1]
            k = torch.sum(model_output[w][start:end] * input_mask_expanded[w][start:end], 0) / torch.clamp(input_mask_expanded[w][start:end].sum(0), min=1e-9)
            result = torch.cat((result, k), dim=0)
        result = result.view(-1, 1024)

    elif i == None:
        for w in range(model_output.size(0)):
            if len(rang[w]) == 0:
                # 하나의 utterance로만 구성된 dialogue는 샘플링 불가
                # 빈 list 형태의 데이터 
                # zero vector 처리를 통해 loss 계산 x
                k = torch.zeros(1024).to(model_output.device)
                result = torch.cat((result, k), dim=0)
                continue 
            start = rang[w][1][1]
            end = rang[w][1][1]
            k = torch.sum(model_output[w][start:end] * input_mask_expanded[w][start:end], 0) / torch.clamp(input_mask_expanded[w][start:end].sum(0), min=1e-9)
            result = torch.cat((result, k), dim=0)
        result = result.view(-1, 1024)

    return result


training_args = Seq2SeqTrainingArguments(
    output_dir="speaker_samsum_save",
    per_device_train_batch_size= 8,
    per_device_eval_batch_size= 8,
    save_total_limit=3,
    evaluation_strategy="steps",
    gradient_accumulation_steps= 1,
    learning_rate= 2e-5,
    max_steps=10000,
    eval_steps=1000,
    save_steps=1000,
    weight_decay= 0.1,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    fp16=True,
    seed=1
)


class BartTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset

        def get_label(dataset):
            return dataset["labels"]

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if "contrastive_labels" in inputs:
            contrastive_label = inputs.pop("contrastive_labels")
            rang = inputs.pop("rang")

        outputs = model(**inputs)

        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            ### Naive Generative Loss
            gen_loss = self.label_smoother(outputs, labels)     
            
            ### Turn level Contrastive Loss
            if len(contrastive_label) > 0:   
                # encoder_last_hidden_state : bsz * input * hidden_dim
                encoder_last_hidden_state = outputs["encoder_last_hidden_state"]
                
                # sampled utterance vector
                # o_i, o_j : bsz * hidden_dim
                o_i = mean_pooling(encoder_last_hidden_state, inputs["attention_mask"], rang, j=None)
                o_j = mean_pooling(encoder_last_hidden_state, inputs["attention_mask"], rang, i=None)
                
                # h 안에서 sampling 한 o_i, o_j vector 부분 mean pooling 이 후 dot() 연산 transpose()
                contrastive_loss = 0.0
                for idx, label in enumerate(contrastive_label):
                    if label == 1:
                        pos_contrastive_loss = -torch.log(torch.sigmoid(torch.dot(o_i[idx], o_j[idx])))
                        contrastive_loss += pos_contrastive_loss
                    else:
                        neg_contrastive_loss = -torch.log(1-torch.sigmoid(torch.dot(o_i[idx], o_j[idx])))
                        contrastive_loss += neg_contrastive_loss
                
                # 논문에서 설정한 Hyperparameter lambda
                lam = 0.01
                contrastive_loss = contrastive_loss * lam
                
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            gen_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            assert 0==1, print("accidient!!!")


        ### Incorporating Contrastive Loss
        if len(contrastive_label) > 0: 
            total_loss = gen_loss + contrastive_loss
        else:
            total_loss = gen_loss

        return (total_loss, outputs) if return_outputs else total_loss


trainer = BartTrainer(
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
