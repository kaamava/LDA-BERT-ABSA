
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch
from torch.autograd import grad

class BertForABSA(BertModel):
    def __init__(self, config, num_labels=3, dropout=None, epsilon=None):
        super(BertForABSA, self).__init__(config)
        self.num_labels = num_labels
        self.epsilon = epsilon
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        pooled_output, bert_emb = self.bert_forward(input_ids, token_type_ids, 
                                                        attention_mask=attention_mask, 
                                                        output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            _loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if pooled_output.requires_grad: #if training mode
                perturbed_sentence = self.adv_attack(bert_emb, _loss)
                perturbed_sentence = self.replace_cls_token(bert_emb, perturbed_sentence)
                adv_loss = self.adversarial_loss(perturbed_sentence, attention_mask, labels)
                return _loss, adv_loss
            return _loss
        else:
            return logits

    def adv_attack(self, emb, loss):
        loss_grad = grad(loss, emb, retain_graph=True)[0]
        loss_grad_norm = torch.sqrt(torch.sum(loss_grad**2, (1,2)))
        perturbed_sentence = emb + self.epsilon * (loss_grad/(loss_grad_norm.reshape(-1,1,1)))
        return perturbed_sentence

    def replace_cls_token(self, emb, perturbed):
        cond  = torch.zeros_like(emb)
        cond[:, 0, :] = 1
        perturbed_sentence = torch.where(cond.byte(), emb, perturbed)
        return perturbed_sentence


    def bert_forward(self, input_ids, token_type_ids=None, 
                            attention_mask=None, output_all_encoded_layers=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, 
                                        extended_attention_mask, 
                                        output_all_encoded_layers=output_all_encoded_layers)
        pooled_output = self.pooler(encoded_layers[-1])
        return pooled_output, embedding_output


    def adversarial_loss(self, perturbed, attention_mask, labels):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoded_layers = self.encoder(perturbed, extended_attention_mask, 
                                        output_all_encoded_layers=False)
        encoded_layers_last = self.pooler(encoded_layers[-1])
        encoded_layers_last = self.dropout(encoded_layers_last)
        logits = self.classifier(encoded_layers_last)
        adv_loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return adv_loss