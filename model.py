from numpy import dtype
import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchmetrics
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

class GRUClassifier(pl.LightningModule):
    def __init__(self, segmentor, tokenizer, encoder, embedding_size=768, hidden_dim=256, output_dim=2,
                 n_layers=2, bidirectional=True, dropout=0.25, fc_dropout=0.5, max_sequence_length=256, device='cuda:0',
                 threshold=0.5):
        super().__init__()
        self.f1_score = []
        self.max_sequence_length = max_sequence_length
        self.segmentor = segmentor
        self.tokenizer = tokenizer
        self.encoder = encoder
        # self.encoder.eval()
        self.rnn = nn.GRU(embedding_size,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        # self.out = nn.Linear(embedding_size * 4, output_dim)
        self.dropout = nn.Dropout(fc_dropout)
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.train_f1 = torchmetrics.F1Score(2, threshold, average='macro')
        self.val_f1 = torchmetrics.F1Score(2, threshold, average='macro')

        self.f1  = torchmetrics.F1Score(num_classes=3)
        self.prediction = []
        self.threshold = threshold

        self.list_f1 = []
        self.list_f1_keras = []
        self.list_acc_keras = []
    def flat_accuracy(self, preds, labels):
        pred_flat = preds.flatten().cpu().numpy()
        labels_flat = labels.flatten().cpu().numpy()
    
        F1_score = f1_score(pred_flat, labels_flat, average='macro')
    
        return accuracy_score(pred_flat, labels_flat), F1_score
    def forward(self, segmented_sentences):
        # batch_tokens = []
        # max_length = -1
        # for sentence in segmented_sentences:
        #     tokens = torch.tensor([self.tokenizer.encode(
        #         sentence)[:self.max_sequence_length]], device=self.device)
        #     batch_tokens.append(tokens)
        #     if tokens.size(1) > max_length:
        #         max_length = tokens.size(1)

        # if pad:
        #     padded_batch_tokens = []
        #     for tokens in batch_tokens:
        #         if tokens.size(1) < max_length:
        #             end_ts = torch.tensor([[2]], dtype=torch.int, device=self.device)
        #             padded_tokens = torch.ones(
        #                 (1, max_length - tokens.size(1)), device=self.device, dtype=torch.int)
        #             tokens = torch.cat([tokens[:, :-1], padded_tokens, end_ts], dim=1)
        #         try:
        #             assert tokens.size(1) == max_length
        #         except Exception as e:
        #             print(f'Size exception: {tokens.size(1)}')
        #         padded_batch_tokens.append(tokens)
        #     batch_tokens = padded_batch_tokens

        # batch_tokens = torch.vstack(batch_tokens)
        # input_ids = self.tokenizer(segmented_sentences, padding=False)['input_ids']
        input_ids = self.tokenizer(segmented_sentences, padding=True, truncation=True, max_length=64)['input_ids']
        text_lengths = [len(sentence) for sentence in input_ids]
        input_ids = self.tokenizer(segmented_sentences, padding=True, truncation=True, max_length=64)[
            'input_ids']
        batch_tokens = torch.tensor(
            input_ids, dtype=torch.int, device=self.device)

        # with torch.no_grad():
        embeddings = self.encoder(batch_tokens).last_hidden_state

        embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, text_lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(embeddings)
        if self.rnn.bidirectional:
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        output = self.out(hidden)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        self.list_f1 = []
        self.list_f1_keras = []
        self.list_acc_keras = []
        self.encoder.train()
        self.out.train()
        sentences, labels = train_batch
        
        segmented_sentences = [self.segmentor.tokenize(
            sentence) for sentence in sentences]
        segmented_sentences = [' '.join(words[0])
                               for words in segmented_sentences]
        freq = torch.bincount(labels)
        weights = torch.max(freq) / freq
        # if weights.size(0) == 1:
        #     main_weight = torch.ones((1,), dtype=torch.float16, device=self.device)
        # else:
        #     main_weight = weights[1]
        # print(f'{freq} -> {weights}')
        output = self.forward(segmented_sentences)

        logsm = F.log_softmax(output, dim=1)
        prod = F.softmax(output, dim=1)
        # loss = F.nll_loss(logsm, labels, weights)
        loss = F.nll_loss(logsm, labels)
        pred = torch.argmax(prod, dim=1)

        self.train_acc(pred, labels)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sentences, labels = val_batch
        segmented_sentences = [self.segmentor.tokenize(
            sentence) for sentence in sentences]
        segmented_sentences = [' '.join(words[0])
                               for words in segmented_sentences]

        freq = torch.bincount(labels)
        weights = torch.max(freq) / freq
        output = self.forward(segmented_sentences)
        logsm = F.log_softmax(output, dim=1)
        
        loss = F.nll_loss(logsm, labels)
        prod = F.softmax(output, dim=1)
        pred = torch.argmax(prod, dim=1)
        self.valid_acc(pred, labels)

        a = self.f1(pred, labels)

        self.log('val_loss', loss)
        self.list_f1.append(a.cpu().numpy())
        print('F1 score:', sum(self.list_f1)/len(self.list_f1))
        self.log('val_acc', self.valid_acc, on_step=True, on_epoch=True)
        print('-------------------------------')
        acc_f1_score = self.flat_accuracy(pred, labels)

        self.list_f1_keras.append(acc_f1_score[-1])
        self.list_acc_keras.append(acc_f1_score[0])
        print('F1 score keras:', sum(self.list_f1_keras)/len(self.list_f1_keras))
        print('ACC score keras:', sum(self.list_acc_keras)/len(self.list_acc_keras))
        print('-------------------------------')
        return loss

    def predict_step(self, predict_batch, batch_idx):
        sentences = predict_batch
        segmented_sentences = [self.segmentor.tokenize(
            sentence) for sentence in sentences]
        segmented_sentences = [' '.join(words[0])
                               for words in segmented_sentences if len(words) > 0]

        preds = []
        probs = []
        for sentence in segmented_sentences:
            with torch.no_grad():
                output = self.forward([sentence])

            prob = F.softmax(output, dim=1)
            pred = torch.argmax(prob, dim=1).cpu().numpy()

            preds.append(pred[0].item())
            probs.append(prob[0][pred[0].item()].item())

        assert len(segmented_sentences) == len(preds) == len(probs)
        for idx, sentence in enumerate(segmented_sentences):
            self.prediction.append(
                (sentence, preds[idx], probs[idx]))
        return preds

    def test_step(self, predict_batch, batch_idx):
        sentences = predict_batch
        segmented_sentences = [self.segmentor.tokenize(
            sentence) for sentence in sentences]
        segmented_sentences = [' '.join(words[0])
                               for words in segmented_sentences if len(words) > 0]
        with torch.no_grad():
            batch_tokens = []
            for sentence in segmented_sentences:
                input_ids = torch.tensor(
                    [self.tokenizer.encode(sentence)], device=self.device)
                if input_ids.size(1) < self.max_sequence_length:
                    padded_tokens = torch.ones(
                        (1, self.max_sequence_length - input_ids.size(1)), device=self.device, dtype=torch.int)
                    input_ids = torch.cat([input_ids, padded_tokens], dim=1)

                assert input_ids.size(1) == self.max_sequence_length
                batch_tokens.append(input_ids)
            batch_tokens = torch.vstack(batch_tokens)
            embeddings = self.encoder(batch_tokens).last_hidden_state

        output = self.forward(embeddings)
        output = F.log_softmax(output, dim=1)
        prod = F.softmax(output, dim=1)
        pred = torch.argmax(prod, dim=1).cpu().numpy()
        for idx, sentence in enumerate(segmented_sentences):
            self.prediction.append((sentence, pred[idx]))
        return pred
