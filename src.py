import torch
import time
import pandas as pd 
import numpy as np
from transformers import BertTokenizer,BertModel
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from settings import get_module_logger
from sklearn.metrics import f1_score
from settings import get_module_logger
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import defaultdict
from flag import get_parser
parser = get_parser()
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
logger = get_module_logger(__name__)

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Dataset:
    def __init__(self, text, targets):
        self.text = text 
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
        self.max_length = args.max_len
        self.targets = targets

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation_strategy="longest_first",
            pad_to_max_length=True,
            truncation=True
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
       
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[item], dtype=torch.float)
        }

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = BertModel.from_pretrained(args.pretrained_model_name,output_hidden_states=True)
        self.drop_out = nn.Dropout(args.dropout) 
        self.l0 =  nn.Linear(args.bert_hidden * 4, args.classes)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def _get_cls_vec(self, vec):
        return vec[:,0,:].view(-1, args.bert_hidden)

    def forward(self,ids,attention_mask, token_type_ids):
        _, _, hidden_states = self.bert(
            ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        vec1 = self._get_cls_vec(hidden_states[-1])
        vec2 = self._get_cls_vec(hidden_states[-2])
        vec3 = self._get_cls_vec(hidden_states[-3])
        vec4 = self._get_cls_vec(hidden_states[-4])

        out = torch.cat([vec1, vec2, vec3, vec4], dim=1)
        out = self.drop_out(out)
        logits = self.l0(out)
        return logits

def loss_fn(y_pred, y_true):
    return nn.BCEWithLogitsLoss()(y_pred, y_true.view(-1,1))

def train_fn(data_loader, model, optimizer, device, scheduler, n_examples):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    start = time.time()
    train_losses = []
    fin_targets = []
    fin_outputs = []
    for bi, d in enumerate(tk0):
        ids = d["ids"]
        mask = d["mask"]
        token_type_ids = d["token_type_ids"]
        targets = d["targets"]
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        # Reset gradients
        model.zero_grad()

        outputs = model(
            ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        loss = loss_fn(outputs, targets)
        train_losses.append(loss.item())

        outputs = torch.round(nn.Sigmoid()(outputs)).squeeze()
        targets = targets.squeeze()
        outputs = outputs.cpu().detach().numpy().tolist()
        targets = targets.cpu().detach().numpy().tolist()

        train_f1 = f1_score(outputs, targets, average='weighted')
        end = time.time()

        f1 = np.round(train_f1.item(), 3)
        if (bi % 100 == 0 and bi != 0) or (bi == len(data_loader) - 1) :
            logger.info(f'bi={bi}, Train F1={f1},Train loss={loss.item()}, time={end-start}')
        
        loss.backward() # Calculate gradients based on loss
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() # Adjust weights based on calculated gradients
        scheduler.step() # Update scheduler
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss = losses.avg)
        fin_targets.extend(targets) 
        fin_outputs.extend(outputs)
    f1 = f1_score(fin_outputs, fin_targets, average='weighted')
    f1 = np.round(f1.item(), 3)
    return f1, np.mean(train_losses)


def run():

    df_train = pd.read_csv(args.training_file).dropna().reset_index(drop=True)
    df_valid = pd.read_csv(args.validation_file).dropna().reset_index(drop=True)

    logger.info("train len - {} valid len - {}".format(len(df_train), len(df_valid)))

    train_dataset = Dataset(
        text=df_train.text.values,
        targets=df_train.target.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = Dataset(
        text=df_valid.text.values,
        targets=df_valid.target.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        shuffle=False,
        num_workers=2
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BERT()
    model.to(device)
   

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / args.train_batch_size * args.epochs)

    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_train_steps
    )
    print("STARTING TRAINING ...\n")
    logger.info("{} - {}".format("STARTING TRAINING",args.model_specification))
    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch + 1}/{args.epochs}')
        logger.info('-' * 10)

        train_acc, train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler, len(df_train))
        logger.info(f'Train loss {train_loss} accuracy {train_acc}')
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
    del model, train_data_loader, valid_data_loader, train_dataset, valid_dataset
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.info("##################################### Task End ############################################")

if __name__ == "__main__":
    run()