from transformers import BertTokenizer,BertModel,BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import logging
logging.set_verbosity_error()
import pandas as pd
import numpy as np
from datasets import load_dataset
import wandb
import copy
import logging
import threading
import sys
from transformers import set_seed
from datetime import datetime
import torch.nn.functional as F




class MyDataset(Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self,idx):
        item = {key:torch.Tensor(val[idx]) for key,val in self.encodings.items()}
        item["label"] = int(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class my_bert_model(nn.Module):
    def __init__(self, freeze_bert=False, hidden_size=768):
        super().__init__()
        config = BertConfig.from_pretrained('bert-base-uncased')
        config.update({'output_hidden_states':True})
        self.bert = BertModel.from_pretrained("bert-base-uncased",config=config)
        self.fc = nn.Linear(hidden_size, 2)
        
        #是否冻结bert，不让其参数更新
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        all_hidden_states = outputs[1]
        
        #因为输出的是所有层的输出，是元组保存的，所以转成矩阵
        # concat_last_4layers = torch.cat((all_hidden_states[-1],   #取最后4层的输出
        #                                  all_hidden_states[-2], 
        #                                  all_hidden_states[-3], 
        #                                  all_hidden_states[-4]), dim=-1)
        
        # cls_concat = concat_last_4layers[:,0,:]   #取 [CLS] 这个token对应的经过最后4层concat后的输出
        result = self.fc(all_hidden_states)
        
        return result



class Trainer:
    def __init__(self, model, gpu_num, loss_fun,lrr, trad_off,celue, optimizer, max_epoch = 30, early_stop_patience = 5 ,lr = 1e-5, warm_up_steps = 200, batch_size = 64):
        self.gpu_num = gpu_num
        self.model = model.to(self.gpu_num)
        self.lrr = lrr
        self.celue = celue
        self.trad_off = trad_off
        self.batch_size = batch_size
        self.best_model = self.model.state_dict()
        self.early_stop_patience = early_stop_patience
        self.loss_fun = loss_fun().to(self.gpu_num)
        self.optim = optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        self.max_epoch = max_epoch
        self.warm_up_steps = warm_up_steps
        self.early_stop_counter = 0 # 判断是否早停
        self.get_dataloader()

        self.scheduler = get_linear_schedule_with_warmup(self.optim, num_warmup_steps = self.warm_up_steps, num_training_steps = len(self.train_dataloader))
        self.best_valid_loss = 100000

        wandb.init(project='text_class_sst_222seed',name = "{} grad lr_{}_trad_off_{}".format(self.celue,self.lrr,self.trad_off))
        logging.basicConfig(filename="/home/ubuntu/bis-varis-lemao/text_classification/{}/{} lr_{}_trad_off_{}.log".format(self.celue,self.celue,self.lrr,self.trad_off), level=logging.INFO)
        logging.info("Trainer init over")

    def compute_KL_grade(self,eta,g1,g2,input_ids,attention_mask,labels):
        batch_num = input_ids.shape[0]
        final_grad = []
        for i in self.model.parameters():
            final_grad.append(torch.zeros_like(i).requires_grad_(False))

        model_g1 = copy.deepcopy(self.model)
        model_g1.eval()
        with torch.no_grad():
            temp = 0
            for i in model_g1.parameters():
                i -= eta*g1[temp]
                temp += 1
        output1 = model_g1(input_ids,attention_mask)
        output1_kl = torch.softmax(output1,dim = 1)

        model_g2 = copy.deepcopy(self.model)
        model_g2.eval()
        with torch.no_grad():
            temp = 0
            for i in model_g2.parameters():
                i -= eta*g2[temp]
                temp += 1
        output2 = model_g2(input_ids,attention_mask)
        output2_kl = torch.softmax(output2,dim = 1)

        def symmetric_kl_divergence(tensor1, tensor2):
            kl_div1 = F.kl_div(tensor1.log(), tensor2, reduction='none').sum(dim=1)
            kl_div2 = F.kl_div(tensor2.log(), tensor1, reduction='none').sum(dim=1)
            symmetric_kl = 0.5 * kl_div1 + 0.5 * kl_div2
            return symmetric_kl
        
        symmetric_kl = symmetric_kl_divergence(output1_kl, output2_kl).sum()
        symmetric_kl.backward()

        final_grad = []
        for i,j in enumerate(model_g1.parameters()):
            final_grad.append(j.grad.detach()+list(model_g2.parameters())[i].grad.detach())


        # for i in range(output1.shape[0]):
        #     for j in range(output1.shape[1]):
        #         p1 = output1_kl[i][j]
        #         p2 = output2_kl[i][j]
        #         p1_grad = torch.autograd.grad(outputs = p1,inputs = model_g1.parameters(),retain_graph=True)
        #         p2_grad = torch.autograd.grad(outputs = p2,inputs = model_g2.parameters(),retain_graph=True)
        #         p1_log_grad = torch.autograd.grad(outputs = torch.log(p1),inputs = model_g1.parameters(),retain_graph=True)
        #         p2_log_grad = torch.autograd.grad(outputs = torch.log(p2),inputs = model_g2.parameters(),retain_graph=True)
        #         p1_logp1 = torch.autograd.grad(outputs = p1*torch.log(p1),inputs = model_g1.parameters(),retain_graph=True)
        #         p2_logp2 = torch.autograd.grad(outputs = p2*torch.log(p2),inputs = model_g2.parameters(),retain_graph=True)
        #         for k in range(len(final_grad)):
        #             final_grad[k] += -torch.log(p2.detach())*p1_grad[k].detach() - p1.detach()*p2_log_grad[k].detach() + p1_logp1[k].detach() -torch.log(p1.detach())*p2_grad[k].detach() - p2.detach()*p1_log_grad[k].detach() + p2_logp2[k].detach()
        #         del p1_grad, p2_grad, p1_log_grad, p2_log_grad, p1_logp1, p2_logp2

        # final_grad = [i / output1.shape[0] for i in final_grad]

        # logging.info("{}----finished one batch".format(datetime.now()))
        return final_grad

    def compute_final_grade(self,eta,g1,g2,input_ids,attention_mask,labels):
        model_g1 = copy.deepcopy(self.model)
        model_g1.eval()

        with torch.no_grad():
            temp = 0
            for i in model_g1.parameters():
                i -= eta*g1[temp]
                temp += 1
        loss_g1 = self.loss_fun(model_g1(input_ids,attention_mask), labels)
        grad_22 = torch.autograd.grad(outputs=loss_g1,inputs=model_g1.parameters())
        model_g2 = copy.deepcopy(self.model)
        model_g2.eval()
        with torch.no_grad():
            temp = 0
            for i in model_g2.parameters():
                i -= eta*g2[temp]
                temp += 1
        loss_g2 = self.loss_fun(model_g2(input_ids,attention_mask), labels)
        grad_33 = torch.autograd.grad(outputs=loss_g2,inputs=model_g2.parameters())
        # for i in range(5):
        #     print("model_g1",list(model_g1.parameters())[i],"model_g2",list(model_g2.parameters())[i])
        # s = str(x)
        # decimal_pos = s.find('.')
        # if decimal_pos != -1:
        #     power = 10 ** (decimal_pos)
        #     y = float(s[decimal_pos+1:]) / power
        #     print(y)
        wandb.log({"abs(loss_g1-loss_g2)": abs(loss_g1-loss_g2)})
        grad_bbb = [2*(loss_g1-loss_g2)*(grad_22[i]-grad_33[i]) for i in range(len(grad_22))]
        return grad_bbb

    def train_step_with_KL(self,epoch):
        self.model.train()
        total_train_loss = 0
        iter_num = 0
        total_iter = len(self.train_dataloader)
        for batch in self.train_dataloader:
            # 正向传播
            self.optim.zero_grad()
            input_ids = batch['input_ids'].to(self.gpu_num).type(torch.int)
            attention_mask = batch['attention_mask'].to(self.gpu_num).type(torch.int)
            labels = batch['label'].to(self.gpu_num)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            this_batch_size = outputs.shape[0]
            loss = self.loss_fun(outputs, labels)

            loss1 = self.loss_fun(outputs[:this_batch_size//2], labels[:this_batch_size//2])
            loss2 = self.loss_fun(outputs[this_batch_size//2:], labels[this_batch_size//2:])

            grad_up = [grad.detach() for grad in torch.autograd.grad(outputs=loss1,inputs=self.model.parameters(),retain_graph=True,allow_unused=True)]
            grad_down = [grad.detach() for grad in torch.autograd.grad(outputs=loss2,inputs=self.model.parameters(),retain_graph=True)]
            final_grad = self.compute_KL_grade(self.lrr,grad_up,grad_down,input_ids,attention_mask,labels)

            total_train_loss += loss.item()

            loss.backward()
            for i,j in enumerate(self.model.parameters()):
                j.grad = (1-self.trad_off)*j.grad + self.trad_off*final_grad[i]

            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)   #梯度裁剪，防止梯度爆炸


            # 参数更新
            self.optim.step()
            self.scheduler.step()

            iter_num += 1
            if(iter_num % 30==0):
                logging.info("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
        logging.info("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(self.train_dataloader)))
        wandb.log({"average_training_loss":total_train_loss/len(self.train_dataloader)})        

    def train_step(self,epoch):
        self.model.train()
        total_train_loss = 0
        iter_num = 0
        total_iter = len(self.train_dataloader)
        for batch in self.train_dataloader:
            # 正向传播
            self.optim.zero_grad()
            input_ids = batch['input_ids'].to(self.gpu_num).type(torch.int)
            attention_mask = batch['attention_mask'].to(self.gpu_num).type(torch.int)
            labels = batch['label'].to(self.gpu_num)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = self.loss_fun(outputs, labels)                
            total_train_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)   #梯度裁剪，防止梯度爆炸

            # 参数更新
            self.optim.step()
            self.scheduler.step()

            iter_num += 1
            if(iter_num % 30==0):
                logging.info("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))

        logging.info("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(self.train_dataloader)))
        wandb.log({"average_training_loss":total_train_loss/len(self.train_dataloader)})        

    def train_step_with_g1_g2(self,epoch):

        self.model.train()
        total_train_loss = 0
        iter_num = 0
        total_iter = len(self.train_dataloader)
        for batch in self.train_dataloader:
            # 正向传播
            self.optim.zero_grad()
            input_ids = batch['input_ids'].to(self.gpu_num).type(torch.int)
            attention_mask = batch['attention_mask'].to(self.gpu_num).type(torch.int)
            labels = batch['label'].to(self.gpu_num)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            
            this_batch_size = outputs.shape[0]
            loss = self.loss_fun(outputs, labels)

            loss1 = self.loss_fun(outputs[:this_batch_size//2], labels[:this_batch_size//2])
            loss2 = self.loss_fun(outputs[this_batch_size//2:], labels[this_batch_size//2:])

            grad_up = [grad.detach() for grad in torch.autograd.grad(outputs=loss1,inputs=self.model.parameters(),retain_graph=True,allow_unused=True)]
            grad_down = [grad.detach() for grad in torch.autograd.grad(outputs=loss2,inputs=self.model.parameters(),retain_graph=True)]
            grad_end = self.compute_final_grade(self.lrr,grad_up,grad_down,input_ids,attention_mask,labels)

            total_train_loss += loss.item()

            loss.backward()
            for i,j in enumerate(self.model.parameters()):
                j.grad = (1-self.trad_off)*j.grad + self.trad_off*grad_end[i]

            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)   #梯度裁剪，防止梯度爆炸

            # 参数更新
            self.optim.step()
            self.scheduler.step()

            iter_num += 1
            if(iter_num % 100==0):
                logging.info("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))

        logging.info("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(self.train_dataloader)))
        wandb.log({"average_training_loss":total_train_loss/len(self.train_dataloader)})

    def flat_accuracy(self,preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def valid_step(self,epoch):
        self.model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        with torch.no_grad():
            for batch in self.valid_dataloader:
                # 正常传播
                input_ids = batch['input_ids'].to(self.gpu_num).type(torch.int)
                attention_mask = batch['attention_mask'].to(self.gpu_num).type(torch.int)
                labels = batch['label'].to(self.gpu_num)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.loss_fun(outputs, labels)
                logits = outputs
                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(self.valid_dataloader)
        logging.info("Accuracy: %.4f" % (avg_val_accuracy))
        avg_loss = total_eval_loss/len(self.valid_dataloader)
        logging.info("Average valid loss: %.4f"%(avg_loss))
        wandb.log({"valid_loss": avg_loss, "valid_accuracy": avg_val_accuracy})

        if(avg_loss<self.best_valid_loss):
            self.best_valid_loss = avg_loss
            self.best_model = self.model.state_dict()
            self.early_stop_counter = 0
            logging.info("Got best valid loss {}".format(avg_loss))
            torch.save(self.best_model,"/home/ubuntu/bis-varis-lemao/text_classification/{}/lr_{}_trad_off_{}.pt".format(self.celue,self.lrr,self.trad_off))
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.early_stop_patience:
                logging.info('Early stopping triggered!')
                return 1
        logging.info("-------------------------------")
        return 0

    def get_best_model(self):
        return self.best_model.state_dict()


    def get_dataloader(self, is_shuffle = True, batch_size = 64):
        dataset = load_dataset("sst2")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        train_encoding = tokenizer(dataset["train"]["sentence"], truncation=True, padding=True, max_length=512)
        test_encoding = tokenizer(dataset["test"]["sentence"], truncation=True, padding=True, max_length=512)
        valid_encoding = tokenizer(dataset["validation"]["sentence"], truncation=True, padding=True, max_length=512)

        train_dataset = MyDataset(train_encoding,dataset["train"]["label"])
        test_dataset = MyDataset(test_encoding,dataset["test"]["label"])
        valid_dataset = MyDataset(valid_encoding,dataset["validation"]["label"])

        self.train_dataloader = DataLoader(train_dataset,shuffle = is_shuffle,batch_size = batch_size)
        self.test_dataloader = DataLoader(test_dataset,shuffle = is_shuffle,batch_size = batch_size)
        self.valid_dataloader = DataLoader(valid_dataset,shuffle = is_shuffle,batch_size = batch_size)

def main(lrr,trad_off,gpu_num,temp_celue):
    torch.manual_seed(222)
    set_seed(222)
    my_model = my_bert_model()
    trainer = Trainer(model = my_model, gpu_num = gpu_num, loss_fun = nn.CrossEntropyLoss , optimizer = AdamW, max_epoch = 30,lrr = lrr, trad_off = trad_off,celue = temp_celue)
    for i in range(30):
        trainer.train_step_with_KL(i)
        if_over = trainer.valid_step(i)
        if if_over==1:
            break
    torch.save(trainer.get_best_model(),"/home/ubuntu/bis-varis-lemao/text_classification/{}/lr_{}_trad_off_{}.pt".format(temp_celue,lrr,trad_off))


# for i in [1e-4,1e-5,1e-6]:
#     for j in [1e-1,1e-2,1e-3,1e-4,1e-5]:
#         print(i,j)
#         threading.Thread(target=main,args=(i,j,t,)).start()
#     t+=1


t = 0
main(1e-3,0.05,1,"KL_model1_sum_model2")