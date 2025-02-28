# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import math
import torch.utils.data as Data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 1. prepare the data
# setences example data
'''
    P : using for padding
    S : start
    E : End
'''
sentences = [['我 是 研 究 生', 'S I am a master student', 'I am a master student E'],
             ['我 喜 欢 机 器 人', 'S I like robotics', 'I like robotics E'],
             ['我 在 德 国 P', 'S I live in Germany', 'I live in Germany E'],
             ['我 喜 欢 牛 肉', 'S I like beef', 'I like beef E'],
             ['我 喜 欢 ', 'S I like P', 'I like E']]
# Mapping of source language characters to indices
src_vocab = {'P':0, '我':1, '是':2, '研':3, '究':4, '生':5, '喜':6, '欢':7, '机':8, '器':9, '人':10, '在':11, '德':12, '国':13, '牛':14, '肉':15}
# Mapping of indices to source language characters
src_idx2word = {src_vocab[key]: key for key in src_vocab}
# Calculate the source language vocabulary size
src_vocab_size = len(src_vocab)

# for the taget language
tgt_vocab = {'P':0, 'S':1, 'E':2, 'I':3, 'am':4, 'a':5, 'master':6, 'student':7, 'like':8, 'robotics':9, 'live':10, 'in':11, 'Germany':12, 'beef':13}
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}
tgt_vocab_size = len(tgt_vocab)
src_len = len(sentences[0][0].split(" "))
tgt_len = len(sentences[0][1].split(" "))

# 2. Convert sentences into dictionary indices
def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []  

    # find the largest lenth
    max_src_len = max(len(s[0].split()) for s in sentences)
    max_tgt_len = max(len(s[1].split()) for s in sentences)

    for i in range(len(sentences)):
        enc_input = [src_vocab[n] for n in sentences[i][0].split()]
        dec_input = [tgt_vocab[n] for n in sentences[i][1].split()]
        dec_output = [tgt_vocab[n] for n in sentences[i][2].split()]

        # **padding**
        enc_input += [0] * (max_src_len - len(enc_input))  # compensate source 
        dec_input += [0] * (max_tgt_len - len(dec_input))  # compensate target
        dec_output += [0] * (max_tgt_len - len(dec_output))  # compensate target output
        enc_inputs.append(enc_input)  
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

# reuse make_data
enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

# 3. define function of dataset
class MyDataSet(Data.Dataset):
  def __init__(self, enc_inputs, dec_inputs, dec_outputs):
     super(MyDataSet,self).__init__()
     self.enc_inputs = enc_inputs   
     self.dec_inputs = dec_inputs
     self.dec_outputs = dec_outputs

  def __len__(self):
    return self.enc_inputs.shape[0]    # return the size of dataset

  def __getitem__(self, idx):
     return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]  # this way is uesd to index samples in a dataset

loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs),2,True)

for batch in loader:
    print(batch)  # output (batch_enc_inputs, batch_dec_inputs, batch_dec_outputs)


# 4. setting the parameters (The parameters in the paper were used)
d_model = 512
d_ff = 2048
d_k = d_v = 64
n_layers = 6
n_heads = 8

# 5. define the positional information
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # shape: (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)  # even dimension
        pe[:, 1::2] = torch.cos(pos * div_term)  # odd dimension

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，increase the batch dimension
        self.register_buffer('pe', pe)  # 

    def forward(self, enc_inputs):
        enc_inputs = enc_inputs + self.pe[:, :enc_inputs.size(1), :].to(enc_inputs.device)  # 匹配输入设备
        return self.dropout(enc_inputs)
    
# 6. Prevent the model from focusing on the location of padding (PAD) when calculating attention
def get_attn_pad_mask(seq_q, seq_k):                       # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)          #
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # expend to multidimension

# 7. use for decoder : Subsequent Mask
def get_attn_subsequence_mask(seq):                               # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)          
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()  #  [batch_size, tgt_len, tgt_len]
    return subsequence_mask     

# 8. Calculate attention information, residual and normalization
# 8.1 Scaled Dot-Production Attention : in order to prevent Gradient explosion
class ProductAttention(nn.Module):
    def __init__(self):
        super(ProductAttention,self).__init__()

    def forward(self,Q,K,V,at_mask):
        scores = torch.matmul(Q,K.transpose(-1,-2))/np.sqrt(d_k)
        scores.masked_fill_(at_mask, -1e9)
        at = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(at,V)
        return context, at

# 8.2 MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias = False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    
    def forward(self, input_Q, input_K, input_V, at_mask):
        residual, batch_size = input_Q, input_Q.size(0)  #Residual Connection
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        at_mask = at_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)              # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        context, at = ProductAttention()(Q, K, V, at_mask)          # context: [batch_size, n_heads, len_q, d_v]
                                                                                 # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), at

# 9. FeedForwoardNet
class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias= False)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm(output + residual)

# 10. single encoder
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer,self).__init__()
        self.enc_self_at = MultiHeadAttention()
        self.pos_ffn = FeedForwardNet()

    def forward(self, enc_inputs, enc_at_mask):
        enc_outputs, at = self.enc_self_at(enc_inputs, enc_inputs, enc_inputs, enc_at_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, at

# 11. whole encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)                     
        self.pos_emb = PositionalEncoding(d_model)                               
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):                                               # enc_inputs: [batch_size, src_len]
        enc_outputs = self.src_emb(enc_inputs)                                   # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs)                                  # enc_outputs: [batch_size, src_len, d_model]   
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)           # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)  # enc_outputs :   [batch_size, src_len, d_model], 
                                                                                 # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# 12. single decoder
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = FeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask): # dec_inputs: [batch_size, tgt_len, d_model]
                                                                                       # enc_outputs: [batch_size, src_len, d_model]
                                                                                       # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
                                                                                       # dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, 
                                                 dec_inputs, dec_self_attn_mask)   # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                   # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, 
                                                enc_outputs, dec_enc_attn_mask)    # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                   # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs = self.pos_ffn(dec_outputs)                                    # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

# 13. whole decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):                               # dec_inputs: [batch_size, tgt_len]
                                                                                          # enc_intpus: [batch_size, src_len]
                                                                                          # enc_outputs: [batsh_size, src_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)                                            # [batch_size, tgt_len, d_model]       
        dec_outputs = self.pos_emb(dec_outputs).cuda()                                    # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()         # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()     # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + 
                                       dec_self_attn_subsequence_mask), 0).cuda()         # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)                     # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:                             # dec_outputs: [batch_size, tgt_len, d_model]
                                                              # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
                                                              # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# 14. Transformer
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder().cuda()
        self.Decoder = Decoder().cuda()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).cuda()
    def forward(self, enc_inputs, dec_inputs):                         # enc_inputs: [batch_size, src_len]  
                                                                       # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)         # enc_outputs: [batch_size, src_len, d_model], 
                                                                       # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)                       # dec_outpus    : [batch_size, tgt_len, d_model], 
                                                                       # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], 
                                                                       # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)                      # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns

# 15. define the model
model = Transformer().to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)   
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)



# 16. train the transformer
for epoch in range(50):
    for enc_inputs, dec_inputs, dec_outputs in loader:
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
        
        outputs, enc_self_attns, dec_self_attns = model(enc_inputs, dec_inputs)
        outputs = outputs.view(-1, tgt_vocab_size)
        loss = criterion(outputs, dec_outputs.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent the gradient explode
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")


def test_model(model, sentences):
    """
    test the model
    """
    model.eval()
    test_sentence = input("Please enter the sentence in Chinese : ") # input the Chinese sentence
    print(f"input the test sentence: {test_sentence}")

    # 1. transfer the input sentence `enc_inputs`
    enc_input = [src_vocab[n] for n in test_sentence.split()]  # token
    enc_input += [0] * (src_len - len(enc_input))  # compensate
    enc_input = torch.LongTensor(enc_input).unsqueeze(0).to(device)  # add the batch dimension

    # 2. `dec_input` first input `S`
    dec_input = torch.LongTensor([[tgt_vocab['S']]]).to(device)

    # 3. **auto create sentence**
    while True:
        # run Transformer
        dec_logits, enc_self_attns, dec_self_attns = model(enc_input, dec_input)
        dec_logits = dec_logits.squeeze(0)  # delete batch dimension

        # choose the highest proximately token as output
        next_token = torch.argmax(dec_logits[-1], dim=-1).item()
        dec_input = torch.cat([dec_input, torch.LongTensor([[next_token]]).to(device)], dim=-1)

        # **'E' = end**
        if next_token == tgt_vocab['E']:
            break

    # 4. **transfer token to words**
    translated_tokens = [idx2word[n.item()] for n in dec_input.squeeze() if n.item() not in [tgt_vocab['S'], tgt_vocab['E']]]
    translated_sentence = ' '.join(translated_tokens)
    print(f"translation result: {translated_sentence}")

# **run the test model**
test_model(model, sentences)
