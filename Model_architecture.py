### README
'''
Python (version 3.9.7) and Pytorch (version 1.8.2) were used for developing deep learning framework.
Statistical analysis were performed using Scikit-learn (version 1.0.2)

Note that the original version of Temporal Fusion Transformer (TFT) was developed by Tensorflow.
(https://github.com/google-research/google-research/tree/master/tft)

The current model has several differences between the original model, like below.
- In RNN module, GRU is used instead of LSTM.
- Multi-horizon forecasting, continuous (with quartile) prediction. -> Many-to-many, binary prediction
- Future sequence masking in Transformer module (the model should not know the future information).
- Allows variable-length timeseries data.
'''



### Package loading
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data



### Variable and label definition
# All columns (variables are identical to those used in the manuscript)
name_columns = [ # Absolute difference of days between the current session and the matched previous sessions (It is 0 in the current session).
                 'HD_day', 
                 # Demographics
                 'PT_age', 'PT_female',
                 # Hemodialysis settings
                 'HD_type', 'HD_dfr', 'HD_prewt', 'HD_priming', 'HD_dialysate', 'HD_dialyzer', 'HD_flux', 'HD_surface', 'HD_vascular', 
                 # Elapsed minutes from the initiation of hemodialysis.
                 'HD_ctime',
                 # Hemodialysis machine information. '_p' means padding information.
                 'HD_SBP', 'HD_SBP_p', 'HD_DBP', 'HD_DBP_p', 'HD_HR', 'HD_HR_p', 'HD_RR', 'HD_RR_p', 'HD_BT', 'HD_BT_p', 'HD_SPO2', 'HD_SPO2_p', 'HD_BFR', 'HD_BFR_p', 'HD_UFT', 'HD_UFT_p', 'HD_VP', 'HD_VP_p', 'HD_TMP', 'HD_TMP_p',
                 # Clinical information
                 'DM', 'HTN', 'CAD', 'AF', 'LC', 'GN', 'Cancer', 'Donor', 'Recipient',
                 # Laboratory findings
                 'WBC', 'Hb', 'HCT', 'PLT', 'Chol', 'Prot', 'Alb', 'Tbil', 'AST', 'ALT', 'ALP', 'PT', 'aPTT', 'BUN', 'Cr', 'Na', 'K', 'CO2', 'Cl', 'Phos', 'Ca', 'UA', 'CRP', 'CysC', 'Glu', 'LDH',
                 # Medication
                 'Anti_HTN', 'Diuretics', 'Statin', 'Anti_PLT', 'Anti_coag', 'OHA', 'INS', 'EPO', 'UA_lowering', 'Pbinder',
                 # Study outcomes
                 'IDH-1', 'IDH-2', 'IDHTN']

# Time-invariant variables vs. Time-varying variables
'''
Note that time-invariant variables were different in processing the current session and the matched previous sessions,
- Some variables did not differ in these sessions (such as patient age, sex, DM, HTN, etc.), and they were excluded in the variables of previous sessions.
- 'HD_day' variable was always 0 in the current session, and it was excluded in the current session.
'''
name_invariant_curr= ['PT_age', 'PT_female', 'HD_type', 'HD_dfr', 'HD_prewt', 'HD_priming', 'HD_dialysate', 'HD_dialyzer', 'HD_flux', 'HD_surface', 'HD_vascular', 'DM', 'HTN', 'CAD', 'AF', 'LC', 'GN', 'Cancer', 'Donor', 'Recipient', 'WBC', 'Hb', 'HCT', 'PLT', 'Chol', 'Prot', 'Alb', 'Tbil', 'AST', 'ALT', 'ALP', 'PT', 'aPTT', 'BUN', 'Cr', 'Na', 'K', 'CO2', 'Cl', 'Phos', 'Ca', 'UA', 'CRP', 'CysC', 'Glu', 'LDH', 'Anti_HTN', 'Diuretics', 'Statin', 'Anti_PLT', 'Anti_coag', 'OHA', 'INS', 'EPO', 'UA_lowering', 'Pbinder']
name_invariant_prev= ['HD_day', 'HD_type', 'HD_dfr', 'HD_prewt', 'HD_priming', 'HD_dialysate', 'HD_dialyzer', 'HD_flux', 'HD_surface', 'HD_vascular', 'WBC', 'Hb', 'HCT', 'PLT', 'Chol', 'Prot', 'Alb', 'Tbil', 'AST', 'ALT', 'ALP', 'PT', 'aPTT', 'BUN', 'Cr', 'Na', 'K', 'CO2', 'Cl', 'Phos', 'Ca', 'UA', 'CRP', 'CysC', 'Glu', 'LDH', 'Anti_HTN', 'Diuretics', 'Statin', 'Anti_PLT', 'Anti_coag', 'OHA', 'INS', 'EPO', 'UA_lowering', 'Pbinder']
name_varying = ['HD_ctime', 'HD_SBP', 'HD_SBP_p', 'HD_DBP', 'HD_DBP_p', 'HD_HR', 'HD_HR_p', 'HD_RR', 'HD_RR_p', 'HD_BT', 'HD_BT_p',  'HD_SPO2', 'HD_SPO2_p', 'HD_BFR', 'HD_BFR_p', 'HD_UFT', 'HD_UFT_p', 'HD_VP', 'HD_VP_p', 'HD_TMP', 'HD_TMP_p']

# Categorical variables (embedding layer) vs. Continuous variables (dense layer)
name_invariant_categorical = ['PT_female', 'HD_type', 'HD_priming', 'HD_dialysate', 'HD_dialyzer', 'HD_flux', 'HD_vascular', 'DM', 'HTN', 'CAD', 'AF', 'LC', 'GN', 'Cancer', 'Donor', 'Recipient', 'Anti_HTN', 'Diuretics', 'Statin', 'Anti_PLT', 'Anti_coag', 'OHA', 'INS', 'EPO', 'UA_lowering', 'Pbinder']
name_varying_categorical = ['HD_SBP_p', 'HD_DBP_p', 'HD_HR_p', 'HD_RR_p', 'HD_BT_p', 'HD_SPO2_p', 'HD_BFR_p', 'HD_UFT_p', 'HD_VP_p', 'HD_TMP_p']

name_invariant_curr_categorical = [x for x in name_invariant_curr if x in name_invariant_categorical]
name_invariant_prev_categorical = [x for x in name_invariant_prev if x in name_invariant_categorical]
name_invariant_categorical = list(set(name_invariant_curr_categorical) | set(name_invariant_prev_categorical))
name_varying_categorical = [x for x in name_varying if x in name_varying_categorical]

name_invariant_curr_continuous = [x for x in name_invariant_curr if x not in name_invariant_categorical]
name_invariant_prev_continuous = [x for x in name_invariant_prev if x not in name_invariant_categorical]
name_varying_continuous = [x for x in name_varying if x not in name_varying_categorical]

# Labels
name_label = ['IDH-1', 'IDH-2', 'IDHTN']

# Get indexes from variables
def get_idx_column(name_column, name_columns = name_columns):

    idx_column = [name_columns.index(x) for x in name_column]
    return idx_column

idx_invariant_curr = get_idx_column(name_invariant_curr)
idx_invariant_curr_categorical = get_idx_column(name_invariant_curr_categorical, name_invariant_curr)
idx_invariant_curr_continuous = get_idx_column(name_invariant_curr_continuous, name_invariant_curr)

idx_invariant_prev = get_idx_column(name_invariant_prev)
idx_invariant_prev_categorical = get_idx_column(name_invariant_prev_categorical, name_invariant_prev)
idx_invariant_prev_continuous = get_idx_column(name_invariant_prev_continuous, name_invariant_prev)

idx_invariant_categorical = get_idx_column(name_invariant_categorical)

idx_varying = get_idx_column(name_varying)
idx_varying_categorical = get_idx_column(name_varying_categorical, name_varying)
idx_varying_continuous = get_idx_column(name_varying_continuous, name_varying)

idx_label = get_idx_column(name_label)



### Generate pseudodata
'''
About continuous & Categorical variables
- Continuous variables were normalized by the mean and the standard deviation from the training set.
- Categorical variables were processed by embedding layers.

About time-invariant variables
- Only the first values of time-invariant variables in multiple timestamps were used.

About time-varying variables
- The dataset consisted of variable-length time-series data.
- Missing timestamps were padded with a padding value.
'''

# Define shape
n_session = 1000 # The number of hemodialysis sessions (302,774 in the manuscript)
n_column = len(name_columns) # The number of all variables and study outcomes
n_timestamp = 56 # The maximum number of timestamps (it is the same to the original data of the manuscript)
n_matched_session = 6 # The number of the current and the matched five previous sessions
v_padding = 100 # Padding value for post-padding in variable-length data
v_absence = -1 # If there is no previous session, this value will be padded.

n_invariant_curr = len(idx_invariant_curr)
n_invariant_prev = len(idx_invariant_prev)
n_varying = len(idx_varying)
n_label = len(idx_label)
n_prev_session = n_matched_session-1

# Create pseudodata array
datasheet_hd = np.random.random((n_session, n_timestamp, n_column, n_matched_session))
# datasheet_hd.shape # shape = (n_session, n_timestamp, n_column, n_matched_session)
# datasheet_hd[:,:,:,0] # It will be used as the current session.
# datasheet_hd[:,:,:,1:] # It will be used as the five previous sessions.

# Create categorical variables with randomly selected levels between 2 and 10
# ... In time-invariant variables, only the first value of the timestamps will be used.
max_invariant_categorical = np.random.randint(2, 10, size=len(idx_invariant_categorical))
max_invariant_curr_categorical = max_invariant_categorical[[name_invariant_categorical.index(x) for x in name_invariant_curr_categorical]]
max_invariant_prev_categorical = max_invariant_categorical[[name_invariant_categorical.index(x) for x in name_invariant_prev_categorical]]
for (t_idx, t_max) in zip(idx_invariant_categorical, max_invariant_categorical):
    datasheet_hd[:,0,t_idx,:] = np.random.randint(0, t_max, size=(n_session, n_matched_session))

# ... In time-varying categorical variables, only padding columns were categorical in the manuscript, though model can accept categorical values with more levels.
max_varying_categorical = np.repeat(2, repeats=len(idx_varying_categorical))
for (t_idx, t_max) in zip(idx_varying_categorical, max_varying_categorical):
    datasheet_hd[:,:,t_idx,:] = np.random.randint(0, t_max, size=(n_session, n_timestamp, n_matched_session))

# ... In labels, binary classification were used.
max_label = np.repeat(2, repeats=len(idx_label))
for (t_idx, t_max) in zip(idx_label, max_label):
    datasheet_hd[:,:,t_idx,:] = np.random.randint(0, t_max, size=(n_session, n_timestamp, n_matched_session))

# ... The dataset consisted of variable-length time-series data.
for t_num in range(n_matched_session):
    for t_sess in range(n_session):
        n_valid_timestamp = np.random.randint(2, n_timestamp)
        datasheet_hd[t_sess,n_valid_timestamp:,:,t_num] = v_padding

# ... Some dataset did not have previous sessions.
for t_sess in range(n_session):
    n_valid_previous = np.random.randint(1, n_matched_session)
    datasheet_hd[t_sess,:,:,n_valid_previous:] = v_absence



### Dataloader
class Custom_Dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_total = self.dataset.shape[0]

    def __len__(self):
        return self.n_total
    
    def __getitem__(self, index):
        npy_t = self.dataset[index,:,:,:].astype('float')
        return npy_t

def Train_loader(dataset):
    data_loader = Custom_Dataset(dataset)
    return data.DataLoader(data_loader, batch_size=100, shuffle=True)

def Test_loader(dataset):
    data_loader = Custom_Dataset(dataset)
    return data.DataLoader(data_loader, batch_size=100, shuffle=False)

train_loader = Train_loader(datasheet_hd[:800,:,:]) # Training/Test set splitting
test_loader = Test_loader(datasheet_hd[800:,:,:])
npy = next(iter(train_loader))

# Single batch data from dataset loader will be used to illustrate model structure in downstream codes.
npy.shape # single batch, shape = (n_batch, n_timestamp, n_column, n_matched_session)



### Model hyperparameters
param_hidden = 128 # number of hidden nodes
param_shrinkage = 8 # shrinkage parameter to reduce computational cost
param_dropout = 0 # dropout is not used in the current model, though this code represents where it can be located.
param_head = 1 # number of heads in transformer



### Basic functions
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Gated_linear_unit(nn.Module):
    def __init__(self, num_hidden=param_hidden, output_size=None, dropout_rate=None):
        super().__init__()
        if output_size is None:
            output_size = num_hidden
        if dropout_rate is not None:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.linear = nn.Linear(num_hidden, output_size)
        self.linear_gate = nn.Linear(num_hidden, output_size)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x_linear = self.linear(x)
        x_gate = self.linear_gate(x); x_gate = torch.sigmoid(x_gate)
        x_GLI = torch.mul(x_linear, x_gate)
        return x_GLI, x_gate

class Skip_connector(nn.Module):
    def __init__(self, num_hidden=param_hidden):
        super().__init__()
        self.layernorm = nn.LayerNorm(num_hidden)
    
    def forward(self, x):
        x1, x2 = x
        x_norm = self.layernorm(x1+x2)
        return x_norm

class Gated_residual_network(nn.Module):
    # Note that GRN can have different input & output sizes
    def __init__(self, num_hidden=param_hidden, output_size=None, dropout_rate=param_dropout, additional=False, num_additional=None, return_gate=False):
        super().__init__()
        if output_size is not None:
            self.skip_layer = nn.Linear(num_hidden, output_size)
        else:
            output_size = num_hidden
            self.skip_layer = None
        if additional is True:
            self.additional_layer = nn.Linear(num_additional, num_hidden, bias=False)
        else:
            self.additional_layer = None

        self.return_gate = return_gate
        self.feedforward_1 = nn.Linear(num_hidden, num_hidden)
        self.feedforward_2 = nn.Linear(num_hidden, num_hidden)
        self.dropout = nn.Dropout(dropout_rate)

        self.GLI = Gated_linear_unit(num_hidden=num_hidden, output_size=output_size, dropout_rate=dropout_rate)
        self.skip_connector = Skip_connector(num_hidden=output_size)

    def forward(self, x, additional_context=None):
        # Prepare skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x

        # Feedforward network
        x = self.feedforward_1(x)
        if self.additional_layer is not None:
            additional = self.additional_layer(additional_context)
            x += additional
        x = nn.ELU()(x)
        x = self.feedforward_2(x)

        # Gated linear unit (GLU)
        x, gate = self.GLI(x)
        x = self.skip_connector([x, skip])

        if self.return_gate == True:
            return x, gate
        else:
            return x



### Rnn packing related functions
'''
The current dataset consists of variable-length time-series data.
"PackedSequence" function in Pytorch will be used to pack these data.
'''
def sequence_to_pack(x, x_meta, return_pack=False):
    # x.shape = (n_batch, n_timestamp, n_hidden)
    x_pack = nn.utils.rnn.pack_padded_sequence(x, x_meta['sequence'], batch_first=True, enforce_sorted=False)
    x_pack = x_pack.to(x.device)
    if return_pack == True:
        return x_pack
    else:
        return x_pack[0]

def restore_pack(x, x_meta):
    x_packmeta = x_meta['packmeta']
    xp = nn.utils.rnn.PackedSequence(x, x_packmeta[0], x_packmeta[1], x_packmeta[2])
    return xp

def pack_to_sequence(x, x_meta, val_padding = 0):
    xp = restore_pack(x, x_meta)
    xr, _len_timestamp = nn.utils.rnn.pad_packed_sequence(xp, batch_first=True, padding_value=val_padding, total_length=n_timestamp)
    return xr

def static_to_pack(x, x_meta, num_hidden=param_hidden):
    x_sequence = x_meta['sequence']
    n_batch = x.shape[0]
    x_expanded = x.repeat(1, n_timestamp).reshape(n_batch, n_timestamp, num_hidden).float()
    x_pack = nn.utils.rnn.pack_padded_sequence(x_expanded, x_sequence, batch_first=True, enforce_sorted=False)
    x_pack = x_pack.to(x.device)
    xp = x_pack[0]
    return xp



### Getting valid sessions
def get_valid_previous_session(npy):
    npy_curr = npy[:,:,:,0]
    list_npy_prev = []
    list_valid_idx = []

    for i in range(n_prev_session):
        npy_prev = npy[:,:,:,i+1]
        idx_valid = npy_prev[:,0,idx_invariant_prev_categorical[0]] != v_absence
        npy_prev_valid = npy_prev[idx_valid,:,:]
        list_npy_prev.append(npy_prev_valid)
        list_valid_idx.append(idx_valid)
    return npy_curr, list_npy_prev, list_valid_idx

npy_curr, list_npy_prev, list_idx_valid = get_valid_previous_session(npy)

# It will divide the information of matched sessions between the current session and the previous sessions
npy_curr.shape # current session ... (n_batch, n_timestamp, n_column)
list_npy_prev[0].shape # list of previous sessions ... (n_batch, n_timestamp, n_column)
list_idx_valid[0].sum() # it is the number of valid previous sessions. (Note that some sessions did not have their previous sessions)



### Splitting variables and labels
def splitting(npy, encoding_previous=False):
        # npy.shape = (n_batch, n_timestamp, n_column, n_matched_session)
        npy = npy.float()

        # Splitting
        if encoding_previous == False:
            npy_invariant = npy[:,0,idx_invariant_curr] # We use values of the first timestamps in time-invariant variables.
        else:
            npy_invariant = npy[:,0,idx_invariant_prev]
        npy_varying = npy[:,:,idx_varying] # We use all values of timestamps in time-varying variables.
        npy_label = npy[:,:,idx_label]

        return npy_invariant, npy_varying, npy_label

def make_mask(sequence, encoding_previous=False):
    # sequence.shape = (n_batch, n_timestamp, n_hidden)
    sequence_sub = sequence[:,:,-1] # sequence_sub = (n_batch, n_timestamp)
    # padding sequence mask for variable-length time-series data
    sequence_pad_mask = (sequence_sub != v_padding).unsqueeze(1).unsqueeze(2) # sequence_pad_mask = (n_batch, 1, 1, n_timestamp)

    if encoding_previous == False:
        # Masking future sequence in the current session
        # Note that there was no future sequence masking in the previous sessions (because all data were already known).
        sequence_len = sequence.shape[1]
        sequence_sub_mask = torch.tril(torch.ones((sequence_len, sequence_len), device=sequence.device)).bool() # sequence_sub_mask = (n_timestamp, n_timestamp)
        sequence_mask = sequence_pad_mask & sequence_sub_mask # sequence_mask = (n_batch, 1, n_timestamp, n_timestamp)
        return sequence_mask
    else:
        return sequence_pad_mask

def generate_metadata(npy_varying, encoding_previous=False):
        # Sequence length 계산
        pos_masking = npy_varying[:,:,0] == v_padding # current
        x_sequence = n_timestamp-pos_masking.detach().to('cpu').numpy().sum(axis=1)-1 # 맨 마지막 record는 무조건 drop됨

        x_pack = nn.utils.rnn.pack_padded_sequence(npy_varying, x_sequence, batch_first=True, enforce_sorted=False)
        x_pack = x_pack.to(npy_varying.device)

        x_mask = make_mask(npy_varying, encoding_previous=encoding_previous)
        x_meta = {'sequence':x_sequence, 'packmeta':x_pack[1:], 'mask': x_mask}
        return x_meta

x_curr_invariant, x_curr_varying, x_curr_label = splitting(npy_curr, encoding_previous=False)
x_curr_meta = generate_metadata(x_curr_varying, encoding_previous=False)

# x_curr_invariant.shape # invariant features from the current session ... (n_batch, n_invariant)
# x_curr_varying.shape # varying features from the current sesssion ... (n_batch, n_timestamp, n_varying)
# x_curr_label.shape # labels ... (n_batch, n_timestamp, n_label)
# x_curr_meta['sequence'] # metadata of time-sequences in the current session
# x_curr_meta['mask'] # masking of time-sequences in the current session (future masking)

x_prev_invariant, x_prev_varying, x_prev_label = splitting(list_npy_prev[0], encoding_previous=True)
x_prev_meta = generate_metadata(x_prev_varying, encoding_previous=True)

# x_prev_invariant.shape # invariant features from the first previous session ... (n_batch, n_invariant)
# x_prev_varying.shape # varying features from the first previous session ... (n_batch, n_timestamp, n_varying)
# x_prev_label.shape # labels (it will not be used in downstream codes) ... (n_batch, n_timestamp, n_label)
# x_prev_meta['sequence'] # metadata of time-sequenes in the first previous session
# x_prev_meta['mask'] # masking of time-sequences in the current session (no future masking)



### Processing pipeline
class Embedding_invariant(nn.Module):
    def __init__(self, num_hidden=param_hidden, ratio_shrinkage=param_shrinkage, encoding_previous=False):
        super().__init__()
        self.encoding_previous = encoding_previous
        if self.encoding_previous == True:
            self.n_invariant = n_invariant_prev
            self.idx_invariant_categorical = idx_invariant_prev_categorical
            self.max_invariant_categorical = max_invariant_prev_categorical
        else:
            self.n_invariant = n_invariant_curr
            self.idx_invariant_categorical = idx_invariant_curr_categorical
            self.max_invariant_categorical = max_invariant_curr_categorical
        invariant_layers = []
        for n in range(self.n_invariant):
            # Categorical variable -> Embedding layer
            if n in self.idx_invariant_categorical:
                idx = self.idx_invariant_categorical.index(n)
                max_cat = self.max_invariant_categorical[idx]
                l_layer = nn.Embedding(max_cat, int(num_hidden/ratio_shrinkage)) # Parameter shrinkage
                '''
                Note that GRN can have different sizes of input & output.
                Encoding process by GRN has the highest number of parameters in the current model pipeline.
                To attenuate computational cost from encoding process, the number of hidden nodes is decreased to a fourth (=ratio_shrinkage) of its former size. 
                '''
            # Continuous variable -> Dense layer
            else:
                l_layer = nn.Linear(1, int(num_hidden/ratio_shrinkage))
            invariant_layers.append(l_layer)
        self.invariant_layers = nn.ModuleList(invariant_layers)

    def forward(self, x):
        # x.shape = (n_batch, n_invariant)
        x = x.float()
        for n in range(self.n_invariant):
            x_temp = torch.unsqueeze(x[:,n], dim=-1)
            if n in self.idx_invariant_categorical:
                x_temp = x_temp.to(torch.int64)
            x_temp_emb = self.invariant_layers[n](x_temp)
            if len(x_temp_emb.shape)==2: # output shape from dense layer => (n_batch, 1, num_hidden)
                x_temp_emb = torch.unsqueeze(x_temp_emb, dim=1)
            if n == 0: # output shape from embedding layer=> (n_batch, num_hidden)
                x_emb = x_temp_emb
            else:
                x_emb = torch.cat([x_emb, x_temp_emb], dim=1)
        # x_emb.shape = (n_batch, n_invariant, num_hidden)
        return x_emb

class Encoder_invariant(nn.Module):
    def __init__(self, num_hidden=param_hidden, dropout_rate=param_dropout, ratio_shrinkage=param_shrinkage, encoding_previous=False):
        super().__init__()
        self.encoding_previous=encoding_previous

        if encoding_previous==True:
            num_feature = n_invariant_prev
        else:
            num_feature = n_invariant_curr+n_prev_session

        self.flatten_GRN = Gated_residual_network(int(num_hidden/ratio_shrinkage)*num_feature, num_feature, dropout_rate=dropout_rate)
        self.invariant_GRNs = nn.ModuleList([Gated_residual_network(num_hidden=int(num_hidden/ratio_shrinkage), output_size=num_hidden, dropout_rate=dropout_rate) for _ in range(num_feature)])

    def forward(self, x, x_prev=None, list_prev_idx=None):
        if self.encoding_previous==False:
            x = torch.cat([x, x_prev], dim=1)

        # Attention weight
        flatten = torch.flatten(x, start_dim=1)
        flatten = self.flatten_GRN(flatten)
        if self.encoding_previous==False:
            for i, prev_idx in enumerate(list(reversed(list_prev_idx))):
                s = i+1; flatten[~prev_idx,-s] = -1e10 # Remove weights if there is no previous sessions.
        weight = torch.softmax(flatten, dim=1)
        weight_exp = torch.unsqueeze(weight, dim=-1) # weight.shape = (n_batch, n_invariant, 1)

        # GRN embedding
        for i, layer in enumerate(self.invariant_GRNs):
            x_temp = layer(x[:,i:i+1,:]) # x_temp.shape = (n_batch, 1, num_hidden)
            if i == 0:
                x_emb = x_temp
            else:
                x_emb = torch.cat([x_emb, x_temp], dim=1)

        x_comb = torch.mul(x_emb, weight_exp) # x_comb.shape = (n_batch, n_invariant, n_hidden)
        x_vec = torch.sum(x_comb, dim=1) # x_vec.shape = (n_batch, n_hidden)

        return x_vec, weight

class Embedding_varying(nn.Module):
    def __init__(self, num_hidden=param_hidden, ratio_shrinkage=param_shrinkage):
        super().__init__()
        varying_layers = []
        for n in range(n_varying):
            if n in idx_varying_categorical:
                idx = idx_varying_categorical.index(n)
                max_cat = max_varying_categorical[idx]
                l_layer = nn.Embedding(max_cat, int(num_hidden/ratio_shrinkage)) # Parameter shrinkage
            else:
                l_layer = nn.Linear(1, int(num_hidden/ratio_shrinkage))
            varying_layers.append(l_layer)
        self.varying_layers = nn.ModuleList(varying_layers)

    def forward(self, x, x_meta):
        # x.shape = (n_batch, n_timestamp, n_varying)
        x = x.float()

        # packing
        xp = sequence_to_pack(x, x_meta)
        # xp.shape = (n_pack, n_varying)

        for n in range(n_varying):
            x_temp = xp[:,n:n+1]
            # x_temp.shape = (n_pack, 1)
            if n in idx_varying_categorical:
                x_temp = x_temp.to(torch.int64)
            x_temp_emb = self.varying_layers[n](x_temp)
            if len(x_temp_emb.shape)==2:
                x_temp_emb = torch.unsqueeze(x_temp_emb, dim=1)
            if n == 0:
                x_emb = x_temp_emb
            else:
                x_emb = torch.cat([x_emb, x_temp_emb], dim=1)
        # x_emb.shape = (n_pack, n_varying, num_hidden)
        return x_emb

class Encoder_varying(nn.Module):
    def __init__(self, num_hidden=param_hidden, dropout_rate=param_dropout, num_varying=n_varying, ratio_shrinkage=param_shrinkage):
        super().__init__()

        self.num_hidden = num_hidden
        # self.num_varying = num_varying

        self.flatten_GRN = Gated_residual_network(int(num_hidden/ratio_shrinkage)*num_varying, num_varying, dropout_rate=dropout_rate)
        self.varying_GRNs = nn.ModuleList([Gated_residual_network(int(num_hidden/ratio_shrinkage), output_size=num_hidden, dropout_rate=dropout_rate) for _ in range(num_varying)])

    def forward(self, x, x_meta, invariant_vs=None):
        # x.shape = (n_pack, n_varying, n_hidden)
        # invariant_vs.shape = (n_batch, n_hidden)
        
        expanded_invariant_vs_pack = static_to_pack(invariant_vs, x_meta, self.num_hidden)
        # expanded_invariant_vs_pack.shape = (n_pack, n_hidden)

        # attention weight
        flatten = torch.flatten(x, start_dim=1) # flatten.shape = (n_pack, n_hidden * n_varying)

        flatten = self.flatten_GRN(flatten, expanded_invariant_vs_pack) # flatten.shape = (n_pack, n_varying)
        weight = torch.softmax(flatten, dim=1)
        weight_exp = torch.unsqueeze(weight, dim=-1) # weight.shape = (n_pack, n_varying, 1)

        # GRN embedding
        for i, layer in enumerate(self.varying_GRNs):
            x_temp = layer(x[:,i:i+1,:]) # x_temp.shape = (n_pack, 1, n_hidden)
            if i == 0:
                x_emb = x_temp
            else:
                x_emb = torch.cat([x_emb, x_temp], dim=1)

        # x_emb.shape = (n_pack, n_varying, n_hidden)
        x_comb = torch.mul(x_emb, weight_exp) # x_comb.shape = (n_pack n_varying, n_hidden)
        x_vec = torch.sum(x_comb, dim=1) # x_vec.shape = (n_pack n_hidden)

        return x_vec, weight

class RNN_GRU(nn.Module):
    def __init__(self, num_hidden=param_hidden, dropout_rate=param_dropout):
        super().__init__()

        self.GRU_varying = nn.GRU(num_hidden, num_hidden, batch_first=True)
        self.GLI = Gated_linear_unit(num_hidden=num_hidden, output_size=num_hidden, dropout_rate=dropout_rate)
        self.skip_connector = Skip_connector(num_hidden=num_hidden)

    def forward(self, x, x_meta, state_h=None):
        # x.shape = (n_pack, n_hidden)
        # state_h.shape = (n_batch, n_hidden)
        skip = x
        state_h = torch.unsqueeze(state_h, dim=0) #  cell state.shape => (1, batch, n_hidden)

        xp = restore_pack(x, x_meta) # PackedSequence

        gru, _ = self.GRU_varying(xp, state_h)
        gru = gru[0] # gru.shape = (n_pack, n_hidden)

        # Gated linear unit (GLI)
        gru, _ = self.GLI(gru)
        gru = self.skip_connector([gru, skip])
        return gru

class Pre_enrichment(nn.Module):
    def __init__(self, num_hidden=param_hidden, dropout_rate=param_dropout):
        super().__init__()

        self.num_hidden = num_hidden
        self.enrich_GRN = Gated_residual_network(num_hidden=num_hidden, dropout_rate=dropout_rate, additional=True, num_additional=num_hidden)

    def forward(self, x, x_meta, invariant_enrich):
        # x.shape = (n_pack, n_hidden)
        # invariant_enrich.shape = (n_batch, n_hidden)
        expanded_invariant_enrich = static_to_pack(invariant_enrich, x_meta, num_hidden=self.num_hidden)
        # expanded_invariant_enrich.shape = (n_pack, n_hidden)
        x_enrich = self.enrich_GRN(x, expanded_invariant_enrich)
        return x_enrich

class Transformer_block(nn.Module):
    def __init__(self, return_single=False, num_head=param_head, num_hidden=param_hidden, dropout_rate=param_dropout):
        super().__init__()

        self.hid_dim = num_hidden
        self.n_heads = num_head
        self.head_dim = num_hidden // num_head

        self.fc_q = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.fc_k = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.fc_v = nn.Linear(self.hid_dim, self.hid_dim, bias=False)

        self.return_single = return_single
        if self.return_single == False:
            self.fc_o = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        else:
            self.fc_o = nn.Linear(n_timestamp, 1, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, x_meta):
        mask = x_meta['mask']
        len_sequence = x_meta['sequence']
        query, key, value = x, x, x
        # query, key, value shape = (n_batch, n_timestamp, n_hidden)
        n_batch = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(n_batch, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(n_batch, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(n_batch, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q, K, V shape = (n_batch, n_heads, src_len, head_dim)

        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(x.device)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        energy = energy.masked_fill(mask == 0, -1e10) # Sequence masking
        # energy.shape = (n_batch, n_heads, src_len, src_len)
        
        attention = torch.softmax(energy, dim = -1)
        scale_len = torch.sqrt(torch.FloatTensor(len_sequence)).to(x.device)
        # Length of sequences in hemodialysis sessions were relatively short (ex. 2-4 lengths),
        # and these effects were attenuated by multiplying a square root of each sequence length.
        scale_len = scale_len.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # attention.shape = (n_batch, n_heads, src_len, src_len)

        out = torch.matmul(self.dropout(attention * scale_len), V)
        # out.shape = (n_batch, n_heads, src_len, head_dim)
        
        out = out.permute(0,2,1,3).reshape(n_batch, -1, self.hid_dim)
        # out.shape = (n_batch, src_len, hid_dim)

        out = self.fc_o(out)
        # out.shape = (n_batch, src_len, hid_dim)

        return out, attention

class Transformer_enrichment(nn.Module):
    def __init__(self, num_hidden=param_hidden, dropout_rate=param_dropout, num_head=param_head):
        super().__init__()

        self.transformer = Transformer_block(num_head=num_head, num_hidden=num_hidden, dropout_rate=dropout_rate)
        self.GLI_trans = Gated_linear_unit(num_hidden=num_hidden, output_size=num_hidden, dropout_rate=dropout_rate)
        self.skip_connector_trans = Skip_connector(num_hidden=num_hidden)
        self.final_GRN = Gated_residual_network(num_hidden=num_hidden, dropout_rate=dropout_rate)
        self.GLI_final = Gated_linear_unit(num_hidden=num_hidden, output_size=num_hidden, dropout_rate=dropout_rate)
        self.skip_connector_final = Skip_connector(num_hidden=num_hidden)

    def forward(self, x, x_meta, skip=None):
        # x.shape = (n_pack, n_hidden)
        x_pre = x

        x_r = pack_to_sequence(x, x_meta)
        x_r, transformer_att = self.transformer(x_r, x_meta)
        x = sequence_to_pack(x_r, x_meta)

        # Gated linear unit (GLI), transformer
        x, _ = self.GLI_trans(x)
        x = self.skip_connector_trans([x, x_pre])
        x = self.final_GRN(x)

        # Gated linear unit (GLI), final
        x, _ = self.GLI_final(x)
        x = self.skip_connector_final([x, skip])

        return x, transformer_att

class Processing_pipeline(nn.Module):
    def __init__(self, num_hidden=param_hidden, dropout_rate=param_dropout, num_head=param_head, ratio_shrinkage=param_shrinkage, encoding_previous=False):
        super().__init__()
        skip_dropout = 0
        self.encoding_previous = encoding_previous
        self.embedding_invariant = Embedding_invariant(num_hidden=num_hidden, ratio_shrinkage=ratio_shrinkage, encoding_previous=encoding_previous)
        self.embedding_varying = Embedding_varying(num_hidden=num_hidden, ratio_shrinkage=ratio_shrinkage)

        self.encoding_invariant = Encoder_invariant(num_hidden=num_hidden, dropout_rate=dropout_rate, ratio_shrinkage=ratio_shrinkage, encoding_previous=encoding_previous)
        self.selection_invariant = Gated_residual_network(num_hidden=num_hidden, dropout_rate=dropout_rate)
        self.enrich_invariant = Gated_residual_network(num_hidden=num_hidden, dropout_rate=dropout_rate)
        self.hidden_state_invariant = Gated_residual_network(num_hidden=num_hidden, dropout_rate=dropout_rate)

        self.encoding_varying = Encoder_varying(num_hidden=num_hidden, dropout_rate=dropout_rate, ratio_shrinkage=ratio_shrinkage)
        self.rnn_gru = RNN_GRU(num_hidden=num_hidden, dropout_rate=skip_dropout)
        self.pre_enrichment = Pre_enrichment(num_hidden=num_hidden, dropout_rate=skip_dropout)
        self.transformer_enrichment = Transformer_enrichment(num_hidden=num_hidden, dropout_rate=skip_dropout, num_head=num_head)

        if self.encoding_previous == True:
            self.modifier = Modifier_prev(num_hidden=num_hidden, ratio_shrinkage=ratio_shrinkage)
        else:
            self.modifier = Modifier_classification(num_hidden=num_hidden)

    def forward(self, x_invariant, x_varying, x_meta, x_prev=None, list_prev_idx=None):
        # Dense or embedding
        x_invariant = self.embedding_invariant(x_invariant)
        x_varying = self.embedding_varying(x_varying, x_meta)

        # Encoding invariant
        x_invariant, x_invariant_weight = self.encoding_invariant(x_invariant, x_prev=x_prev, list_prev_idx=list_prev_idx)
        x_selection = self.selection_invariant(x_invariant)

        # Encoding varying
        # ... The information of previous sessions will be transformed and entered into current session as 'time-invariant' features.
        x_encoded, x_varying_weight = self.encoding_varying(x_varying, x_meta, x_selection)

        # RNN (GRU)
        x_enrich = self.enrich_invariant(x_invariant)
        x_hidden = self.hidden_state_invariant(x_invariant)
        x_rnn = self.rnn_gru(x_encoded, x_meta, x_hidden)
        x_out = self.pre_enrichment(x_rnn, x_meta, x_enrich)

        # Transformer
        x_out, x_transformer_weight = self.transformer_enrichment(x_out, x_meta, x_rnn)

        # Weights
        x_varying_weight = pack_to_sequence(x_varying_weight, x_meta)
        x_invariant_weight = x_invariant_weight.cpu().detach().numpy().astype('float16')
        x_varying_weight = x_varying_weight.cpu().detach().numpy().astype('float16')
        x_transformer_weight = x_transformer_weight.cpu().detach().numpy().astype('float16')
        x_weights = [x_invariant_weight, x_varying_weight, x_transformer_weight]

        # Classification
        x_fin = self.modifier(x_out, x_meta)

        return x_fin, x_weights

processing_prev_variables = Processing_pipeline(encoding_previous=True)
n_batch = npy_curr.shape[0]
matched_previous = torch.zeros(n_batch, n_prev_session, int(param_hidden/param_shrinkage))

for i, (npy_prev, idx_valid) in enumerate(zip(list_npy_prev, list_idx_valid)):
    # The information of five previous sessions will pass "Processing pipeline".
    if sum(idx_valid).cpu() == 0:
        continue
    x_prev_invariant, x_prev_varying, x_prev_label = splitting(npy_prev, encoding_previous=True)
    x_prev_meta = generate_metadata(x_prev_varying, encoding_previous=True)
    x_prev_out, x_prev_weights = processing_prev_variables(x_prev_invariant, x_prev_varying, x_prev_meta)
    matched_previous[idx_valid,i,:] = x_prev_out

processing_curr_variables = Processing_pipeline(encoding_previous=False)
x_curr_out, x_curr_weights = processing_curr_variables(x_curr_invariant, x_curr_varying, x_curr_meta, x_prev=matched_previous, list_prev_idx=list_idx_valid)

# x_curr_out.shape # 2848 packed sequences, three labels
# x_curr_weights[0].shape # Weights of time-invariant features
# x_curr_weights[1].shape # Weights of time-varying features
# x_curr_weights[2].shape # Weights of sequences in Transformer


### Final model (modified TFT)
class Modifier_prev(nn.Module):
    def __init__(self, num_hidden=param_hidden, ratio_shrinkage=param_shrinkage):
        super().__init__()
        self.linear_1 = nn.Linear(n_timestamp, 1, bias=False)
        self.linear_2 = nn.Linear(num_hidden, int(num_hidden/ratio_shrinkage), bias=False)
    
    def forward(self, x, x_meta):
        # x.shape = (n_pack, n_hidden)
        x = pack_to_sequence(x, x_meta) # missing values were 0 padded.
        # x.shape = (n_batch, n_timestamp, n_hidden)
        x = x.permute(0, 2, 1)
        # x.shape = (n_batch, n_hidden, n_timestamp)
        x = self.linear_1(x)
        # x.shape = (n_batch, n_hidden, 1)
        x = x[:, :, 0]
        # x.shape = (n_batch, n_hidden)
        x = self.linear_2(x)
        # x.shape = (n_batch, n_hidden_shrinkage)
        return x

class Modifier_classification(nn.Module):
    def __init__(self, num_hidden=param_hidden):
        super().__init__()
        self.classifier = nn.Linear(num_hidden, n_label)
    
    def forward(self, x, x_meta=None):
        x = self.classifier(x)
        return x

class Modified_TFT(nn.Module):
    def __init__(self, num_hidden=param_hidden, dropout_rate=param_dropout, num_head=param_head, ratio_shrinkage=param_shrinkage):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_head = num_head
        self.ratio_shrinkage = ratio_shrinkage
        self.processing_prev_variables = Processing_pipeline(num_hidden=num_hidden, dropout_rate=dropout_rate, num_head=num_head, ratio_shrinkage=ratio_shrinkage, encoding_previous=True)
        self.processing_curr_variables = Processing_pipeline(num_hidden=num_hidden, dropout_rate=dropout_rate, num_head=num_head, ratio_shrinkage=ratio_shrinkage, encoding_previous=False)

    def forward(self, npy):
        npy_curr, list_npy_prev, list_idx_valid = get_valid_previous_session(npy)
        # npy_curr.shape # (n_batch, n_timestamp, n_variables)

        x_curr_invariant, x_curr_varying, x_curr_label_sequence = splitting(npy_curr, encoding_previous=False)
        x_curr_meta = generate_metadata(x_curr_varying, encoding_previous=False)
        # x_curr_invariant.shape # (n_batch, n_invariant)
        # x_curr_varying.shape # (n_batch, n_timestamp, n_varying)
        # x_curr_label.shape # (n_batch, n_timestamp, 1)

        n_batch = npy_curr.shape[0]
        matched_previous = torch.zeros(n_batch, n_prev_session, int(self.num_hidden/self.ratio_shrinkage)).to(npy_curr.device)
        prev_weight_invariant = np.full([n_batch, n_prev_session, n_invariant_prev], np.nan).astype('float16')
        prev_weight_varying = np.full([n_batch, n_prev_session, n_timestamp, n_varying], np.nan).astype('float16')
        prev_weight_transformer = np.full([n_batch, n_prev_session, self.num_head, n_timestamp, n_timestamp], np.nan).astype('float16')

        for i, (npy_prev, idx_valid) in enumerate(zip(list_npy_prev, list_idx_valid)):
            if sum(idx_valid).cpu() == 0:
                continue
            x_prev_invariant, x_prev_varying, x_prev_label = splitting(npy_prev, encoding_previous=True)
            x_prev_meta = generate_metadata(x_prev_varying, encoding_previous=True)
            x_prev_out, x_prev_weights = self.processing_prev_variables(x_prev_invariant, x_prev_varying, x_prev_meta)

            matched_previous[idx_valid,i,:] = x_prev_out
            prev_weight_invariant[idx_valid.cpu().detach().numpy(),i,:] = x_prev_weights[0]
            prev_weight_varying[idx_valid.cpu().detach().numpy(),i,:,:] = x_prev_weights[1]
            prev_weight_transformer[idx_valid.cpu().detach().numpy(),i,:,:,:] = x_prev_weights[2]

        x_curr_out_pack, x_curr_weights = self.processing_curr_variables(x_curr_invariant, x_curr_varying, x_curr_meta, x_prev=matched_previous, list_prev_idx=list_idx_valid)
        x_curr_out_sequence = pack_to_sequence(x_curr_out_pack, x_curr_meta)
        x_curr_label_pack = sequence_to_pack(x_curr_label_sequence, x_curr_meta)
        
        pred = x_curr_out_pack
        label = x_curr_label_pack
        x_prev_weights_merged = [prev_weight_invariant, prev_weight_varying, prev_weight_transformer]

        return pred, label, [x_curr_out_sequence, x_curr_label_sequence, [x_curr_weights, x_prev_weights_merged]]

mTFT = Modified_TFT(num_hidden=128, dropout_rate=0, num_head=1, ratio_shrinkage=8)
pred, label, [pred_sequence, label_sequence, [weights_curr, weights_prev]] = mTFT(npy)

count_parameters(mTFT) # 6,094,404 parameters
pred.shape # packed_sequences, prediction
label.shape # packed_sequences, label
pred_sequence.shape # (n_batch, n_timestamp, n_label)
label_sequence.shape # (n_batch, n_timestamp, n_label)
weights_curr[0].shape # (n_batch, n_invariant + 5 previous sessions)
weights_curr[1].shape # (n_batch, n_timestamp, n_varying)
weights_curr[2].shape # (n_batch, n_head, n_timestamp, n_timestamp)
