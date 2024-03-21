"""
Created on Thu Mar 21 07:50:04 2024

@author: XinzeLee
@github: https://github.com/XinzeLee

@reference:
    1: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network
        Authors: Xinze Li, Fanfan Lin (corresponding and co-first author), Huai Wang, Xin Zhang, Hao Ma, Changyun Wen and Frede Blaabjerg
        Paper DOI: 10.1109/TIE.2024.3352119
    2: Data-Light Physics-Informed Modeling for the Modulation Optimization of a Dual-Active-Bridge Converter
        Authors: Xinze Li, Fanfan Lin (corresponding and co-first author), Xin Zhang, Hao Ma and Frede Blaabjerg
        Paper DOI: 10.1109/TPEL.2024.3378184

"""

from pinn_utils import *


class CustomDataset(Dataset):
    def __init__(self, states, inputs, targets):
        super(CustomDataset, self).__init__()
        self.states = states # states are training x
        self.inputs = inputs
        self.targets = targets
        
    def __getitem__(self, index):
        return self.states[index], self.inputs[index], self.targets[index]
        
    def __len__(self):
        return len(self.states)
    
    
def train(model_implicit_PINN, data_loader, Vin, epoch=200):
    param_list = ['cell.Lr', 'cell.RL']
    params = list(filter(lambda kv: kv[0] in param_list, model_implicit_PINN.named_parameters()))
    base_params = list(filter(lambda kv: kv[0] not in param_list, model_implicit_PINN.named_parameters()))
    optimizer_implicit_PINN = torch.optim.Adam([{"params": [param[1] for param in params], "lr":1e-5},
                                    {"params": [base_param[1] for base_param in base_params]}], lr=1e-2)
    
    
    clamp1 = WeightClamp(['cell.Lr', 'cell.RL', 'cell.n'], 
                         [(100e-6, 500e-6), (10e-6, 2000e-6), (1.5, 2.5)]) # clamp the coefficient F
    loss_pinn = nn.MSELoss()
    device = "cpu" # it is a waste to use gpu for this network
    
    MIN_implicit_PINN = np.inf
    best_implicit_PINN = None
    
    model_implicit_PINN.train()
    model_implicit_PINN = model_implicit_PINN.to(device)
    for epoch in range(200):
        #Forward pass
        total_loss = 0.
        for data in data_loader:
            """ Logic is:
                input_ (full length) -> smooth_all -> PINN pred -> segment final Tslen*2 points -> sync
            """
            state, input_, target = data
            state, input_, target = state.to(device), input_.to(device), target.to(device)
            state0 = state[:, :1] # should be zero to avoid learning the initial state
    #         state0 = torch.zeros(state.shape).to(device) # should be zero to avoid learning the initial state
            pred = model_implicit_PINN.forward(input_, state0)
    #         pred = pred-(pred[:, -2*Tslen:].max(dim=1)[0]+pred[:, -2*Tslen:].min(dim=1)[0])[..., None]/2
            pred, _ = transform(input_[:, -2*Tslen:], pred[:, -2*Tslen:], Vin)
            
            loss_train = loss_pinn(pred, target)
            optimizer_implicit_PINN.zero_grad()
            loss_train.backward()
            optimizer_implicit_PINN.step()
            clamp1(model_implicit_PINN) # comment out this line if using pure data-driven model for dk
            total_loss += loss_train.item()
        print(list(map(lambda x: round(x.item(), 7), model_implicit_PINN.parameters())))
        print(f"Epoch {epoch}, Training loss {total_loss/len(data_loader)}")  
        if epoch % 1 == 0:
            *_, test_loss = evaluate(test_inputs[:, 1:], test_states, model_implicit_PINN)
            if test_loss < MIN_implicit_PINN:
                MIN_implicit_PINN, best_implicit_PINN = test_loss, copy.deepcopy(model_implicit_PINN)
                print(f"New loss is {MIN_implicit_PINN}.")
                print('-'*81)
    return best_implicit_PINN