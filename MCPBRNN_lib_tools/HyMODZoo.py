import torch
import torch.nn as nn
from torch import Tensor

class MCPBRNN_Generic_PETconstraint_constantoutput_variableLoss_MCA2(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_constantoutput_variableLoss_MCA2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))    
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        self.relu_l = nn.ELU()
        self.relu = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))     
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])  
        gw_n = torch.zeros([batch_size, hidden_size])                
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_oogw = torch.zeros([batch_size, hidden_size])        
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_olc = torch.zeros([batch_size, hidden_size])        
        Gate_f = torch.zeros([batch_size, hidden_size])

        # MLE sigma
        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        #bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = 0#236.6076
        ml = 2.9086
        so = 1#63.5897
        sl = 1.8980        
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm) + torch.exp(self.weight_r_yom_gw))
                oo2 = 0 #torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 #* torch.sigmoid(oo2)

                oogw1 = torch.exp(self.weight_r_yom_gw)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm) + torch.exp(self.weight_r_yom_gw))
                oogw2 = 0 #torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oogw = oogw1 #* torch.sigmoid(oo2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm) + torch.exp(self.weight_r_yom_gw))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - ol_constraint - oogw)

                ol_constraint = self.relu(ol_constraint)
                f = self.relu(f)

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                lc_1 = ol_constraint * c_1                 
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 
                lc_0 = ol_constraint * c_0 
                gw_0 = oogw * c_0 

            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0 
                gw_n[b,:] = gw_0                                
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_oogw[b,:] = oogw                
                Gate_ol[b,:] = ol
                Gate_olc[b,:] = ol_constraint                
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)
        return h_n, c_n, l_n, lc_n, gw_n, bp_n, Gate_ib, Gate_oo, Gate_oogw, Gate_ol, Gate_olc, Gate_f, h_nout, obs_std


class MCPBRNN_Generic_constantoutput_variableLoss_MCA2(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_constantoutput_variableLoss_MCA2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))    
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        #self.relu_l = nn.ELU()
        #self.relu = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))     
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        #lc_n = torch.zeros([batch_size, hidden_size])  
        gw_n = torch.zeros([batch_size, hidden_size])                
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_oogw = torch.zeros([batch_size, hidden_size])        
        Gate_ol = torch.zeros([batch_size, hidden_size])
        #Gate_olc = torch.zeros([batch_size, hidden_size])        
        Gate_f = torch.zeros([batch_size, hidden_size])

        # MLE sigma
        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        #bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)
        mo = 0#236.6076
        ml = 2.9086
        so = 1#63.5897
        sl = 1.8980        
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm) + torch.exp(self.weight_r_yom_gw))
                oo2 = 0 #torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 #* torch.sigmoid(oo2)

                oogw1 = torch.exp(self.weight_r_yom_gw)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm) + torch.exp(self.weight_r_yom_gw))
                oogw2 = 0 #torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oogw = oogw1 #* torch.sigmoid(oo2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm) + torch.exp(self.weight_r_yom_gw))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                #if c_0 > 0:
                #    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                #else:
                #    ol_constraint = ol

                f = (1.0 -  oo - ol - oogw)

                ##ol_constraint = self.relu(ol_constraint)
                ##f = self.relu(f)

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                l_1 = ol * c_1 
                bp_0 = ib * g
                h_0 = oo * c_0
                l_0 = ol * c_0 
                gw_0 = oogw * c_0 

            # save state     
                q_n[b,:] = h_0 
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 
                l_n[b,:] = l_0
                gw_n[b,:] = gw_0                                
                bp_n[b,:] = bp_0                        

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_oogw[b,:] = oogw                
                Gate_ol[b,:] = ol
                #Gate_olc[b,:] = ol_constraint                
                Gate_f[b,:] = f 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd

        h_nout = torch.cat((h_n, obs_std), 1)
        return h_n, c_n, l_n, gw_n, bp_n, Gate_ib, Gate_oo, Gate_oogw, Gate_ol, Gate_f, h_nout, obs_std

class MCPBRNN_SW_Constant_Routing(nn.Module):
 
    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_SW_Constant_Routing, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))        
    
    def forward(self, x, epoch, time_lag, y_obs):

        if self.batch_first:
            x = x.transpose(0, 1)

        x = x[:,:, None]
        seq_len, batch_size, _ = x.size()
        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
      
        # Gate Function
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)    

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)
                oo = torch.sigmoid(self.weight_r_yom)
                f = (1.0 - oo) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                h_0 = oo * c_0                
                bp_0 = ib * g

            # save state     
                h_n[b,:] = h_0 
                c_n[b,:] = c_0 # modify from c_1 to c_0

            # save gate     
                Gate_oo[b,:] = oo
                Gate_f[b,:] = f 
                h_x = (h_1, c_1)
                
        return h_n, c_n, Gate_oo, Gate_f

class MCPBRNN_GWConstant_Routing(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_GWConstant_Routing, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
 
    def forward(self, x, epoch, time_lag, y_obs):


        if self.batch_first:
            x = x.transpose(0, 1)

        x = x[:,:, None]
        seq_len, batch_size, _ = x.size()
        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = y_obs[0,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)/torch.sigmoid(self.weight_r_yom)

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
      
        # Gate Function
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)   

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo = torch.sigmoid(self.weight_r_yom)
                f = (1.0 - oo) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                h_0 = oo * c_0                
                bp_0 = ib * g

            # save state     
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0

            # save gate     
                Gate_oo[b,:] = oo
                Gate_f[b,:] = f 
                h_x = (h_1, c_1)
                
        return h_n, c_n, Gate_oo, Gate_f

class MCPBRNN_GWVariant_Routing(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_GWVariant_Routing, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, Ini_C):

        if self.batch_first:
            x = x.transpose(0, 1)

        x = x[:,:, None]
        seq_len, batch_size, _ = x.size()
        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = Ini_C * torch.ones([hidden_size, hidden_size]) 

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
      
        # Gate Function
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))

        mo = c_mean
        so = c_std

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)   

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)            
                f = (1.0 - oo) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                h_0 = oo * c_0                
                bp_0 = ib * g

            # save state     
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0

            # save gate     
                Gate_oo[b,:] = oo
                Gate_f[b,:] = f 
                h_x = (h_1, c_1)
                
        return h_n, c_n, Gate_oo, Gate_f

class MCPBRNN_GWVariant_Routing_MRRegular(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_GWVariant_Routing_MRRegular, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yvm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))                 
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.relu_v = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yvm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))                 
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.rand(self.hidden_size))

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, Ini_C):

        if self.batch_first:
            x = x.transpose(0, 1)

        x = x[:,:, None]
        seq_len, batch_size, _ = x.size()
        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = Ini_C * torch.ones([hidden_size, hidden_size]) 

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        MR_n = torch.zeros([batch_size, hidden_size])      
        # Gate Function
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        Gate_ov = torch.zeros([batch_size, hidden_size])        
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_yrm = (self.bias_b0_yrm.unsqueeze(0).expand(1, *self.bias_b0_yrm.size()))

        mo = c_mean
        so = c_std
        scale_mr = 500
        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)   

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)            
                f = (1.0 - oo) 

                ov0 = torch.mm(c_0/scale_mr - torch.exp(bias_b0_yrm), torch.exp(self.weight_s_yvm))
                ov1 = torch.sigmoid(self.weight_r_yvm) * torch.tanh(ov0)

                ov = ov1 - self.relu_v(ov1 - f)

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g - ov * torch.abs(c_0 - torch.exp(bias_b0_yrm) * scale_mr)                    
                h_1 = oo * c_1
                h_0 = oo * c_0                
                bp_0 = ib * g

            # save state     
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0
                MR_n[b,:] = - ov * torch.abs(c_0 - torch.exp(bias_b0_yrm) * scale_mr)   
            # save gate     
                Gate_oo[b,:] = oo
                Gate_ov[b,:] = ov                
                Gate_f[b,:] = f 
                h_x = (h_1, c_1)
                
        return h_n, c_n, Gate_oo, Gate_f, Gate_ov, MR_n

class MCPBRNN_GWVariant_Routing_MRRegular_Relaxed(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_GWVariant_Routing_MRRegular_Relaxed, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size       
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yvm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))                 
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.relu_v = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yvm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))                 
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.weight_s_yvm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_yrm = nn.Parameter(torch.rand(self.hidden_size))

    def forward(self, x, epoch, time_lag, y_obs, c_mean, c_std, Ini_C):

        if self.batch_first:
            x = x.transpose(0, 1)

        x = x[:,:, None]
        seq_len, batch_size, _ = x.size()
        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = Ini_C * torch.ones([hidden_size, hidden_size]) 

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
        MR_n = torch.zeros([batch_size, hidden_size])      
        # Gate Function
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        Gate_ov = torch.zeros([batch_size, hidden_size])        
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_yrm = (self.bias_b0_yrm.unsqueeze(0).expand(1, *self.bias_b0_yrm.size()))

        mo = c_mean
        so = c_std
        scale_mr = 500
        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)   

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)            
                f = (1.0 - oo) 

                ov0 = torch.mm(c_0/scale_mr - bias_b0_yrm, torch.exp(self.weight_s_yvm))
                ov1 = torch.sigmoid(self.weight_r_yvm) * torch.tanh(ov0)

                ov = ov1 - self.relu_v(ov1 - f)

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g - ov * torch.abs(c_0 - bias_b0_yrm * scale_mr)                    
                h_1 = oo * c_1
                h_0 = oo * c_0                
                bp_0 = ib * g

            # save state     
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0
                MR_n[b,:] = - ov * torch.abs(c_0 - bias_b0_yrm * scale_mr)  
            # save gate     
                Gate_oo[b,:] = oo
                Gate_ov[b,:] = ov                
                Gate_f[b,:] = f 
                h_x = (h_1, c_1)
                
        return h_n, c_n, Gate_oo, Gate_f, Gate_ov, MR_n

class MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))           
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        self.relu_l = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_gw = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])       
        h_n = torch.zeros([batch_size, hidden_size])
        gw_n = torch.zeros([batch_size, hidden_size])        
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_oogw = torch.zeros([batch_size, hidden_size])        
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        # MLE sigma
        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_yom_gw = (self.bias_b0_yom_gw.unsqueeze(0).expand(1, *self.bias_b0_yom_gw.size()))        
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))

        #torch.set_printoptions(precision=20)

        mo = p_mean
        ml = 2.9086
        so = p_std
        sl = 1.8980

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                oogw1 = torch.exp(self.weight_r_yom_gw)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oogw2 = torch.addmm(bias_b0_yom_gw, (c_0-mo)/so, self.weight_b1_yom_gw)
                oogw = oogw1 * torch.sigmoid(oogw2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 #u2 * torch.mm(c_0, self.weight_b3_ylm)
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:                
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - oogw - ol_constraint) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g
                h_1 = oo * c_1
                gw_1 = oogw * c_1                
                l_1 = ol * c_1
                bp_0 = ib * g
                h_0 = oo * c_0
                gw_0 = oogw * c_0                
                l_0 = ol * c_0
                lc_0 = ol_constraint * c_0               
            # save state     
                q_n[b,:] = h_0
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0
                gw_n[b,:] = gw_0    

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint 
                Gate_oogw[b,:] = oogw 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd#torch.addmm(torch.exp(self.weight_siga0), torch.exp(self.weight_siga1), ANLLtemp)

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, gw_n, Gate_ib, Gate_oo, Gate_ol, Gate_ol_constraint, Gate_f, Gate_oogw, h_nout, obs_std

class MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate_BYPASSM0(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate_BYPASSM0, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))           
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        self.theltaC = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))        
        self.relu = nn.ReLU()
        self.relu_l = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_gw = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
        self.theltaC = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])       
        h_n = torch.zeros([batch_size, hidden_size])
        gw_n = torch.zeros([batch_size, hidden_size])        
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_oogw = torch.zeros([batch_size, hidden_size])        
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        # MLE sigma
        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_yom_gw = (self.bias_b0_yom_gw.unsqueeze(0).expand(1, *self.bias_b0_yom_gw.size()))        
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))
        
        #torch.set_printoptions(precision=20)
        scale_factor = 1
        mo = p_mean
        ml = 2.9086
        so = p_std
        sl = 1.8980

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     
                px = self.relu(u1 + c_0 - torch.exp(self.theltaC) * scale_factor)
            # 1st cell
                if u1==0:
                   ib = 0
                elif u1>0:
                   ib = px / u1

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                oogw1 = torch.exp(self.weight_r_yom_gw)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oogw2 = torch.addmm(bias_b0_yom_gw, (c_0-mo)/so, self.weight_b1_yom_gw)
                oogw = oogw1 * torch.sigmoid(oogw2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 #u2 * torch.mm(c_0, self.weight_b3_ylm)
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:                
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - oogw - ol_constraint) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (u1 - px) 
                h_1 = oo * c_1
                gw_1 = oogw * c_1                
                l_1 = ol * c_1
                bp_0 = px
                h_0 = oo * c_0
                gw_0 = oogw * c_0                
                l_0 = ol * c_0
                lc_0 = ol_constraint * c_0               
            # save state     
                q_n[b,:] = h_0
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0
                gw_n[b,:] = gw_0    

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint 
                Gate_oogw[b,:] = oogw 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd#torch.addmm(torch.exp(self.weight_siga0), torch.exp(self.weight_siga1), ANLLtemp)

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, gw_n, Gate_ib, Gate_oo, Gate_ol, Gate_ol_constraint, Gate_f, Gate_oogw, h_nout, obs_std

class MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate_BYPASSM1(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Two_VariantOutputGate_BYPASSM1, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))           
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        
        self.weight_b1_yum = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yum = nn.Parameter(torch.FloatTensor(self.hidden_size))  

        self.relu_l = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_gw = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
        self.weight_b1_yum = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_yum = nn.Parameter(torch.rand(self.hidden_size))  

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])       
        h_n = torch.zeros([batch_size, hidden_size])
        gw_n = torch.zeros([batch_size, hidden_size])        
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_oogw = torch.zeros([batch_size, hidden_size])        
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        # MLE sigma
        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_yom_gw = (self.bias_b0_yom_gw.unsqueeze(0).expand(1, *self.bias_b0_yom_gw.size()))        
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))
        bias_b0_yum = (self.bias_b0_yum.unsqueeze(0).expand(1, *self.bias_b0_yum.size()))
        #torch.set_printoptions(precision=20)

        mo = p_mean
        ml = 2.9086
        so = p_std
        sl = 1.8980
        u1_max = 221.5190
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

            # 1st cell
                ib1 = torch.addmm(bias_b0_yum, (c_0-mo)/so, self.weight_b1_yum)
                ib2 = torch.mm(u1/u1_max, self.weight_b1_yum)                
                ib = torch.sigmoid(ib1+ib2)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                oogw1 = torch.exp(self.weight_r_yom_gw)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                oogw2 = torch.addmm(bias_b0_yom_gw, (c_0-mo)/so, self.weight_b1_yom_gw)
                oogw = oogw1 * torch.sigmoid(oogw2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 #u2 * torch.mm(c_0, self.weight_b3_ylm)
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:                
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - oogw - ol_constraint) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g
                h_1 = oo * c_1
                gw_1 = oogw * c_1                
                l_1 = ol * c_1
                bp_0 = ib * g
                h_0 = oo * c_0
                gw_0 = oogw * c_0                
                l_0 = ol * c_0
                lc_0 = ol_constraint * c_0               
            # save state     
                q_n[b,:] = h_0
                h_n[b,:] = h_0 + bp_0
                c_n[b,:] = c_0 # modify from c_1 to c_0
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0
                gw_n[b,:] = gw_0    

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint 
                Gate_oogw[b,:] = oogw 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd#torch.addmm(torch.exp(self.weight_siga0), torch.exp(self.weight_siga1), ANLLtemp)

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, c_n, l_n, lc_n, bp_n, gw_n, Gate_ib, Gate_oo, Gate_ol, Gate_ol_constraint, Gate_f, Gate_oogw, h_nout, obs_std

class MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_fp = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))                 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_fp = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom_fp = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))                   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))         
        self.relu_l = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_fp = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.weight_r_yom_gw = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))                 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_gw = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_fp = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom_fp = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
 
    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])        
        h_n = torch.zeros([batch_size, hidden_size])
        h_fp_n = torch.zeros([batch_size, hidden_size])       
        gw_n = torch.zeros([batch_size, hidden_size])        
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_oogw = torch.zeros([batch_size, hidden_size])
        Gate_oofp = torch.zeros([batch_size, hidden_size])                
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        # MLE sigma
        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_yom_gw = (self.bias_b0_yom_gw.unsqueeze(0).expand(1, *self.bias_b0_yom_gw.size()))        
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))
        bias_b0_yom_fp = (self.bias_b0_yom_fp.unsqueeze(0).expand(1, *self.bias_b0_yom_fp.size()))  

        #torch.set_printoptions(precision=20)

        mo = p_mean
        ml = 2.9086
        so = p_std
        sl = 1.8980

        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm) + torch.exp(self.weight_r_yom_fp))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                oogw1 = torch.exp(self.weight_r_yom_gw)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm)  + torch.exp(self.weight_r_yom_fp))
                oogw2 = torch.addmm(bias_b0_yom_gw, (c_0-mo)/so, self.weight_b1_yom_gw)
                oogw = oogw1 * torch.sigmoid(oogw2)

                oofp1 = torch.exp(self.weight_r_yom_fp)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm)  + torch.exp(self.weight_r_yom_fp))
                oofp2 = torch.addmm(bias_b0_yom_fp, (c_0-mo)/so, self.weight_b1_yom_fp)
                oofp = oofp1 * torch.sigmoid(oofp2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm)  + torch.exp(self.weight_r_yom_fp))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 #u2 * torch.mm(c_0, self.weight_b3_ylm)
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:                
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - oofp - oogw - ol_constraint) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g
                h_1 = oo * c_1
                h_fp_1 = oofp * c_1                
                gw_1 = oogw * c_1                
                l_1 = ol * c_1
                bp_0 = ib * g
                h_0 = oo * c_0
                h_fp_0 = oofp * c_0                
                gw_0 = oogw * c_0                
                l_0 = ol * c_0
                lc_0 = ol_constraint * c_0
            # save state     
                q_n[b,:] = h_0
                h_n[b,:] = h_0 + bp_0
                h_fp_n[b,:] = h_fp_0             
                c_n[b,:] = c_0 # modify from c_1 to c_0
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0
                gw_n[b,:] = gw_0    

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_oofp[b,:] = oofp
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint 
                Gate_oogw[b,:] = oogw 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd#torch.addmm(torch.exp(self.weight_siga0), torch.exp(self.weight_siga1), ANLLtemp)

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, h_fp_n, c_n, l_n, lc_n, bp_n, gw_n, Gate_ib, Gate_oo, Gate_oofp, Gate_ol, Gate_ol_constraint, Gate_f, Gate_oogw, h_nout, obs_std

class MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate_BYPASSM0(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate_BYPASSM0, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_fp = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))                 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_fp = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom_fp = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))                   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))   
        self.theltaC = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))                 
        self.relu_l = nn.ReLU()
        self.relu = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_fp = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.weight_r_yom_gw = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))                 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_gw = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_fp = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom_fp = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
        self.theltaC = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))    

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])        
        h_n = torch.zeros([batch_size, hidden_size])
        h_fp_n = torch.zeros([batch_size, hidden_size])       
        gw_n = torch.zeros([batch_size, hidden_size])        
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_oogw = torch.zeros([batch_size, hidden_size])
        Gate_oofp = torch.zeros([batch_size, hidden_size])                
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        # MLE sigma
        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_yom_gw = (self.bias_b0_yom_gw.unsqueeze(0).expand(1, *self.bias_b0_yom_gw.size()))        
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))
        bias_b0_yom_fp = (self.bias_b0_yom_fp.unsqueeze(0).expand(1, *self.bias_b0_yom_fp.size()))  

        #torch.set_printoptions(precision=20)

        mo = p_mean
        ml = 2.9086
        so = p_std
        sl = 1.8980
        scale_factor = 1
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     
                px = self.relu(u1 + c_0 - torch.exp(self.theltaC) * scale_factor)
            # 1st cell
                if u1==0:
                   ib = 0
                elif u1>0:
                   ib = px / u1 

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm) + torch.exp(self.weight_r_yom_fp))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                oogw1 = torch.exp(self.weight_r_yom_gw)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm)  + torch.exp(self.weight_r_yom_fp))
                oogw2 = torch.addmm(bias_b0_yom_gw, (c_0-mo)/so, self.weight_b1_yom_gw)
                oogw = oogw1 * torch.sigmoid(oogw2)

                oofp1 = torch.exp(self.weight_r_yom_fp)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm)  + torch.exp(self.weight_r_yom_fp))
                oofp2 = torch.addmm(bias_b0_yom_fp, (c_0-mo)/so, self.weight_b1_yom_fp)
                oofp = oofp1 * torch.sigmoid(oofp2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm)  + torch.exp(self.weight_r_yom_fp))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 #u2 * torch.mm(c_0, self.weight_b3_ylm)
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:                
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - oofp - oogw - ol_constraint) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (u1 - px)
                h_1 = oo * c_1
                h_fp_1 = oofp * c_1                
                gw_1 = oogw * c_1                
                l_1 = ol * c_1
                bp_0 = px
                h_0 = oo * c_0
                h_fp_0 = oofp * c_0                
                gw_0 = oogw * c_0                
                l_0 = ol * c_0
                lc_0 = ol_constraint * c_0
            # save state     
                q_n[b,:] = h_0
                h_n[b,:] = h_0 + bp_0
                h_fp_n[b,:] = h_fp_0             
                c_n[b,:] = c_0 # modify from c_1 to c_0
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0
                gw_n[b,:] = gw_0    

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_oofp[b,:] = oofp
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint 
                Gate_oogw[b,:] = oogw 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd#torch.addmm(torch.exp(self.weight_siga0), torch.exp(self.weight_siga1), ANLLtemp)

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, h_fp_n, c_n, l_n, lc_n, bp_n, gw_n, Gate_ib, Gate_oo, Gate_oofp, Gate_ol, Gate_ol_constraint, Gate_f, Gate_oogw, h_nout, obs_std

class MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate_BYPASSM1(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_Generic_PETconstraint_Three_VariantOutputGate_BYPASSM1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_fp = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))                 
        self.weight_r_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))  
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_fp = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom_fp = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))                   
        self.bias_b0_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))    
        self.weight_b1_yum = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.bias_b0_yum = nn.Parameter(torch.FloatTensor(self.hidden_size))  

        self.relu_l = nn.ReLU()
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yom_fp = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))         
        self.weight_r_yom_gw = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))                 
        self.weight_r_ylm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))   
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_gw = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom_gw = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size)) 
        self.bias_b0_yom_fp = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom_fp = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
        self.bias_b0_ylm = nn.Parameter(torch.rand(self.hidden_size))        
        self.weight_b2_ylm = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))         
        self.weight_b1_yum = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
        self.bias_b0_yum = nn.Parameter(torch.rand(self.hidden_size))  

    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):

        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()

        hidden_size = self.hidden_size
        #hidden_sizeM = self.hidden_sizeM
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        l_n = torch.zeros([batch_size, hidden_size])
        lc_n = torch.zeros([batch_size, hidden_size])        
        h_n = torch.zeros([batch_size, hidden_size])
        h_fp_n = torch.zeros([batch_size, hidden_size])       
        gw_n = torch.zeros([batch_size, hidden_size])        
        c_n = torch.zeros([batch_size, hidden_size])
        bp_n = torch.zeros([batch_size, hidden_size])
        q_n = torch.zeros([batch_size, hidden_size])        

        # Gate Function
        Gate_ib = torch.zeros([batch_size, hidden_size])
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_oogw = torch.zeros([batch_size, hidden_size])
        Gate_oofp = torch.zeros([batch_size, hidden_size])                
        Gate_ol = torch.zeros([batch_size, hidden_size])
        Gate_ol_constraint = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])

        # MLE sigma
        obs_std = torch.zeros([batch_size, hidden_size])

        # expand bias vectors to batch size
        bias_b0_yum = (self.bias_b0_yum.unsqueeze(0).expand(1, *self.bias_b0_yum.size()))
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))
        bias_b0_yom_gw = (self.bias_b0_yom_gw.unsqueeze(0).expand(1, *self.bias_b0_yom_gw.size()))        
        bias_b0_ylm = (self.bias_b0_ylm.unsqueeze(0).expand(1, *self.bias_b0_ylm.size()))
        bias_b0_yom_fp = (self.bias_b0_yom_fp.unsqueeze(0).expand(1, *self.bias_b0_yom_fp.size()))  

        #torch.set_printoptions(precision=20)

        mo = p_mean
        ml = 2.9086
        so = p_std
        sl = 1.8980
        u1_max = 221.5190
        obsstd = torch.std(y_obs[self.spinLen:self.traintimeLen])  

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)
                u2 = x[t,b,1].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size) 
                usig = y_obs[b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)     

            # 1st cell
                ib1 = torch.addmm(bias_b0_yum, (c_0-mo)/so, self.weight_b1_yum)
                ib2 = torch.mm(u1/u1_max, self.weight_b1_yum)                
                ib = torch.sigmoid(ib1+ib2)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm) + torch.exp(self.weight_r_yom_fp))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)

                oogw1 = torch.exp(self.weight_r_yom_gw)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm)  + torch.exp(self.weight_r_yom_fp))
                oogw2 = torch.addmm(bias_b0_yom_gw, (c_0-mo)/so, self.weight_b1_yom_gw)
                oogw = oogw1 * torch.sigmoid(oogw2)

                oofp1 = torch.exp(self.weight_r_yom_fp)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm)  + torch.exp(self.weight_r_yom_fp))
                oofp2 = torch.addmm(bias_b0_yom_fp, (c_0-mo)/so, self.weight_b1_yom_fp)
                oofp = oofp1 * torch.sigmoid(oofp2)

                ol1 = torch.exp(self.weight_r_ylm)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yom_gw) + torch.exp(self.weight_r_ylm) + torch.exp(self.weight_r_yfm)  + torch.exp(self.weight_r_yom_fp))
                ol2 = 0 #torch.mm((u2_1-ml)/sl, self.weight_b1_ylm)
                ol3 = torch.addmm(bias_b0_ylm, (u2-ml)/sl, self.weight_b2_ylm)
                ol4 = 0 #u2 * torch.mm(c_0, self.weight_b3_ylm)
                ol = ol1 * torch.sigmoid(ol2 + ol3 + ol4)

                if c_0 > 0:                
                    ol_constraint = ol - self.relu_l(ol - u2/c_0)
                else:
                    ol_constraint = ol

                f = (1.0 -  oo - oofp - oogw - ol_constraint) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g
                h_1 = oo * c_1
                h_fp_1 = oofp * c_1                
                gw_1 = oogw * c_1                
                l_1 = ol * c_1
                bp_0 = ib * g
                h_0 = oo * c_0
                h_fp_0 = oofp * c_0                
                gw_0 = oogw * c_0                
                l_0 = ol * c_0
                lc_0 = ol_constraint * c_0
            # save state     
                q_n[b,:] = h_0
                h_n[b,:] = h_0 + bp_0
                h_fp_n[b,:] = h_fp_0             
                c_n[b,:] = c_0 # modify from c_1 to c_0
                l_n[b,:] = l_0
                lc_n[b,:] = lc_0                
                bp_n[b,:] = bp_0
                gw_n[b,:] = gw_0    

            # save gate     
                Gate_ib[b,:] = ib
                Gate_oo[b,:] = oo
                Gate_ol[b,:] = ol
                Gate_oofp[b,:] = oofp
                Gate_f[b,:] = f
                Gate_ol_constraint[b,:] = ol_constraint 
                Gate_oogw[b,:] = oogw 

                h_x = (h_1, c_1)
                obs_std[b,:] = obsstd#torch.addmm(torch.exp(self.weight_siga0), torch.exp(self.weight_siga1), ANLLtemp)

        h_nout = torch.cat((h_n, obs_std), 1)

        return h_n, h_fp_n, c_n, l_n, lc_n, bp_n, gw_n, Gate_ib, Gate_oo, Gate_oofp, Gate_ol, Gate_ol_constraint, Gate_f, Gate_oogw, h_nout, obs_std

class MCPBRNN_SW_Variant_Routing(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_SW_Variant_Routing, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))  
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))        
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
    
    def forward(self, x, epoch, time_lag, y_obs, p_mean, p_std):

        if self.batch_first:
            x = x.transpose(0, 1)

        x = x[:,:, None]
        seq_len, batch_size, _ = x.size()
        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
      
        # Gate Function
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))

        mo = p_mean
        so = p_std

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)    

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, (c_0-mo)/so, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)
                f = (1.0 - oo) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                h_0 = oo * c_0                
                bp_0 = ib * g

            # save state     
                h_n[b,:] = h_0 
                c_n[b,:] = c_0 # modify from c_1 to c_0

            # save gate     
                Gate_oo[b,:] = oo
                Gate_f[b,:] = f 
                h_x = (h_1, c_1)
                
        return h_n, c_n, Gate_oo, Gate_f

class MCPBRNN_SW_Variant_Routing_Norm(nn.Module):

    def __init__(self,
                 input_size: int,
                 gate_dim: int,
                 spinLen: int,
                 traintimeLen: int,
                 batch_first: bool = True,
                 hidden_size: int = 1,
                 initial_forget_bias: int = 0):
        super(MCPBRNN_SW_Variant_Routing_Norm, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size      
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias
        self.gate_dim = gate_dim  
        self.spinLen = spinLen  
        self.traintimeLen = traintimeLen       
        # create tensors of learnable parameters                   
        self.weight_r_yom = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size)) 
        self.weight_r_yfm = nn.Parameter(torch.FloatTensor(self.hidden_size,self.hidden_size))         
        self.bias_b0_yom = nn.Parameter(torch.FloatTensor(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):

        self.weight_r_yom = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))  
        self.weight_r_yfm = nn.Parameter(torch.rand(self.hidden_size,self.hidden_size))        
        self.bias_b0_yom = nn.Parameter(torch.rand(self.hidden_size))  
        self.weight_b1_yom = nn.Parameter(torch.rand(self.hidden_size, self.hidden_size))
    
    def forward(self, x, epoch, time_lag, y_obs, p_norm):

        if self.batch_first:
            x = x.transpose(0, 1)

        x = x[:,:, None]
        seq_len, batch_size, _ = x.size()
        hidden_size = self.hidden_size
     
        h_0 = x.data.new(1, self.hidden_size).zero_()
        c_0 = x.data.new(1, self.hidden_size).zero_()

        h_x = (h_0, c_0)

        # Temporarily 2 dimension when seq_length = 1
        h_n = torch.zeros([batch_size, hidden_size])
        c_n = torch.zeros([batch_size, hidden_size])
      
        # Gate Function
        Gate_oo = torch.zeros([batch_size, hidden_size])
        Gate_f = torch.zeros([batch_size, hidden_size])
        bias_b0_yom = (self.bias_b0_yom.unsqueeze(0).expand(1, *self.bias_b0_yom.size()))

        #mo = p_mean
        #so = p_std

        for b in range(0+time_lag, batch_size):            
            for t in range(seq_len):
           
                h_0, c_0 = h_x                                        
            # calculate gates            
                u1 = x[t,b,0].unsqueeze(0).unsqueeze(0).expand(-1, hidden_size)    

            # 1st cell
                #ib1 = torch.mm(c_0, self.weight_b1_yum)
                #ib2 = torch.addmm(bias_b0_yum, u1, self.weight_b2_yum)
                #ib3 = u1 * torch.mm(c_0, self.weight_b3_yum)
                ib = 0 #torch.sigmoid(ib1 + ib2 + ib3)

                oo1 = torch.exp(self.weight_r_yom)/(torch.exp(self.weight_r_yom) + torch.exp(self.weight_r_yfm))
                oo2 = torch.addmm(bias_b0_yom, c_0/p_norm, self.weight_b1_yom)
                oo = oo1 * torch.sigmoid(oo2)
                f = (1.0 - oo) 

                g = u1
            # update state for next timestep
                c_1 = f * c_0 + (1.0 - ib) * g                     
                h_1 = oo * c_1
                h_0 = oo * c_0                
                bp_0 = ib * g

            # save state     
                h_n[b,:] = h_0 
                c_n[b,:] = c_0 # modify from c_1 to c_0

            # save gate     
                Gate_oo[b,:] = oo
                Gate_f[b,:] = f 
                h_x = (h_1, c_1)
                
        return h_n, c_n, Gate_oo, Gate_f