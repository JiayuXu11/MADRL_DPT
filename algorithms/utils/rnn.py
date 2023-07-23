import torch
import torch.nn as nn

"""RNN modules."""


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, recurrent_N, use_orthogonal,rnn_name='GRU'):
        super(RNNLayer, self).__init__()
        self._recurrent_N = recurrent_N
        self._use_orthogonal = use_orthogonal

        
        if rnn_name == "GRU":
            self.rnn = nn.GRU(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        elif rnn_name == "LSTM":
            self.rnn = nn.LSTM(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        elif rnn_name == "RNN":
            self.rnn = nn.RNN(inputs_dim, outputs_dim, num_layers=self._recurrent_N)
        else:
            raise ValueError("Invalid RNN type. Please choose from 'GRU', 'LSTM', or 'RNN'.")

        # # 一些实验（开始）
        # def print_name(m):
        #     classname = m.__class__.__name__
        #     print(classname,classname.find('Linear'))
        # self.rnn.apply(print_name)
        # # 一些实验（结束）
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, hcs, masks):
        if x.size(0) == hxs.size(0):
            if isinstance(self.rnn, nn.LSTM):
                hidden_state = ((hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous(),
                                (hcs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
                x, (hxs,hcs) =self.rnn(x.unsqueeze(0),hidden_state)

                hcs = hcs.transpose(0,1)
            else:
                x, hxs = self.rnn(x.unsqueeze(0),
                                  (hxs * masks.repeat(1, self._recurrent_N).unsqueeze(-1)).transpose(0, 1).contiguous())
                
            x = x.squeeze(0)
            hxs = hxs.transpose(0, 1)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.transpose(0, 1)
            hcs = hcs.transpose(0,1)

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                if isinstance(self.rnn, nn.LSTM):
                    temp = ((hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous(),
                            (hcs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous(),)
                    rnn_scores, (hxs,hcs) = self.rnn(x[start_idx:end_idx], temp)
                else:
                    temp = (hxs * masks[start_idx].view(1, -1, 1).repeat(self._recurrent_N, 1, 1)).contiguous()
                    rnn_scores, hxs = self.rnn(x[start_idx:end_idx], temp)

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

            # flatten
            x = x.reshape(T * N, -1)
            hxs = hxs.transpose(0, 1)
            hcs = hcs.transpose(0, 1)

        x = self.norm(x)
        return x, hxs, hcs
