$N\text{ - agent num}, \\
T\text{ - episode length}\\
l\text{ - lead time }\\
c_i\text{ - order cost per item}\\
h_i\text{ - holding cost per item}\\ p_i\text{ - penalty cost per item}\\
o^t_i\text{ - order quantity}\\
I^t_i\text{ - inventory after "pipline stock" arrive and transship}\\
\hat{I}^t_i\text{ - inventory after demand fulfillment}\\
d_i^t \text{ - demand}\\
TC_{ij}\text{ - transship cost from i to j}\\
M_{ij}\text{ - transship amount from i to j}\\
(S_{t-1}^{i},S_{t-2}^{i},...,S_{t-l}^{i}) \text{ -  in transit orders for agent i}$
 $$Obj = \sum_{t = 1}^T\sum_{i = 1}^N -c_i o^t_i  -  h_i [I^t_i - d^t_i]^+ - p_i [d^t_i - I^t_i]^- -\sum_{j = 1}^N  TC_{ij}(M_{ij})^- $$
 $I^t_i = \hat{I}^{t-1}_i + S_{t-l}^{i} - \sum_{j = 1}^N M_{ij}$
 $\hat{I}^{t}_i = (I^{t}_i-d_i^t)^+$
