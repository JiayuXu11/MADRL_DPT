
$N\text{ - agent num}, \\
T\text{ - episode length}\\
l\text{ - lead time }\\
c_i\text{ - order cost per item}\\
K_i\text{ - fixed cost for ordering}\\
h_i\text{ - holding cost per item}\\ p_i\text{ - penalty cost per item}\\
q^t_i\text{ - order quantity}\\
I^t_i\text{ - inventory after "pipline stock" arrive and transship}\\
\hat{I}^t_i\text{ - inventory after demand fulfillment}\\
d_i^t \text{ - demand}\\
TC_{ij}^t\text{ - transshiping cost from i to j}\\
M_{ij}^t\text{ - amount of transshipment from i to j}\\
(q_{i}^{t-1},q_{i}^{t-2},...,q_{i}^{t-l}) \text{ -  in transit orders for agent i}$

## central:
 $$C_{all}^t = \sum_{i = 1}^N (c_i q^t_i  + K_i \mathbb{I}(q^t_i>0)+  h_i [I^t_i - d^t_i]^+ + p_i [d^t_i - I^t_i]^+ +\sum_{j = 1}^N  TC_{ij}^t(M_{ji}^t)^+ )$$
 $I^t_i = \hat{I}^{t-1}_i + q_{i}^{t-l} - \sum_{j = 1}^N M_{ij}$
 $\hat{I}^{t}_i = (I^{t}_i-d_i^t)^+$


## decentral:
$r \text{ - revenue from transshiping per unit}$
 $$C_i^t =  c_i q^t_i  + K_i \mathbb{I}(q^t_i>0) +  h_i [I^t_i - d^t_i]^+ + p_i [d^t_i - I^t_i]^+ +\sum_{j = 1}^N  TC_{ij}^t(M_{ji}^t)^+ -\sum_{j = 1}^N  rM_{ij}^t$$
 $I^t_i = \hat{I}^{t-1}_i + q_{i}^{t-l} - \sum_{j = 1}^N M_{ij}$
 $\hat{I}^{t}_i = (I^{t}_i-d_i^t)^+$

 ## Partially Observable Markov Game (POMG)
The decision variables for every retailer $i$ in the above problem are order quantity $q^t_i$ and the amount of transshipment from retailer $i$ to all the other retailers $M_{ij}^t$.  

However, the transshipment is based on communication and agreement between both retailers involved in the transaction, and cannot be unilaterally decided, which means the amount of transshipment from retailer $i$ to $j$, $M_{ij}^t$ needs to be jointly determined by retailer $i$ and retailer $j$. For example, retailer $i$ can only transship goods to retailer $j$ if retailer $i$ is willing to provide the goods and retailer $j$ is willing to accept them. If retailer $i$ is willing to provide the goods unilaterally but retailer $j$ is unwilling to accept them, the transshipment cannot be realized. Similarly, if retailer $j$ is unilaterally eager to receive the goods but retailer $i$ is unwilling to provide them, the transshipment cannot be realized either. 

To apply MADRL to the decentral problem, we introduce a new variable $m^t_i$, which represents the amount of goods retailer i want to get from transshipment. Afterward, We will follow the principle of allocating based on proximity (the minimum transshipment cost principle) to consolidate the demands of each retailer and generate the actual transshipment plan $M^t$ to be implemented.  

For any retailer $i$, we use MADRL to jointly make decisions on the order quantity $q^t_i$ and the desired amount of goods to be obtained through transshipment $m^t_i$, which means retailer $i$'s action is $a^t_i=(q^t_i,m^t_i)$. Based on this, we have a discrete action space for each retailer $i$, $A^i=\{(0,-m^{max}),...,(0,m^{max}),...,(q^{max},-m^{max}),...,(q^{max},m^{max})\}$, where $m^{max}$ denotes the maximum transshiping quantity in each period and $q^{max}$ denotes the maximum ordering quantity. 

We define actor i’s partial observation which contains on-hand inventory, demand, and in transit orders:
$$o^i_t=( \hat{I}^{t-1}_i, d_i^{t-1},q_{i}^{t-1},q_{i}^{t-2},...,q_{i}^{t-l} )$$

Each agent $i$ finds its optimal policy by minimizing a total discounted cost as
$$\max _{\pi^{i}} E_{a_{0: T}^{1} \sim \pi^{1}, \ldots, a_{0: T}^{N} \sim \pi^{N}, s_{0: T} \sim T}\left[-\sum_{t=0}^{T} \gamma^{t} C_{i}^{t}\right]$$
2. critic obs (所有的actor obs + 未来需求信息)
3. norm_cost (1.norm，使得reward均值为0，从而降低return波动范围2.based on demand因为每天的收益和需求高度相关，但需求本身是并不是决策变量。当reward较大程度上由决策变量主导时，会有更好的效果)
4. rnn+multi-discrete