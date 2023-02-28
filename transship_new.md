$D_{kn}$:retailer $k$'s demand at period $n$  
$I_{kn}$:retailer $k$'s inventory at the end of period $n$  
$\hat{I}_{kn}$:retailer $k$'s inventory at the start of period $n$
$x_{kn}$:transship quantity  
$q_{kn}$:order quantity  
$S_{kn}$:intransit order
$\hat{I}_{kn}=I_{kn-1}+x_{kn}+S_{kn-L}$
$I_{kn}=(\hat{I}_{kn}-D_{kn})^+$

$C_{kn}=-c(x_{kn}+q_{kn})-\tau (x_{kn})^+ +SC*I(q_{kn}>0)+r_{kn}E[\hat{I}_{kn} \cap D_{kn}]-h_{kn}E[\hat{I}_{kn}-D_{kn}]^+ - p_{kn}E[\hat{I}_{kn}-D_{kn}]^-$  

actor $k$'s observation:  
$o_{kn}=(I_{kn-1},D_{kn-1},S_{kn-L}...S_{kn-1})$

heuristic: 6.296
no transship:6.237
MADRL: 5.925

## 目前进展:
1.	基本完成考虑transship的多期库存问题的建模
2.	对happo强化学习框架熟悉并予以多次尝试

## 后续计划：
1. 学习maddpg框架（Multi-agent Deep Deterministic Policy Gradient）。在对happo尝试过程中发现训练效果并不好，猜测是action space较大导致policy network的训练较为困难，而maddpg中的policy network的输出为某一确定action，而非每个action的概率，故可有效应对action space这一问题。（论文：Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments，代码：https://github.com/openai/maddpg）
2. 尝试令各retailer共享policy，即只训练一个policy network以供所有agent使用。当各retailer的参数设定基本一致时，retailer间具有同质性，policy应当也不会有太大变化，可以尝试让各retailer共享一个policy network，即由多智能体强化学习转为单智能体强化学习了。可作为一个baseline，对比多智能体强化学习的效果。
3. 尝试centralized controller作为baseline，通过调整transship的交易费用来试图使得decentral接近central的效果。
4. 灵敏度分析。检验多智能体强化学习在不同设定下的效果：
   + Transship的运输费用为fixed cost 
   + 考虑不同零售商间的transship运输费用不同的情况 
   + Transship有lead time，如1天  
   + 不同零售商的需求并非独立，而存在一定的联动  
  
寒假计划先完成前3个

1. 看看在continue架构下，不要transship会不会更好，如果会更好，则说明transship的加入导致环境易变，干扰了训练，而非决策空间增大导致了训练困难
2. 先尝试centralized controller，以保证外部环境的相对稳定，其中transship的部分有以下几种处理方式：1. 原处理方式 2. 只输出n-1个agent的transship需求，第n个agent的transship为负的n-1个transship之和
3. 先让代码学习启发式算法试试，不然感觉门都找不到，v的估计直接令其必须大于拟合启发性策略得到的v
4. 让模型在需求相对静态的环境训练，即一个需求sample让模型多跑几遍。一来在同一需求sample下，v的估计会更准从而有助于训练；二来有助于agent之间默契习得transship功能



连续的训练，记得要换act.py中mode和sample的bound
newsvendor-continue时需要保证订货量为+，故直接在distributions.py中的DiagGaussian中加了个relu
把tensorboard用上，挺想看看均值与方差的变动情况（这个明天要搞一下了）
手动控制波动性？训练前期需要对策略进行充分的探索，而后面慢慢要追求一个更稳定的策略。每次调整波动性，就重新加载验证集表现最好的那个模型 20 10 5 1 0
让标准差也可以自己学
happo_trainer的return_batch看一下
学出来经常是-2、30，但实际上是不能订负的货的，故会直接不订货。但实际求概率的时候，又是用-2、30去算的概率，那这样算出来就是0，也就是不订货的概率是0，但实际是100%，显然概率失真了。但如果直接加relu又一定程度上剥夺了不订货的部分可能，如果让他可以根据环境调整标准差或许会有改善。（选择采用了relu+手动调整标准差，果然好了非常多）
这个eval是一个一个跑的，多少是有点慢,也可以改，但有点懒得改了

博弈式训练，就是当训练某一个agent时，固定另外两个agent

展示不同agent transship的总量

训练结果还是比不上heuristic，但是已经比newsvendor的训练情况好了，想想怎么改善：
把heuristic的中间变量拿出来训练看看？
加深网络深度，hidden_size
让他先学启发式算法

由于波动性的设置，他最多只会顶20的货，这可能是他表现不如baseline的原因
故采取了加大波动性，但又发现，当波动性较大时，如标准差设置为20，则网络输出的均值则不再是真正的均值。因为当均值较小时，分布有较大部分处于小于0的位置，而这一部分我们又会扔掉

方法1，通过改bias的方式
方法2，直接令输出的均值（一般是0-1）*100

当sigma越小，同样均值更新情况下，新旧概率的比值的振幅就会增大，从而带来以下两个影响：1. 更多的变化会被clip掉 2. 同样lr下变化会更快 快差不多4倍

torch.clamp用在最后一层上

把return改成了正常的return，把gae的sample mean也去掉了

cent_obs有重复列

直接让value network的输入为整个T期的需求,但这样会让value network较大，可以先用均值标准差试试

展示的observation_space有误导性，有时间可以改一下

这个normalize的均值和方差到底是多少

单agent试一下呢

gae下的return虽然也无偏，可以作为critic network学习的目标，但当critic network不准时，将严重影响数值的正确性，从而导致critic network学习较慢。可以先用真正的returns学，后面数值稳定了（至少正常了）再用gae的return学

value network滚动给均值

这个num_env_steps要改一下，不然每次跑太久了