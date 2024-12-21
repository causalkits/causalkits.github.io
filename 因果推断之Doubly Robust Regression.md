


因果推断领域一个重要的研究方向是观察数据建模（Observational Data Modeling），这个领域一个重要的概念是Doubly Robust，想理解这个耳熟能详的概念其实是不容易的，特别是对于工业界做算法的同学，没有太多时间去钻研那些神秘莫测的公式。然后多数paper又是讳莫如深得列出一大堆公式来论证自己的合理性，虽然多数创新也就是加一个简单的结构或者正则项，但依然设置了一堵非常高的学术墙。所以这篇文档试图从一个更加易于理解的角度，阐述一下Doubly Robust，希望对于大家看论文或者做创新提供一把好梯子。还是老规矩，这篇文档里面会中英文夹杂，主要的目的也还是避免蹩脚的翻译，方便大家能够对应上paper中的内容。

在讲述Doubly Robust之前，我们还是要回到因果推断领域，回到我们需要解什么样的问题上来。大部分出现Doubly Robust的研究基本都会出现在CATE （Conditional Average Treatment Effect）的场景，一般情况下我们可以表示为（如果你对这个公式还不是很熟悉，说明你还是一个刚入坑因果推断的同学，后面的内容看起来不是那么顺畅其实也是正常的。

$$
\tau(X) =\mathbb{E}_{X}\{\mathbb{E}(y|T=1,X=x)-\mathbb{E}(y|T=0,X=x)\} 
$$

回到公式，简单地理解就是给定某个子人群  $X=x$  ，假设施加某个Treatment（ $T=1$ ）对应的增量是多少？这看起来并不是一个很难的问题，为什么要引入一个玄之又玄的Doubly Robust的概念呢？

故事要从  $\mathbb{E}(y|T=1,X=x)$ 说起，在很多paper中也被写成  $m(1,x)$ ，可以简单理解成一个模型能够产出不同子人群的outcome期望。这里你肯定有一个问题，明明都可以直接从样本中直接统计出来期望，为啥要用一个模型来预估呢？其实答案也很简单：一个样本不能同时存在treatment和contol组，如果不用模型，对于这样的potential outcome就无法通过统计得出了，这里需要注意 $\mathbb{E}_X$ 是针对所有的样本，既包含实验组也包含对照组的样本。另外这样的设计也是有利于设计更加honest的模型，比如可以通过其他的样本来预估  $m(1,x)$  ，从而减少Conditional Average Treatment Effect的variance的累积。

如果我们的数据分布完美的契合真实的分布  $\mathcal{P}$  ，以随机的方式施加Treatment，那么对应的$\tau(x)$可以直接通过数据统计进行估计。然而在Observational Data数据分布 $\tilde{\mathcal{P}}$下，施加Treatment的概率 $e(x)=P(T=1|X=x)$就不再是一个随机分布了，而是现实世界中真实收集的数据。在这个子人群上可能某些 $x_i$ 被施加Treatment的概率 $e(x)$会天然更高或者更低，这种情况下利用上述的方法计算得到的 $\tau(x)$ 就是一个bias estimation。

前面已经说过了通过简单的统计方法学到的是一个有偏的估计值，那有没有办法做一下debias呢？方法其实有很多，这里我先给出一个最简单的形式 $\mathbb{E}(\frac{Y}{e(X)})$ ，一般我们称之为IPW（Inverse Propensity Weighting），也就是针对更加有可能被放置到Treatment组的样本进行降权，而针对概率比较小的样本进行提权，寄希望通过这样的方式将$\tilde{\mathcal{P}}$ 还原到真实分布 $\mathcal{P}$，从而获得一个无偏的CATE的估计。


![](https://picx.zhimg.com/80/v2-7c7985031203926f341369f8ff015787_1440w.png?source=d16d100b)

$$
\tau_{plug-in}=\frac{10*1+40*1}{10+10+40+10}-\frac{30*1+20*1}{30+0+20+30}=0.0892
$$

$$
\tau_{IPW}=\frac{25*1+80*1}{25+25+80+20}-\frac{50*1+40*1}{50+0+40+60}=0.1
$$


这里画了一张图来帮助大家理解IPW，假设整体有150个样本，Observational Data里面 male的样本有50个，而female的样本有100个。在性别上其实存在confounding，也就是Treatment=1和Treatment=0的分布不一致， $e(x)$分别为2/5和1/2。Y=1和Y=0分别表示最后样本的outcome是1或者是0，count表示这样的样本个数，而pseudo_count则表示由IPW构造出来的加权数量。如果按照一开始提到的简单的统计方法，也就是plug-in的估计方法，得出来的ATE值为0.0892，而用如果用IPW估计出来的值是0.1。IPW形象地理解就是通过$\frac{count}{e(x)}$构造了一个pseudo_count，将实验组和对照组的样本都调成了150个，从而更加接近无偏的参数估计。其实这个例子也非常形象地展示了辛普森悖论的问题，是可以通过IPW的方法来校正的。

写到这里貌似这个有偏性的问题已经被解决了，然而现实肯定是残酷的，IPW最大的问题是不太稳定，特别是 $e(x)$接近0的时候，就会造成上述的参数收敛一致性（consistency）比较差，很容易被某个极端值带偏。更为严重的是如果 $e(x)$并不能准确预估Treatment的分配概率，就会带来更大的bias。为了解决这个问题，大神们又捣鼓出了AIPW(Augmented Inverse Propensity Weighting)：

$$
AIPW(X,T=1)=\frac{1}{n}\sum_i \frac{T}{e(x)}(y_i-m(1,x_i))+m(1,x_i)
$$

如果仔细观察这个公式，你会发现它没有针对样本整体去做Inverse Propensity Weighting，而是先通过模型预估出来一个 $m(1,x)$。如果这个样本是Treatment组样本 $T=1$ ，然后针对它和真实值之间的偏差做IPW：$\frac{T}{e(x)}(y_1-m(1,x_i))$ ，然后利用加权后的残差对 $m(1,x)$进行了修正。如果这个样本是Control组样本 $T=0$，就用 $m(1,x_i)$直接预估出来一个potential outcome，残差部分由于T=0所以完全不影响结果。

同样的操作也可以施加到$\mathbb{E}(y|T=0,X=x)$上，CATE的计算公式可以表示为：

$$
\tau(X)_{AIPW}=\frac{1}{n}\sum_i\{\frac{T}{e(x)}(y_1-m(1,x_i))\\
-\frac{1-T}{1-e(x)}(y_0-m(0,x_i))+(m(1,x_i)-m(0,x_i))\}
$$

这个增强之后的公式就具备了doubly robust的性质，上面公式里面有两个模型： $m(T,x)$和 $e(x)$ ，现在我们假设以下两种情况：

1.如果 $m(T,x)$预估得很精准，基本上与真实值一模一样，但是 $e(x)$ 预估不准，公式中第一二项都会变为0，公式退化为 ：$\tau(X)=\frac{1}{n}\sum_i(m(1,x_i)-m(0,x_i))$ ，而由于 $m(T,x)$是准确的预估，所以$\tau(x)$也是一个无偏估计

2.如果 $m(T,x)$预估地不准，但是 $e(x)$预估地很精准，公式会退化为：

$$
\begin{align}
\tau(X)=&\frac{1}{n}\sum_i\{(y_1-m(1,x_i))-(y_0-m(0,x_i))+(m(1,x_i)-m(0,x_i))\}\\
=&\frac{1}{n}\sum_i\{y_1-y_0\}
\end{align}
$$

也就是plug-in estimator，成功避免了 $m(T,x)$不准的情况。

所以这个参数估计的值无论在哪个模型的精度取得比较大提升的情况下，都会导致CATE精度更高，收敛速度也更快的性质就被成为double robust。

写到这里估计就会有同学产生跟我一样的疑惑，为啥单单挑这个一长串不利于记忆的公式来做改进呢？难道是这些大神们真的是通灵术超强，在意识海中凭空召唤了一个这么恶心又恰好满足doubly robust的公式？我也是翻了很多历史古籍，才发现这里面是有迹可循的，后面的内容将从我的理解方式逐步展开。
### 参数估计

要想理解上面的AIPW的公式以及doubly robust，还是要回到最基本的概念也就是参数估计，只不过在这里参数是特定的CATE。预估某个参数 $\Psi(\mathcal P)$，如果能够获得真实的分布 $\mathcal P$，那么我们可以豪不费力地给出一个无偏的预估值，比如 $\mathbb E(Y)$的真实值，也就是我们经常在文档中看到的plug-in Estimator。然而现实的残酷在于，我们只能获得n个Empirical Sample，可以记作分布 $\mathcal P_n$，如果需要从$\mathcal P_n$ 出发，去计算一个真实分布的某个参数的预估值 $\Psi(\mathcal P_n)$，它与真实的误差 $\Psi(\mathcal P_n) - \Psi(\mathcal P)$其实就取决于 $\mathcal P_n$ 和 $\mathcal P$ 的差异有多大。所以如果去探究某个参数最后的收敛性，问题就变成了随着 $n \rightarrow \infty$ ， $\mathcal P$ 和 $\mathcal P_n$ 有多接近。

既然聊到了收敛性，这里先简单科普一下convergence学术领域的常用的符号 $O(n^d)$以及 $o(n^d)$ 表达的含义:

1. 如果说到某个函数 $f(n)$是 $O(n^d)$，表达的意思是当 $n \rightarrow \infty$，$f(n)$和 $n^d$ 以同样的速率增长或者缩小。如果 $d<0$，就表示 $f(n)$和 $n^d$ 以同样的速率向0衰减。

2. 如果说到某个函数 $f(n)$是 $o(n^d)$，表达的意思是当 $n \rightarrow \infty$， $\frac{f(n)}{n^d} \rightarrow 0$ ，也就是 $f(n)$以小于 $n^d$ 的速率进行增长或者衰减。

3. 如果提到 $a_n$ 是 $O_p(n^d)$ ，一般是指Order in Probablity。可以理解为对于任意一个 $\epsilon>0$ ，都存在一个 $K >0$ ，使得当 $n \rightarrow \infty$ ，$Prob(|a_n/n^d|>K) <\epsilon$

4. 如果提到 $a_n$ 是 $o_p(n^d)$，一般是指当 $n \rightarrow \infty$,$Prob(|a_n/n^d|>K) <\epsilon$，表达的含义与2中表达的事同一个意思。

5. 有很多一致性的Estimator被称为“root-n consistent”，表达的含义其实就是 $\hat{\theta}-\theta = O_p(n^{-1/2})$ ，或者写成 $\sqrt{n}(\hat{\theta}-\theta)=O_p(1)$。从一个更加利于理解的角度看 $\sqrt{n}$ 会随着 $n \rightarrow \infty$ 趋近于无穷大，但是$\hat{\theta}-\theta$ 会随着 $n$ 的增加，把乘积拉到一个常数水平。

这些符号在很多阐述收敛性的文章中都会出现，这里啰嗦一点简单介绍一下，便于大家理解后面的内容。

在一些简单的情况下，我们可以推导出公式，进而给出对应的分布，例如我们可以简单推导一下$\Psi(\mathcal{P})\mathbb{E}(Y)=\mathcal{P}()=\mu$。

$$
\sqrt{n}(\Psi(\hat{\mathcal{P}})-\Psi(\mathcal{P}))\\ =\sqrt{n}(\frac{1}{n}\sum_i^nY_i-\mu)\\ =\sqrt{n}(\frac{1}{n}\sum_i^n(Y_i-\mu))\sim\mathcal{N}(0,\sigma^2)\\
$$

通过中心极限定理可以看到预估参数和真实参数之间的差异随着 $n \rightarrow \infty$ 过程中逐渐趋近一个正态分布，这就意味着 $\Psi(\hat{\mathcal{P}})-\Psi(\mathcal{P}))$ 具备 $O_p(n^{-1/2})$ 一致性。

然而对其他某些形式的参数则不一定能够给出一个简单的推导和数据分布，例如用模型预估出来的 $\Psi(\mathcal{P}) = m(1,x)=\hat{\mathbb{E}}(Y|T=1,X=x)$ ，这样的推导就几乎不可能了，那么在这样的情况下我们该如何计算这个偏差，并且判断参数估计的收敛性？

这时一个比较巧妙的方法就被提出来，并且很好地解决了这个问题：我们仔细观察$\sqrt{n}(\Psi(\mathcal P_n) - \Psi(\mathcal P))$这个公式，其中两项都是关于某个数据分布的函数，这两个分布又有千丝万缕的联系：$\mathcal{P}_n$ 是真实分布$\mathcal{P}_n$ 采样出来的一个分布，其中真实分布$\mathcal{P}$ 是未知的，如果我们能够把 $\Psi(\mathcal{P})$在已知分布$\mathcal{P}_n$ 处做一下泰勒展开，展开公式中除了 $\Psi(\mathcal{P}_n)$ ，还有 $\Psi'(\mathcal{P}_n)(\mathcal{P-\mathcal{P}_n})$和 $\Psi''(\mathcal{P}_n)(\mathcal{P-\mathcal{P}_n})^2$等。这样 $\sqrt{n}(\Psi(\mathcal P_n) - \Psi(\mathcal P))$相减的问题就转化成了求导数 $\sum_{m=1}^M\frac{\Psi^m(\mathcal{P}_n)}{m!}(\mathcal{P-\mathcal{P}_n})^m$ 问题了，想要多大的精度可以视倒数推导难度来综合决定了。
既然要用泰勒展开就涉及到一个亟待解决的问题：应该如何求导？基于这样一个朴素的问题，就引出了一个重要概念：Efficient Influence Function。

### Efficient Influence Function

有时候我也在想这个名字的出处，为什么叫做Efficient Influence，实在不太理解取这个名字除了增加理解的壁垒，还有什么其他的用意。既然提到如何进行求导，我们还是从微分的推导的视角来看看这个问题。

我们是希望求解的是：预估参数$\Psi(\mathcal{P})$关于生成数据的分布 $\mathcal{P}$在某个观测分布 $\mathcal{P}_t$ 导数，这是一个泛函的概念，我们可以假设在生成数据真实分布 $\mathcal{P}$一点点的扰动 $\tilde{\mathcal{P}}$就可以产生这个观测分布$\mathcal{P}_t$。这里引入了一个新概念

parametric submodel，也就是通过定义一个简单的线性的组合来表达这个扰动：

$$
\mathcal{P}_t=t\tilde{\mathcal{P}} + (1-t)\mathcal{P}
$$

可以认为每一个$\mathcal{P}_t$ 都对应着一个参数的子模型。这里我没有延续上文用到的 $\mathcal{P}_n$，而是用 $\mathcal{P}_t$ 来表示，主要是为了与论文中的符号保持一致，另外也在强调这是一个更加一般的分布表示。
当下面的微分方程存在的时候，就被称为pathwise differentiable

$$
\mathop{\lim}_{t\rightarrow 0} (\frac{\Psi({\mathcal{P_t}})-\Psi(\mathcal{P})}{t})=\frac{d \Psi(\mathcal{P}_t)}{d t} |_{t=0}
$$

这个是一个泛函的概念，简单可以理解成函数关于函数的微分。

$$
\frac{\mathrm{d}\Psi(t)}{\mathrm{d}t}=\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t}\frac{\mathrm{d}\mathcal{P}_t}{\mathrm{d}t}=\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t}(\tilde{\mathcal{P}}-\mathcal{P})
$$

基本的假设：

$$
\mathcal{P}=\{\mathcal{P}(o_0)...\mathcal{P}(o_j)...\mathcal{P}(o_n)\}
$$

如果从每一个观察样本角度来看，每一个样本都有自身真实概率分布，这个概率分布可以是已知或者未知的。

从全微分的角度看，也就是我们可以针对每个样本的真实分布来求导，可以进行一下的拆解：

$$
\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t}|_{t=0}=\sum^n_{j=0}\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t (o_j)}\frac{\mathrm{d}\mathcal{P}_t(o_j)}{\mathrm{d}t}
$$

由于$\{\mathcal{P}(o_0)...\mathcal{P}(o_j)...\mathcal{P}(o_n)\}$的概率总和为1，他们之间是存在某种牵连的，可以改造一下消除这个关联性。引入一个更加一般的函数 $\Phi(o,\mathcal{P})$来表示：

$$
\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t}|_{t=0}=\int \frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}(o)}\mathrm{d}\ \tilde{\mathcal{P}}(o)-\mathcal{P}(o)=\int \phi(o,\mathcal{P})\mathrm{d}\ \tilde{\mathcal{P}}(o)-\mathcal{P}(o)
$$

这里 $\phi(O,\mathcal{P})$就被成为Efficient Influence Function，或者被叫做canonical gradient，它主要表达了：随着生成数据分布发生一点点变化，对应的Estimand的敏感性。

这里 $\phi(O,\mathcal{P})$ 是我们构造的一个梯度，既然是构造，那么我们最好选择一个比较方便的形式，由于$\phi(O,\mathcal{P})$加上任意一个constant都是不敏感的，这是因为任何一个分布的积分结果都是1，所以加上任何一个constant都会相互抵消为0。因此可以不失一般性地把$\phi(O,\mathcal{P})$设定为一个期望为0的函数，也就是

$$
\mathbb{E}(\phi(O,\mathcal{P}))=\mathcal{P}(\phi(O,\mathcal{P}))=0
$$

所以

$$
\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t}|_{t=0}=\tilde{\mathcal{P}}(\phi(O,\mathcal{P}))=\mathbb{E}_{\tilde{\mathcal{P}}}(\phi(O,\mathcal{P}))
$$


这里有一个很有趣的问题是 $\tilde{\mathcal P}\{\phi(O,\tilde{\mathcal P})\}$是否为0？答案是$\tilde{\mathcal P}\{\phi(O,\tilde{\mathcal P})\}=0$，这是由于我们在定义pathwise differentiable的时候有这样一个表述：要求存在一个均值为0，方差是有限值，且 $\phi(O,\mathcal{P}_t)$ 满足上面积分公式。因此要求所有的parametric submodel都需要满足 $\mathcal{P}_t(\phi(O,\mathcal{P}_t))=0$ 。所以这里的问题就变成了是否存在这样一个Efficient Influence Function使得所有的parametric submodel的 $\mathcal{P}_t(\phi(O,\mathcal{P}_t))=0$ ？好吧，既然是假设，那么我们就假设存在了，如果我们很幸运推导出来某种形式满足这个条件，我们是不是可以用这个公式当做Efficient Influence Function来使用？所以这里我们先不急于证明是否存在，先看看后面咱们是否可以推导出某种公式满足这个条件。此外对于不同的路径， $\phi(O,\mathcal{P}_t)$ 也是呈现出不用的函数形式。

这里不是太容易理解，这里举一个 $\Psi(\mathcal{P})=\mathbb{E}(Y)$例子，它的Efficient Influence Function是： $\Phi(\mathcal{P})=Y-\Psi(\mathcal{P})$，代入到前面说到的公式可以得到： $\mathcal{P}_t(\Phi(\mathcal{P}_t))=\int [Y-\Psi(\mathcal{P}_t)] \mathrm{d} \mathcal{P}_t=0$

可以根据这个公式大致给出一个比较形象的理解：在某一个submodel $\mathcal{P}_t$ 分布上计算出来的Efficient Influence Function有点像在 $\mathcal{P}_t$ 上求得的所有样本关于该分布的梯度期望为0。这里你是不是联想到无偏估计了，无论数据分布是什么，关于参数的预估都有一个统一的无偏估计的表达式。

### 推导CATE的Efficient Influence Function

前面介绍了Efficient Influence Function的数学定义和物理含义，但是应该如何来计算 $\phi(O,\mathcal{P})$ ，这里有很多方法，但最为简单的方法叫做：point mass contamination。既然是我们想知道目标Estimand关于 $\mathcal{P}$附近的一点点波动的敏感性，那这个波动是不是可以是某一个样本 $\tilde{o}$ 。可以简单假设一下如果我们在 $\mathcal{P}$ 基础上扰动一个分布：

$$
\tilde{\mathcal{P}}=\{\begin{array}{l} 1 \ at\  \tilde{o} \\ 0\  for\  else \\ \end{array}
$$

 ，这个Dirac Delta分布只在$\tilde{o}$点概率密度为1，在其他的点都为0。做这样一个假设本质上是把问题简化到一个我们可以推到的情况，然后利用求得结果外推到更加一般的情况。现在我们就来审视一下这个degenerate的分布：$\mathcal{P}_t = (1-t)\mathcal{P}+t\mathbb{I}(\tilde{o})$ ，是完全符合pathwise differential属性的。另外我们也观察到一个现象，如果我们单看 $\tilde{o}$这个点的Efficient Influence Function $\Phi(\tilde{o},\mathcal{P})$，由前面推到的公式可以得知：

$\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t}|_{t=0}=\mathbb{E}_{\tilde{\mathcal{P}}}(\phi(O,\mathcal{P}))=\Phi(\tilde o,\mathcal{P})$ ，所以我们是可以通过$\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t}|_{t=0}$来反推 $\Phi(\tilde o,\mathcal{P})$，所以还是拿前面目标预估参数为例： $\Psi(\mathcal{P})=\mathbb{E}[\mathbb{E}(Y|T=1,X=x)]$，我们可以做一下简单的推导：

$$
\begin{align}
 \Psi(\mathcal{P}_t)&=\mathbb{E}_{\mathcal{P}_t}[\mathbb{E}_{\mathcal{P_t}}(Y|T=1,X=x)]\\ 
 &=\int y\frac{f_t(x,y,1)}{f_t(1,x)}f_t(x)\mathrm{d}y\\ 
 \end{align}
$$

注意一下上述公式里面的 $f_t(x,y,1)$ 和 $f_t(x,1)$，首先要关注的是这两个概率密度函数是关于 $t$ 的泛函。另外为什么要写成这种相除的形式呢？主要原因是 $f_t(y|x,t=1)$ 这个概率密度函数不太好直接给出，另外在处理 $t\mathbb{I}(\tilde{o})$这个分布时，条件概率并不好直接处理。下面的公式中会出现 $f_{t=0}(1,x,y)$，本质上就是 $\mathcal{P}$ 。

$$
\begin{align}  \frac{\mathrm{d}\Psi(\mathcal{P}_t)}{\mathrm{d}t} \Bigg\vert_{t=0} &=\frac{\mathrm{d}\int y\frac{f_t(x,y,1)f_t(x)}{f_t(1,x)}\mathrm{d}y\mathrm{d}x}{\mathrm{d}t}\Bigg|_{t=0}\\ &=\int(y\{\frac{f'_t(x,y,1)f_t(x)f_t(1,x)+f'_t(x)f_t(x,y,1)f_t(1,x)}{(f_t(1,x))^2}\\&-\frac{f_t'(1,x)f_t(x,y,1)f_t(x)}{(f_t(1,x))^2})\}\mathrm{d}y\mathrm{d}x\Bigg|_{t=0}\\ 
&=\int y\{\frac{(\mathbb{I}(\tilde{x},\tilde{y},1)-f_{t=0}(1,x,y))f_{t=0}(1,x)f_{t=0}(x)}{(f_{t=0}(1,x))^2}\\
&+\frac{(\mathbb{I}(\tilde{x})-f_{t=0}(x))f_{t=0}(x,y,1)f_{t=0}(1,x)}{(f_{t=0}(1,x))^2}\\
&-\frac{(\mathbb{I}(1,\tilde{x})-f_{t=0}(1,x))f_{t=0}(x,y,1)f_{t=0}(x)}{(f_{t=0}(1,x))^2}\}\mathrm{d}y\mathrm{d}x\\ 
&=\int y \frac{f(1,x,y)f(x)}{f(1,x)}(\frac{\mathbb{I}(\tilde{x},\tilde{y},1)}{f(1,x,y)}+\frac{\mathbb{I}(\tilde{x})}{f(x)}-\frac{\mathbb{I}(1,\tilde{x})}{f(1,x)}-1)\mathrm{d}y\mathrm{d}x\\ 
&=\frac{\mathbb{I}(\tilde{x})}{\pi(\tilde{x},\mathcal{P})}\tilde{y}+m(1,\tilde{x})-\frac{\mathbb{I}(\tilde{x})}{\pi(\tilde{x})}m(1,\tilde{x})-\Psi(\mathcal{P}) \\ 
&=\frac{\mathbb{I}(\tilde{x})}{\pi(\tilde{x},\mathcal{P})}(\tilde{y}-m(1,\tilde{x}))+m(1,\tilde{x})-\Psi(\mathcal{P})\\ 
&=\Phi(\tilde{o},\mathcal{P}) \end{align}
$$

这个公式是不是似曾相识，说实话这个CATE公式的推导其实困扰了我好多年，花了很多气力才把这些艰深莫测的公式理解清楚，但每次一看到这个奇怪的公式又觉得自己根本没有理解，所以在这里用这个例子当做一个里程碑的总结。这个公式就是 $\Psi(\mathcal{P}_t)$在某个点 $\tilde{o}(\tilde{x},\tilde{y},1)$对应的EIF，其中：

$m(1,x) = \int yf(y|1,x)\mathrm{d}y\mathrm{d}x$ ，也就是 $y$ 的条件概率的期望。

$\pi(\tilde{x},\mathcal{P}) = \frac{f(1,x)}{f(x)}|_{x=\tilde{x}}$ ，其实就是很多文献中写的Propensity Score： $e(\tilde{x})$。

依据我们对 $\Phi_{T=1}(O,\mathcal{P})=\frac{1}{n}\sum_{i=0}^{n}[\frac{\mathbb{I}(X_i)}{\pi(X_i,\mathcal{P})}(Y_i-m(1,X_i))+m(1,X_i)-\Psi_{T=1}(\mathcal{P})]$

同理我们也可以得出： $\Phi_{T=0}(O,\mathcal{P})=\frac{1}{n}\sum_{i=0}^{n}[\frac{\mathbb{I}(X_i)}{1-\pi(X_i,\mathcal{P})}(Y_i-m(0,X_i))+m(0,X_i)-\Psi_{T=0}(\mathcal{P})]$

至此我们把某个目标预估值与分布的导数求出来了，后面我们所有的假设都是基于EIF已经推到出来的基础之上，看看这个求出来的EIF到底有什么用。

### von Miles Extension

还记得前面的章节提到我们可以把目标参数 $\Psi(\mathcal{P})$ 在 $\mathcal{P}_n$ 点进行泰勒展开：

$$
\Psi(\mathcal{P})=\Psi(\mathcal{P}_n)+\sum_{m=1}^M\frac{\Psi^m(\mathcal{P}_n)}{m!}(\mathcal{P-\mathcal{P}_n})^m
$$

这里我们先简单看一下 $t=1$处的一阶展开：

$$
\Psi(\mathcal{P})=\Psi(\mathcal{P}_n)+\frac{\mathrm{d}\Psi(\mathcal{P}_n)}{\mathrm{d}t}|_{t=1}(0-1)+R(\mathcal{P},\mathcal{P}_n)
$$

注意下这里的并不是 $\frac{\mathrm{d}\Psi(\mathcal{P}_n)}{\mathrm{d}t}\Big|_{t=0}$ 而是 $\frac{\mathrm{d}\Psi(\mathcal{P}_n)}{\mathrm{d}t}\Big|_{t=1}$ ，原因主要是我们希望在分布 $\mathcal{P}_n$ 处进行泰勒展开，这是我们能够收集到的真实数据，可以做各种参数的预估和统计。那问题就变成了这个求导结果是什么？

答案也比较简单，根据前面推导的求导公式：

$$
\frac{\mathrm{d}\Psi(t)}{\mathrm{d}t}=\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t}\frac{\mathrm{d}\mathcal{P}_t}{\mathrm{d}t}=\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t}(\tilde{\mathcal{P}}-\mathcal{P})=(\tilde{\mathcal{P}}-\mathcal{P})(\phi(O,\mathcal{P}_t))
$$

所以当 $t=1$ 时，可以得到：

$$
\frac{\mathrm{d}\Psi(t)}{\mathrm{d}\mathcal{P}_t}\frac{\mathrm{d}\mathcal{P}_t}{\mathcal{d} t}\Big|_{t=1}=(\tilde{\mathcal{P}}-\mathcal{P})(\phi(O,\tilde{\mathcal{P}}))=-\mathcal{P}\phi(O,\tilde{\mathcal{P}})
$$

针对泰勒展开公式做一下简单的变换：

$$
\begin{align} \sqrt{n}(\Psi(\mathcal{P})-\Psi(\mathcal{P}_n))&=\sqrt{n}\ \frac{\mathrm{d}\Psi(\mathcal{P}_n)}{\mathrm{d}t}|_{t=1}(0-1)+R(\mathcal{P},\mathcal{P}_n)\\ &=\sqrt{n}\  \mathbb{E}_\mathcal{P}(\phi(O,\mathcal{P}_n))+R(\mathcal{P},\mathcal{P}_n)  \end{align}
$$

如果希望前者具备 $O_p(n^{-1/2})$一致性，就需要后面的一次展开项趋近于0，这个就是CATE预估领域出现各种算法的基石了，本质上都是在做如何让这一项趋近于0。如果更加细致的思考这个问题，你会发现即使一次项趋近于0，高阶余项残差 $R(\mathcal{P},\mathcal{P}_n)$也有可能是不趋近于0的。我们先把可能存在的更高阶的展开放到一边，根据我们对泰勒展开的理解，这样的一阶展开至少在一个比较小的范围内，这个小的范围可以理解为 $\mathcal{P}_n$ 与 $\mathcal{P}$ 分布差异没有那么大的情况下，是对公式左侧参数比较接近的一个估计。

### One-Step Estimator

为了保证$\sqrt{n}\  \mathbb{E}_\mathcal{P}(\phi(O,\mathcal{P}_n))$能够趋近于0，一个简单的做法是我们可以在公示的两边都减去这一项，可以得到：

$$
\begin{align} \sqrt{n}(\Psi(\mathcal{P})-[\Psi(\mathcal{P}_n)+\  \mathbb{E}_\mathcal{P}(\phi(O,\mathcal{P}_n))]) &=R(\mathcal{P},\mathcal{P}_n)  \end{align}
$$

我们可以把 $\Psi(\mathcal{P}_n)+\  \mathbb{E}_\mathcal{P}(\phi(O,\mathcal{P}_n))$看做一个Estimator，如果在一个小范围内，可以把 $\mathcal{P}_n$ 看做真实分布$\mathcal{P}$ 的一种近似，所以上面的公式可以写成：

$$
\Psi^\star(\mathcal{P}_n)=\Psi(\mathcal{P}_n)+\  \mathbb{E}_\mathcal{P}(\phi(O,\mathcal{P}_n))\approx \Psi(\mathcal{P}_n)+\frac{1}{n}\sum_{i}^n \phi(O_i,\mathcal{P}_n)
$$

### TMLE

Targeted Learning 的思路比较巧妙，它的主要思路是通过调整 $\hat{\mathcal{P}_n}$，从而找到某个参数的估计值$\Psi(\hat{\mathcal{P}_n})$ 是一个plug-in Estimator，这也就意味着用这个分布 $\hat{\mathcal{P}_n}$代入到One-Step Estimator使得求出来的其他项为0。这样描述比较抽象，这里我们就以前面推导的ATTE为例：

从One-Step Estimator构造出来的计算公式看，下面这个公式是一个unbiased估计。 

$$
\Psi_{T=1}(\mathcal{P})+\Phi_{T=1}(O,\mathcal{P})=\frac{1}{n}\sum_{i=0}^{n}[\frac{\mathbb{I}(X_i)}{\pi(X_i,\mathcal{P})}(Y_i-m(1,X_i))+m(1,X_i)]
$$

这里 $m(1,X_i)$是通过 $\mathcal{P}_n$ 训练出来的模型，如果我们希望训练出来的模型就是一个unbiased estimator，就需要要求：

$$
\frac{1}{n}\sum_{i=0}^{n}\frac{\mathbb{I}(X_i)}{\pi(X_i,\mathcal{P})}(Y_i-m(1,X_i))=0
$$

现实中肯定不存在这样的好事，所以就需要Tuning一下 $\mathcal{P}_n$ ，让它进行能产生一个模型 $m^\star(1,X_i)$使得

$\frac{1}{n}\sum_{i=0}^{n}\frac{\mathbb{I}(X_i)}{\pi(X_i,\mathcal{P})}(Y_i-m^\star(1,X_i))=0$ ，这样的好处就是把消除偏差的过程融入到整个训练过程中，得出来的模型就是一个无偏的估计。

那么如何做到这一点呢？一个简单的做法就是设计一个新的plug-in estimator:

$m(1,\hat{\mathcal{P}_n})=m(1,\mathcal{P}_n)+\epsilon \frac{1}{\pi(X_i,\mathcal{P}_n)}$ 选取一个合适的 $\epsilon$ 让其他项：

$\frac{1}{n}\sum_{i=0}^{n}\frac{\mathbb{I}(X_i)}{\pi(X_i,\mathcal{P})}(Y_i-m(1,X_i)-\epsilon)=0$，所以在训练模型 $m^\star(1,X_i)$过程中，通过模型训练一个 $\epsilon$ 让这个恒等式始终成立即可。看到这里，你大概也就能理解DragonNet里面提到的Targeted Regularization了，其实DragonNet也是我写这篇文章最原始的起心动念之一。

至此为止我们聊了一大圈，把One-Step Estimator和TMLE的基本原理搞清楚了，但并没有解释他们为何是Doubly Robust的。你可能要问了既然是讲Doubly Robust的文章，为什么要讲这么多不那么相关的概念？
**核心原因还是我觉得前面的讲述有助于咱们理解Doubly Robust，两者有异曲同工之妙。**

更加令我们困惑还不仅仅是为啥TMLE是Doubly Robust的，而是这个神奇的概念是怎么被提出来？这是一个非常值得探究的问题。所以下面我会从Victor Chernozhukov和Denis Chernozhukov这对父子的视角来阐述Doubly Robust。


为了讲清楚什么Doubly Robust，最好的方式还是从一个简单的例子展开：

### Partial Linear Regression

$$
\begin{align}
Y &= D\theta+g(X)+U, \mathbb{E}(U|X,D)=0\\
 D&=e(X)+V, \mathbb{E}(V|X)=0
\end{align}
$$

partial linear regression是非常典型的因果推断的建模，假设是treatment的效应 $D$ 与 $Y$ 是一个线性关系，并且这也是一个unconfounded问题，不存在潜在的不可以观测变量。这里面有几个概念需要了解一下，我们一般称 $g(x)$叫做nuisance parameter，表达的是：对于结果 $Y$ 产生影响，但是与Treatment不相关的参数。 $e(x)$是通常我们理解的propensity score。面对这样的假设和建模方式，我们想对 $\theta$ 进行参数估计，有很多方法可以使用，这些方法中有些是bias的，有些是unstable的，有些是doubly robust的。这里举一个例子，比如可以直接用第一个公式做一个参数回归，回归得到的 $\theta$ 就是最后的CATE，所以loss我们就可以构建成：

$$
l (X,D,Y;\theta,g)=\frac{1}{n}\sum(Y-\theta D-g(X))^2
$$

这里 $g(X)$可以理解成包含各种参数的某个模型，比如我们可以通过梯度下降对最后的$\theta$ 进行参数预估。但是这样的预估存在一个非常典型的问题，那就是我们没有用上

$$
D=e(X)+V
$$

相当于切断了covariate $X$ 和Treatment $D$ 之间的联系，这会带来$\theta$ 预估结果的偏差。除此之外按照我们对于凸函数的理解，$l(X,D,Y;\theta,g)$会在关于$\theta$ 的导数为0的地方取得极值，也就是$\frac{1}{n}\sum(Y-\theta D-g(X))D=0$，这样一个导数如果在 $g(X)$预估不准的情况下会引起$\frac{1}{n}\sum(Y-\theta D-g(X))D$ 有比较大的扰动，从而使得$\theta$ 的预估和 $g(X)$的预估扰动非常相关，比较难以收敛。其实这就是double robust想要解决的问题。那怎样才能获得一个更加robust的参数预估呢？
我们先做一个变换：

$$
Y-g(X)=\theta D+U
$$

对两边取一下条件概率下的期望值：

$$
\mathbb{E}(Y-g(X)|X)=\mathbb{E}(\theta D+U|X)
$$

化简以后可以得到

$$
\mathbb{E}(Y|X)-g(X)=\theta\mathbb{E}( D|X)
$$

然后和 $Y-g(X)=\theta D+U$ 左右进行相减得到：

$$
Y-\mathbb{E}(Y|X)=\theta (D-\mathbb{E}(D|X))+U
$$

通过上面的公式可以构造出来新的𝜃的预估loss，这里需要注意一个特别容易看错的点：**$g(x)$ 和$\mathbb{E}(Y|X)$ 分别表达的含义不一致**。前者是在control组情况下对应的outcome预估函数，后者是treatment和control样本放在一起的期望值，多数文章中用 $m(x)$来表示。上述等式其实就是R-Learner基本的推导公式，也是我们经常说到的Robinson的partialling-out方法。利用 $e(x)$来预估$\mathbb{E}(D|X)$，利用上述相同的方法这个预估得出的loss的导数是：

$$
\frac{1}{n}\sum(Y-\mathbb{E}(Y|X)-\theta (D-e(X)))(D-e(X))=0\\
\frac{1}{n}\sum(Y-\theta D-g(X))(D-e(X))=0
$$

这个导数即使在 $g(X)$预估不准的情况下，只需要 $e(X)$能够预估准确，也能够使得loss接近于0，从而有效抵抗由于 $g(X)$或者 $e(X)$扰动带来的收敛性问题，这其实就是Doubly Robust性质的直观理解。

### Neyman Orthogonality
理论上通过上面这个例子，我们大致就可以直观的理解了什么叫做Doubly Robust，在我看来也就够了。但是Double Machine Learning其实是一套很完整的理论体系，后面的内容我会从理论的视角继续阐述我对于Doubly Robust更多的理解，更加重要的是阐述一下Doubly Robust在CATE场景上构造的estimator为什么长成那个样子。
通过上面的例子，我们大致可以有一个感受，可以通过对loss进行求导，然后观察这个偏导本身是否具有doubly robust性质，就可以判定目前的参数预估是不是可以加速收敛。这个偏导就引出一个重要的概念，叫做**score function**，可以用$\varphi(X,Y,D;\hat{\theta},\hat{g})$表示，更加通用的写法可以写成$\varphi(O;\theta,\eta)$，这里$\eta$ 表述nuinsane parameters。一个无偏的预估器一定要满足，这个条件被称为**moment condition**：

$$
g(O;\hat{\theta},\hat{\eta})=\mathbb{E} (\varphi(O;\hat{\theta},\hat{\eta}))=0
$$

如果上面这个式子关于**nuinsanse parameter**的偏导在真实参数$\theta_0,g_0$处还等于0，也就是在真实参数值出很稳定健壮，我们就趁称这样的性质叫做**Neyman Orthogonality**，可以写成：

$$
\partial_{\eta}\mathbb{E} (\varphi(O;\hat{\theta},\hat{\eta}))|_{\eta=\eta_0,\theta=\theta_0}=0
$$

只要满足Neyman Orthogonality的参数预估方法就具备了Doubly Robust的性质，好了让各位看官久等了，我们总算说到主题了。

### 如何构造Doubly Robust的Estimator
在《Double/debiased machine learning for treatment and structural parameters》论文中给出了非常多的构造Doubly Robust参数预估的方法，里面也举了非常多的例子来说明具体的流程，我这里就不赘述了。
我最好奇的问题是如果遇到一个CATE预估的问题，我们该如何推导出来具备这样性质的Estimator？后面我会围绕这个问题逐步展开，我先来写一下CATE的数学表达形式：

$$
\begin{align}
Y &= m(D,X)+U, \mathbb{E}(U|X,D)=0\\
 D&=e(X)+V, \mathbb{E}(V|X)=0
\end{align}
$$

论文之中给出的满足Neyman Orthogonality的CATE预估公式的score function是：

$$
\psi(O;\theta,\eta)=(m(1,X)-m(0,X))+\frac{D(Y-m(1,X))}{e(X)}\\-\frac{(1-D)(Y-m(0,X))}{1-e(X)}-\Psi(X)
$$

这里的$\eta=\{m,e\}$，把上面的式子求期望=0，得出的$\Psi(X)$的预估结果是不是似曾相识，没错，就是前文讲述EIF时推导给出预估公式基本一致。你会不会也跟我一样发出相同的疑问，为什么从不同的角度出发得到的就够是如此的一致？这个问题直到我仔细研究Neyman orthogonal score和influence function的关系时，我才理解这其中的道理：他们都用了von Miles Extension这样的方法来研究这个问题，前者是研究关于模型分布$\mathcal{P}$的展开，而后者是关于模型参数$\eta$ 的展开，其实两者有非常紧密的联系。
Chernozhukov V提出可以用一下的公式来构造Neyman Orthogonalized score function：

$$
\psi(O;\theta,\eta) = \varphi(O;\theta,\beta)+\phi(O;\theta,\eta)
$$

其中$\varphi$ 是**origin score**，$\phi$ 是**efficient influence function**，你会不会很疑惑：为什么是这个公式？为什么这样就可以构造一个满足Neyman Orthogonality的score function？
这个问题的答案在《Locally robust semiparametric estimation》中进行了比较全面的回答，但是表述过于学术不太容易理解，这里我用简单的方式来阐述一下。
首先我们知道某个参数$\theta$ 是要满足moment condition：

$$
g(O;\hat{\theta},\hat{\eta})=\mathbb{E} [(\varphi (O;\hat{\theta},\hat{\eta})])=0
$$

这里就引入了**orign score**的概念，origin score $\varphi$ 你可以理解为从loss求偏导得出的。
然后我们想知道$\mathbb{E}[g(O;\hat{\theta},\hat{\eta})]$在真实参数$(\theta_0,\eta_0)$的偏导等于0，才能保证具备Neyman Orthogonality性质。这里我们借用一下前面EIF的概念

$$
\mathcal{P}_t=t\tilde{\mathcal{P}} + (1-t)\mathcal{P}
$$

也就是在真实样本分布做一个小小的扰动，本质上也就是$\eta$ 的预估产生产生了一个小小的扰动。用前面的概念 参考一下前面介绍的EIF推导过程，我们就可以得到下面的公式：

$$
 \frac{\partial \mathbb{E}_{\mathcal{P}_t} (\varphi(O;\hat{\theta},\hat{\eta}))}{\partial t} =\mathbb{E}_{\mathcal{\tilde{P}}}(\phi(O,\mathcal{P}))-\mathbb{E}_{\mathcal{P}}(\phi(O,\mathcal{P})) 
$$

这个公式咋一看让人费解，仔细看其实就是前面EIF章节讲述的内容，只不过这里换了一个写法，更加简单的理解这就是定义不同而已。在EIF那一节，我们研究的是目标参数在样本扰动下的EIF，而这里是moment condition在样本扰动下的EIF。这里就可以看出他们的联系了，满足moment condition的$\theta$ 本质上就是一个参数预估，所以只是换种形式来表达参数关于样本扰动的EIF。基于我们对EIF定义的认识可以得出对于，要求任何一个分布，对应的都满足$\mathbb{E}_{\mathcal{P}_t}[\phi(O,\mathcal{P}_t)]=0$，这个是我们前面描述的EIF本身的定义决定的。有从这个公式出发就可以证明$\psi(O;\theta,\eta) = \varphi(O;\theta,\beta)+\phi(O;\theta,\eta)$是一个满足Neyman Orthogonality的score function。

$$
\psi(O;\theta,\eta) = \varphi(O;\theta,\beta)+\phi(O;\theta,\eta)
$$

接下来我们来简单推导一下这个构造的Score Function为什么是Neyman Orthogonality？
首先我们从EIF的等式开始：

$$
\mathbb{E}_{\mathcal{P}_t}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]=0
$$

然后展开这个等式就可以得到：

$$
\Rightarrow t\mathbb{E}_{\mathcal{\tilde{P}}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]+(1-t)\mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]=0\\
\Rightarrow \frac{1}{t}\mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]=\mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]-\mathbb{E}_{\mathcal{\tilde{P}}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]
$$

在 $t\rightarrow 0$时，所以$\mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]=\mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}})]=0$，这是由EIF的定义决定的，所以可以得到：

$$
\frac{\mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]}{t}=-\mathbb{E}_{\mathcal{\tilde{P}}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]
$$

我们在左边公式的分子分母分别减去一个0，得到一个导数极限的表示方式：

$$
\Rightarrow \frac{\mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]-\mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}})]}{t-0}=-\mathbb{E}_{\mathcal{\tilde{P}}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]\\
\Rightarrow \frac{\partial \mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]}{\partial t}|_{t\rightarrow 0}= -\mathbb{E}_{\mathcal{\tilde{P}}}[\phi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}})]|_{t\rightarrow 0}\\
$$

上面的变化利用$\mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}})]=0$这个性质，从而将左侧变成了求微分的形式。

$$
\Rightarrow \frac{\partial \mathbb{E}_{\mathcal{P}}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]}{\partial t}|_{t\rightarrow 0}= -\mathbb{E}_{\mathcal{\tilde{P}}}[\phi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}})]\\
\Rightarrow \frac{\partial \mathbb{E}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]}{\partial t}|_{t\rightarrow 0}+ \mathbb{E}_{\mathcal{\tilde{P}}}[\phi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}})]=0\\
$$

在这里我们引入前面的关于$\varphi$ 的定义：

$$
\frac{\partial \mathbb{E}_{\mathcal{P}_t} (\varphi(O;\theta,\eta))}{\partial t}|_{t\rightarrow 0} =\int \phi(o;\theta,\eta)\mathrm{d}\mathcal{\tilde{P}(o)}=\mathbb{E}_{\mathcal{\tilde{P}}}(\phi(O,\mathcal{P}))
$$

代入上面的公式可以得到：

$$
\frac{\partial \mathbb{E}[\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]}{\partial t}|_{t\rightarrow 0}+\frac{\partial \mathbb{E}_{\mathcal{P}_t} (\varphi(O;\theta,\eta))}{\partial t}|_{t\rightarrow 0}=0
$$

至此我们就证明了：

$$
\psi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}}) = \varphi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}})+\phi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}})
$$

是满足**Neyman Orthogonolity**的，也就是

$$
\frac{\partial \mathbb{E}[\psi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]}{\partial t}|_{t\rightarrow 0}=0
$$

写到这里关于Doubly Robust我想表达的内容基本已经写得差不多了，看到了$\phi$ 这样的表达，你也就不难理解TMLE以及各种debiased estimator都是Doubly Robust的了。

我们可以做一个简单的链式拆解：

$$
\begin{align}
\phi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t}) 
=& \frac{\partial \mathbb{E}[\varphi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]}{\partial t}\\
=&\frac{\partial \mathbb{E}[\varphi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]}{\partial \eta}\frac{\partial \eta}{\partial t}
\end{align}
$$

补充一点我对$\phi(O;\theta_{\mathcal{P_t}},\eta_{\mathcal{P_t}})$一些自己的理解：这个EIF表达是moment condition $\mathbb{E}[\psi]$关于分布$\mathcal{P}_t$ 的微小抖动导致的Score Function期望的波动，这里的$\phi$ 也与前文提到的反映参数预估期望波动的EIF有比较大的差异，两者是不能混为一谈的。这也是我们在读EIF论文时看到它与Double Machine Learning有很多的相似之处，但是又好像有点差异本质的原因。但两者也是有很强的联系的，就如上面的公式所表达的$\frac{\partial \eta}{\partial t}$ 在某些情况下就是求某些参数的EIF，然后通过乘以$\frac{\partial \mathbb{E}[\varphi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]}{\partial \eta}$就可以得到moment condition的EIF。


如果对应到paritial linear regression的模型中$\varphi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})=D(Y-D\theta-g(x))$，这里有一个小技巧那就是：$g(X)=\mathbb{E}[Y-D\theta|X]$，可以理解成一个简单参数估计的公式表达。它对应的$\phi_g(\tilde{x},\mathcal{P})=\tilde{y}-\tilde{d}\theta-\mathbb{E}[Y-D\theta|X]$，所以

$$
\begin{align}
\frac{\partial g}{\partial t}=&\phi(O;\theta,g)\\
=&\mathbb{E}_{\mathcal{P}_t}\{\phi_g(\tilde{x},\mathcal{P})\}\\
=&Y-D\theta-\mathbb{E}[Y-D\theta|X]\\
=&Y-D\theta-g(X)
\end{align}
$$

对应的$\phi$ 可以用下面的公式进行计算：

$$
\begin{align}
\phi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}}) =&\frac{\partial \mathbb{E}[\varphi(O;\theta_{\mathcal{P}_t},\eta_{\mathcal{P}_t})]}{\partial \eta}\frac{\partial \eta}{\partial t}\\
=&\frac{\partial \mathbb{E}[\varphi(O;\theta_{\mathcal{P}_t},g_{\mathcal{P}_t})]}{\partial g}\frac{\partial g}{\partial t}\\
=&\mathbb{E} (-D)*(Y-D\theta-g(X))\\
=&-e(X)*(Y-D\theta-g(X))
\end{align}
$$

而Partial Linear Regression的Origin Score 是：$\varphi(O;\theta,g)=D(Y-D\theta-g(X))$，这个是从loss function求导得来的，把两者相加就可以得到一个满足**Neyman Orthogonality**的score function了：

$$
\begin{align}
\psi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}}) = &\varphi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}})+\phi(O;\theta_{\mathcal{P}},\eta_{\mathcal{P}})\\
=&(D-e(X))(Y-D\theta-g(X))
\end{align}
$$

针对PLR的问题我们总算是把故事讲完整了。那么回到咱们耳熟能详的CATE的score function的推导上。先看看普遍采用的Score Function：

$$
\psi(O;\theta,\eta)=(m(1,X)-m(0,X))+\frac{D(Y(1)-m(1,X))}{e(X)}\\-\frac{(1-D)(Y(0)-m(0,X))}{1-e(X)}-\Psi(X)
$$

我们如果对上面公式中 $m(1,X)$ 和 $e(X)$进行求导，就会发现它是满足在$\eta=\eta_0$处导数为0的条件的。所以上面的公式可以直接作为最终的score function使用，如果反推一下这个公式的loss function，大概就是：

$$
l(O;\theta,\eta)=\mathbb{E}\{(m(1,X)-m(0,X))+\frac{D(Y(1)-m(1,X))}{e(X)}\\-\frac{(1-D)(Y(0)-m(0,X))}{1-e(X)}-\Psi(X)\}^2
$$

所以某种程度上CATE的无偏估计的形式是double robust是一种巧合，正好满足了**Neyman Orthogonality**。我也试图从更加原始的一般的orgin score进行推导，看是否能够推导出来这个公式，一直没有想到一个特别好的loss function形式，另外涉及到 $m(1,X)$、$m(0,X)$以及 $e(X)$三个nuisance model，整个推导的过程也会比较复杂一些。最为简单的理解方式是我们把CATE当做一个参数，Loss Function设定为：$\mathbb{E}\{\Psi(\mathcal{P})-\Psi(\mathcal{P}_n)\}^2$，对应的Origin Score Function是$\mathbb{E}\{\Psi(\mathcal{P})-\Psi(\mathcal{P}_n)\}$，是不是又看到前面提到的One-Step Estimator熟悉的味道了，两者就非常和谐得串联起来了。




### 总结

Doubly Robust是Neyman-Rubin阵营一个重要的概念和性质，特别是在观察数据建模时各种样本偏差都会给最后的预估参数带来毁灭性的打击，Doubly Robust是能够找到的为数不多的好的收敛性质，就像溺水中必须要抓紧的稻草一样。然而更为诡异的是这么重要的一个概念却缺少非常系统的资料可以传播查询，以至于入行多年的因果推断领域的专家也未必能够把前因后果联系到一起。写这篇文章之前我一直也有这样的困扰，前后花了很多时间才打通任督二脉，本以为可深藏功与名了。但现实是每次团队经历人员大的变动，就需要从头开始给大家恶补基础知识，又苦于无法找到一个系统性的资料让所有人自学起来。我本来是一个极其慵懒的人，但是本着更好的知识传播效率，后面我可以直接将这篇文章丢给某个人说：武功秘籍在这里，回去潜心研读。于是痛下决心，不知不觉写了这么多，行文没有太多考虑易读性，证明和阐述也难免有很多疏漏之处，但希望能够做到领大家入门即可，我会在收到大家的反馈后不定时来完善一下。

### 引用


[1] Hines O, Dukes O, Diaz-Ordaz K, et al. Demystifying statistical learning based on efficient influence functions[J]. The American Statistician, 2022, 76(3): 292-304.

[2] Fisher A, Kennedy E H. Visually communicating and teaching intuition for influence functions[J]. The American Statistician, 2021, 75(2): 162-172.

[3] L. Magee. [Asymptotic Concepts](https://socialsciences.mcmaster.ca/magee/761_762/other%20material/asy%20concepts.pdf)

[4] Chernozhukov V, Chetverikov D, Demirer M, et al. Double/debiased machine learning for treatment and structural parameters[J]. 2018.

[5] Chernozhukov V, Escanciano J C, Ichimura H, et al. Locally robust semiparametric estimation[J]. Econometrica, 2022, 90(4): 1501-1535.

[6] Ichimura H, Newey W K. The influence function of semiparametric estimators[J]. Quantitative Economics, 2022, 13(1): 29-61.
