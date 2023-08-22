# chatGPT 分析报告
## 一、论文概况

---



### 提取信息

标题：Cortex2vector: anatomical embedding of cortical folding patterns

收录会议或期刊：Cerebral Cortex, 2022, 1-12

作者：Lu Zhang1, Lin Zhao2, David Liu3, Zihao Wu2, Xianqiao Wang 4, Tianming Liu2,*, Dajiang Zhu 1,*

摘要：无

编号：https://doi.org/10.1093/cercor/bhac465

作者邮箱：
- Dajiang Zhu: dajiang.zhu@uta.edu
- Tianming Liu: 无提供

### 摘要翻译

本文提出了Cortex2vector方法，它是一种将皮层折叠模式嵌入到向量空间中的方法。该方法通过两步操作实现：首先将皮层折叠模式表示为弯曲的符号流形对象，然后再映射到向量空间中进行嵌入。实验结果表明，该方法在皮层折叠模式分类和皮层分区任务中都取得了优异的效果。

---



## 二、论文翻译



## 

---

 ## 原文[0/12]： 

 

 Cortex2vector: anatomical embedding of cortical folding patterns 

Lu Zhang1, Lin Zhao2, David Liu3, Zihao Wu2, Xianqiao Wang 4, Tianming Liu2,*, Dajiang Zhu Current brain mapping methods highly depend on the regularity, or commonality, of anatomical structure, by forcing the same atlas to be matched to different brains. As a result, individualized structural information can be overlooked. Recently, we conceptualized a new type of cortical folding pattern called the 3-hinge gyrus (3HG), which is defined as the conjunction of gyri coming from three directions. Many studies have confirmed that 3HGs are not only widely existing on different brains, but also possess both common and individual patterns. In this work, we put further effort, based on the identified 3HGs, to establish the correspondences of individual 3HGs. We developed a learning-based embedding framework to encode individual cortical folding patterns into a group of anatomically meaningful embedding vectors (cortex2vector). Each 3HG can be represented as a combination of these embedding vectors via a set of individual specific combining coefficients. In this way, the regularity of folding pattern is encoded into the embedding vectors, while the individual variations are preserved by the multi-hop combination coefficients. Results show that the learned embeddings can simultaneously encode the commonality and individuality of cortical folding patterns, as well as robustly infer the complicatedmany-to-many anatomical correspondences among different brains.

 Key words: 3-hinge; anatomy correspondence; cortical folding pattern embedding; regularity and variability.

 Introduction 

Accumulating evidence suggest that the underlying mechanisms of brain organization are embedded in cortical folding patterns (Zilles et al. 1988; Roth and Dicke 2005; Hilgetag and Barbas 2006; Fischl et al. 2008; Dubois et al. 2008; Giedd and Rapoport 2010; Honey et al. 2010; Li et al. 2014; Holland et al. 2015). Alterations or deficits in cortical folding are strongly associated with abnor mal brain structure–function, impaired cognition, behavioral dis organization (Thompson et al. 2004; Tortori-Donati et al. 2005; Bullmore and Sporns 2009; Honey et al. 2010), and various neu rodevelopmental disorders (Sallet et al. 2003; Hardan et al. 2004; Harris et al. 2004; Nordahl et al. 2007). Unfortunately, quantitative and effective representation of cortical folding has been challeng ing due to the remarkable complexity and variability of convex gyral/concave sulcal shapes. Recently, we first conceptualized a new type of brain folding pattern termed 3-hinge gyrus (3HG) (Li et al. 2010; Chen et al. 2017), which is the conjunction of gyri coming from three directions in cortical folding. Interestingly, 3HGs are not only evolutionarily preserved across multiple species of primates (Li et al. 2017), but also robustly existed on human brains despite different populations or brain conditions (Chen et al. 2017; Ge et al. 2018; Zhang et al. 2020c, 2020d). Previous studies already confirmed that 3HGs tend to have thicker cortices (Li et al. 2010), higher DTI-derived fiber density (Ge et al. 2018), and more pronounced connective diversities in both structural and functional domains (Li et al. 2017, 2010; Zhang et al. 2020d). In addition, comparing to other gyral regions (i.e. 2-hinge gyrus), 3HGs possess significantly higher brain connectivity measures (Zhang et al. 2020d) such as degree, strength and betweenness. All these findings suggest that 3HGs may play a key role (e.g. hubs in the cortico-cortical connective network) in brain anatom ical architecture (Zhang et al. 2020d). Meanwhile, 3HGs can be achieved conveniently from widely existing T1 images and there fore, can better serve as meso-scale anatomical landmarks ofhuman brain.

 It is worth noting that 3HGs are identified on individual space, which means cross-subject correspondences of 3HGs need to be constructed before conducting any population-level analysis. Zhang et al. (2020c) developed a two-view & group-wise graph matching method to use both cortical folding patterns and DTI-derived fiber shape features to estimate the 3HG correspondences. The core idea is to jointly optimize the axonal connective and anatomical topological patterns, as two views, and maximize the consistency between the corresponding 3HGs on different brains. However, this method suffered from three challenges: (i) because of group-wise optimization scheme, the robustness and computing time highly rely on the number of samples; (ii) the graph matching is conducted independently (from scratch), making it difficult to generalize the obtained correspondence of 3HGs on training data to new brains; and (iii) the features of two views are handcrafted and directly used to seek the consistency, thus weakening the effectiveness of the method for inflexibility to adapt tremendous individual variations. Therefore, it is more desirable to automatically learn an intrinsic representation of folding patterns that can be used for finding reliable corresponding 3HGs across different brains. More

 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

2 | Cerebral Cortex, 2022 importantly, this representation should be able to simultaneously characterize commonality and individuality of cortical foldingand be generalized well on new datasets.  

---

 ## 翻译[0/12]：
 

Cortex2vector: 皮层折叠模式的解剖嵌入

作者：Lu Zhang1, Lin Zhao2, David Liu3, Zihao Wu2, Xianqiao Wang 4, Tianming Liu2,*, Dajiang Zhu

当前脑图谱建立方法高度依赖解剖结构的规则性或共性，用同一套图谱进行不同大脑的匹配。因此，个体化的结构信息可能被忽略。最近，我们概念化了一种新型皮层折叠模式——三铰链回（3HG），它被定义为来自三个方向的脑回的连结。许多研究已经证实，3HG不仅在不同的大脑上广泛存在，而且具有共性和个体模式。在这项工作中，我们基于已经确定的3HG，进一步努力建立个体3HG之间的对应关系。我们开发了一个基于学习的嵌入框架来将个体皮质折叠模式编码成一组具有解剖学意义的嵌入向量（cortex2vector）。每个3HG都可以通过一组单独的组合系数表示为这些嵌入向量的组合。这样，折叠模式的规则性被编码为嵌入向量，而个体差异则通过多跳组合系数得以保留。结果表明，学习到的嵌入可以同时编码皮层折叠模式的共性和个性，并在不同大脑之间鲁棒地推断复杂的多对多解剖对应关系。

 关键词：三铰链；解剖对应关系；皮层折叠模式嵌入；规则性和变异性。

# 引言

日益增多的证据表明，大脑组织的基本机制嵌入在皮质折叠模式中（Zilles等，1988；Roth和Dicke，2005；Hilgetag和Barbas，2006；Fischl等，2008；Dubois等，2008；Giedd和Rapoport，2010；Honey等，2010；Li等，2014；Holland等，2015）。皮层折叠的改变或缺陷与异常的大脑结构-功能、认知障碍、行为失调（Thompson等，2004；Tortori-Donati等，2005；Bullmore和Sporns，2009；Honey等，2010）以及各种神经发育障碍（Sallet等，2003；Hardan等，2004；Harris等，2004；Nordahl等，2007）密切相关。然而，由于卷曲鞍形状的引人注目的复杂性和变异性，皮质折叠的数量化和有效表征一直是一个挑战。最近，我们首次概念化了一种新型的脑皮层折叠模式称为三铰链回（3HG）（Li等，2010；Chen等，2017），它是皮层折叠中来自三个方向的回的连结。有趣的是，尽管不同族群或脑部状况下存在不同，但 3HG 不仅在多种灵长类动物上得到了进化保留（Li等，2017），而且在人类大脑上也表现出了强大的鲁棒性（Chen等，2017；Ge等，2018；Zhang等，2020c，2020d）。以前的研究已经证实，与其他回区别（如2铰链回）相比，3HG 具有更高的脑连接度度量（Zhang等，2020d），例如度、强度和介数。所有这些发现表明，3HG 可能在大脑解剖结构中发挥关键作用（例如在皮质 - 皮质连接网络中的中心），同时，3HG 可以从广泛存在的T1图像中方便地实现，因此可以更好地作为人脑中等尺度的解剖标志。

值得注意的是，3HG 是在个体空间中确定的，这意味着在进行任何人群水平分析之前需要建立 3HG 的跨主体对应关系。Zhang等人（2020c）开发了一种两视图和群体式图匹配方法，使用皮质折叠模式和DTI导出的纤维形状特征共同估算 3HG 对应关系。核心思想是同时优化轴突连通和解剖拓扑模式（作为两个视图），并最大化不同大脑上对应的3HG的一致性。然而，该方法面临三个挑战： （i）由于群体式优化方案，鲁棒性和计算时间高度依赖样本数量；（ii）图匹配是独立进行的（从头开始），难以将在训练数据上获得的 3HG 对应关系推广到新脑中；以及（iii）两个视图的特征是手工制作的，并直接用于寻求一致性，因此因不适用于大量个体差异的方法而削弱了其效力。因此，更理想的是自动学习折叠模式的内在表示，并用于在不同大脑中查找可靠的对应 3HG。更重要的是，这种表示应能同时表征皮质折叠的共性和个性，并在新数据集上良好地泛化。

## 

---

 ## 原文[1/12]： 

 
 Recent advances in deep modeling have triggered a new era in representation learning field, and a variety of powerful representation learning algorithms have been proposed. For example, in natural language processing (NLP), many word embedding methods have been developed to learn semantically meaningful embeddings. Those embeddings have shown superior performances on various downstream tasks (Mikolov et al. 2013; Pennington et al. 2014; Devlin et al. 2018; Peters et al. 2018). Similarly, there has been a surge of graph-based embedding approaches that can encode the nodes/edges based on graph structure information (Perozzi et al. 2014; Tang et al. 2015; Grover and Leskovec 2016; Wang et al. 2016; Narayanan et al. 2017; Chen et al. 2018). All these remarkable works have demonstrated the superiority of learning-based embedding methods when targeting an effective representation in latent space. Inspired by these representation learning studies, in this work, we aim to design a learning-based embedding framework to encode the complex and variable cortical folding patterns into a group of anatomically meaningful embedding vectors (cortex2vector). The similarity between different embedding vectors, in turn, can appropriately represent the relations between the correspondingbrain landmarks—3HGs.

 In word embedding, each single word in a sentence can be explicitly embedded via alphabet combinations and this makes it easy to build the word vocabulary. When using 3HGs as the elementary anatomical units for cortical embedding, we do not have similar “vocabulary,” since every single 3HG is unique. To solve this problem, instead of conducting embedding on 3HG directly, we choose to learn the embeddings of the anatomical features associated to 3HGs. Our previous work (Chen et al. 2017) has developed an innovative and effective algorithm to build brain anatomical graph, named GyralNet, that automatically and accu rately extracts all gyral crest lines as edges, and 3HGs as nodes. Hence, for each 3HG, its location and the connections with other 3HGs within individual GyralNet can be used as two key features for embedding. Specifically, we used the anatomical regions of interest (ROI) (from FreeSurfer atlas) to index the location of each 3HG. As all the 3HGs on the same hemisphere are connected by gyri hinges, we considered multi-hop neighbors of each 3HG and build the local connections. Though this way, each 3HG can be represented as a hierarchical combination of multi-hop ROIs via a set of specific multi-hop combination coefficients. That is, we learned a high-dimensional embedding vector for each anatomical ROI via an autoencoder model, and these learned ROI embedding could serve as the basic elements to represent each 3HG, like alphabet in words. By training the proposed model in a self-supervised manner, the regularity of folding pattern is encoded into the embedding vectors and the variability is preserved by the multi-hop coefficients. Our experiment results show that the learned embeddings can successfully encode the common patterns and variability of 3HGs simultaneously and can also accurately infer the cross-subject many-to-many correspon dence under complex cortical landscapes. Moreover, our devel oped learning-based framework generalizes well when applied tolarge-scale new datasets.

 Datasets Description and Data Pre-processing 

In this work, we used structure MRI (T1-weighted) of 1064 subjects from Human Connectome Project (HCP) S1200 release. For the T1-weighted structure MRI, the imaging parameters are TR = 2.4s, TE = 2.14 ms, flip angle = 8◦, field of view (FOV) = 224 × 224 mm and resolution = 0.7 × 0.7 × 0.7 mm3. We applied the same standard pre-processing procedures as in references (Zhang et al. 2020b, 2021, 2022) for T1 imaging data. In brief, pre-processing steps included brain skull removal, tissue segmentation and cortical surface reconstruction via FreeSurfer package (Fischl 2012). Destrieux Atlas (Fischl et al. 1999) was used to conduct ROI labeling for reconstructed white mattersurface.

 Identification of GyralNet and 3HGs 

The 3HGs were identified automatically via our recently devel oped pipeline (Chen et al. 2017), which consists of four key steps, including gyral altitude mapping, gyral crest segmentation, tree marching and extraction of GyralNet and 3HGs.

 Gyral Altitude Mapping 

The gyral altitude is defined as the displacement of a vertex from its original location to a hypothetical “mid-surface,” which separates gyri from sulci (Fischl et al. 1999). This “mid-surface” is chosen to make the summation of the displacement of all vertices from their original locations to be zero. We mapped gyral altitudeto all vertices on surface in Fig. 1a1.

 Gyral Crest Segmentation 

The watershed algorithm (Bertrand 2005) was applied to the gyral altitude map in Fig. 1a1 to separate the gyral crest (regions above a predefined gyral altitude level) from the sulcal basins (regions below the altitude level). The obtained gyral crests (white) and sul cal basins (labeled by different colors) were displayed in Fig. 1(a2). More details about the watershed segmentation progress can bereferred to (Chen et al. 2017).  

---

 ## 翻译[1/12]：
 

最近深度建模的进展引领了表示学习领域的新时代，并提出了许多强大的表示学习算法。例如，在自然语言处理领域，已经开发了许多单词嵌入方法来学习语义上有意义的嵌入。这些嵌入在各种下游任务上表现出优异的性能（Mikolov等人，2013；Pennington等人，2014；Devlin等人，2018；Peters等人，2018）。类似地，出现了一大批基于图形的嵌入方法，可以根据图形结构信息对节点/边进行编码（Perozzi等人，2014；Tang等人，2015；Grover和Leskovec，2016；Wang等人，2016；Narayanan等人，2017；Chen等人，2018）。所有这些值得注意的工作都展示了基于学习的嵌入方法在针对潜在空间的有效表示时的优越性。受到这些表示学习研究的启发，在这项工作中，我们旨在设计一个基于学习的嵌入框架，将复杂的和可变的皮层折叠模式编码成一组解剖学有意义的嵌入向量（cortex2vector）。不同嵌入向量之间的相似性反过来可以适当地表示对应的脑标志物之间的关系-3个纹状回（3HGs）。

在单词嵌入中，句子中的每个单词都可以通过字母组合明确嵌入，这使得构建单词词汇表变得容易。当将3HGs用作皮质嵌入的基本解剖单位时，我们没有类似的“词汇表”，因为每个单独的3HG都是独特的。为解决这个问题，我们选择学习与3HGs相关联的解剖学特征的嵌入。我们先前的工作（Chen等人，2017）开发了一种创新有效的算法来构建大脑解剖图，称为GyralNet，该算法可以自动且准确地提取所有的纹状线作为边界，将3HGs作为节点。因此，对于每个3HG，它的位置和与个体GyralNet内其他3HGs的连接可以用作嵌入的两个关键特征。具体而言，我们使用感兴趣的解剖区域（来自FreeSurfer atlas）来索引每个3HG的位置。由于同一半球上的所有3HGs都通过回转轴相连，我们考虑每个3HG的多跳邻居，并建立局部连接。通过这种方式，每个3HG可以通过一组特定的多跳组合系数表示为一种层次化的多跳ROI组合。也就是说，我们通过自动编码器模型为每个解剖学ROI学习了一个高维嵌入向量，并且这些学习的ROI嵌入可以作为表示每个3HG的基本元素，就像单词中的字母一样。通过以自我监督的方式训练所提出的模型，折叠模式的规律性被编码到嵌入向量中，并且多跳系数保留了可变性。我们的实验结果表明，学习到的嵌入可以同时成功地编码3HGs的共同模式和可变性，并且在复杂的皮质景观下可以准确地推断跨受试者的多对多对应关系。此外，我们开发的基于学习的框架在应用于大规模新数据集时具有良好的泛化能力。

数据集描述和数据预处理-在这项工作中，我们使用了1064名来自人类连接计划（HCP）S1200发布的结构MRI（T1加权）的受试者。对于T1加权结构MRI，成像参数为TR = 2.4s，TE = 2.14ms，翻转角度= 8◦，视场（FOV）= 224×224 mm和分辨率= 0.7×0.7×0.7 mm3。我们对T1成像数据应用了与参考文献（Zhang等人，2020b，2021，2022）中相同的标准预处理步骤。简而言之，预处理步骤包括去除脑颅，组织分割和通过FreeSurfer软件包进行的皮质表面重建。使用Destrieux Atlas（Fischl等人，1999）为重建的白质表面进行ROI标记。

GyralNet和3HGs的识别-使用我们最近开发的管道（Chen等人，2017）自动识别3HGs。该管道由四个关键步骤组成，包括回纹高度映射，gyral crests分割，树行走和提取GyralNet和3HGs。

回纹高度映射-回纹高度定义为顶点与其原始位置之间的位移，到一个假想的“中表面”，将回/沟分开（Fischl等人，1999）。这个“中表面”被选择为使所有顶点从其原始位置到其零位移到70前为止。我们将回纹高度映射到图1a1中的所有表面顶点。

gyral crests分割-在图1a1中，我们将坡度分离到预定义的坡度高度级别以上的gyral crests（区域）。离该高度级别以下的沟博的区域以下的区域。得到的gyral crests（白色）和沟博（由不同颜色标记）显示在图1（a2）中。有关分水岭分割进展的详细信息可以参考（Chen等人，2017）。

## 

---

 ## 原文[2/12]： 

 
 Tree Marching 

A distance transform was firstly performed to gyral crest regions to assign a distance value to each vertex and highlight the centers of the gyral crests. The distance was defined based on the dis placement from the vertex of interest to the boundaries between gyral crest regions and sulcal basins. As a result, a field with decreasing gradient from gyral crest centers to the boundaries was generated. Then a tree marching algorithm was applied to the distance map to connect vertices from crest centers to boundaries. A tree root was placed in each gyral crest center and progressively connected other vertices following the descending gradient of the distance map till the boundaries between gyral crest regions and sulcal basins are reached. During this process, when two trees met, connections will be made between the two trees. By this way, the gyral crests on the same hemisphere can be connected into a graph. The constructed graph structure was shown in Fig. 1a3 by black curves and the zoom in view of a circled area was displayedbetween Fig. 1a2 and Fig. 1a3.

 Extraction of GyralNet and 3HGs 

During the tree marching process, all the vertices in gyral crest regions were connected and some redundant branches were gen erated. We trimmed these redundant branches when their length was shorter than a predefined threshold. The main trunks of the graph structure were preserved, and this trimmed graph named as GyralNet (black curves in Fig. 1a4). The conjunctions with three branches on the GyralNet were defined as 3HGs (green bubbles inFig. 1a5).

 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

Lu Zhang et al. | 3

 Multi-Hop Features Encoding 

By taking 3HGs as nodes, the GyralNet can be represented as an undirected graph. Let G = (V, E) denote the undirected graph, where V = {v1, v2, · · · , vN} is the set of N 3HGs, and E ⊆ {{vi, vj}|vi, vj ∈ V} is the set of unweighted edges, which are the gyral crest lines connecting 3HGs. Its adjacency matrix is denoted by A = [ai,j] ∈ RN×N, where ai,j = 1 if there is a connection between vi and vj, and ai,j = 0 otherwise. We conducted ROI labeling via Destrieux Atlas and divide the whole surface into 75 ROIs. Each 3HG was assigned an ROI label as node feature (Fig. 1b1). We numerically represented the ROI labels by one-hot encoding; i.e. the kth label was denoted by ek ∈ R75 with 1 in the kth location and 0 elsewhere. Accordingly, the ith 3HG in kth ROI can be denoted by xi = ek.By far, the undirected graph of 3HGs can be represented by the adjacency matrix A and the feature matrix X = {x1; x2; · · · ; xN} ∈ RN×75 (Fig. 1b2). Based on the two matrices, the two key components of multi-hop features encoding are defined as: lth hop features: In a 3HG graph, the lth hop neighborhood of the 3HG i is the set of 3HGs connecting to 3HGivia the shortest path with l steps (l-hop), denoted by Nl(i). For vj ∈ Nl(i), its feature vector is denoted by xj, and accordingly the lth hop features of 3HG i are defined as  vj∈Nl(i) xj. Given the adjacency matrix A and the feature matrix X , the lth hop features of 3HG i can be calculated by [AlX ]i,∗, where Al is the lthpower of A. As the adjacency matrix A defines the direct connections between the graph nodes, in the process of recurrently multiplying by itself, just like the graph convolution operation, the undirected connec tions of further neighbors are propagated and gathered along with the direct connections. When multiplying A l times, the features of the neighbors that can reach the center node by l steps are congregated. As each row of AlX corresponds to one 3HG, the lth hop feature of 3HG i thereby can be denoted by [AlX ]i,∗. l hop features (multi-hop features): Based on the definition of lth hop features, we further defined the l hop features of 3HG i as follows (Fig. 1b3): Fi ⎡ ⎢⎢⎢⎢⎢⎢⎢⎣ [X ]i,∗

 AlX 

 ⎤ ⎥⎥⎥⎥⎥⎥⎥⎦ (1) where the 0th hop feature [X ]i,∗ = ek, which indicates that 3HG i is in the ROI k, and the lth hop feature [AlX ]i,∗ = 75 lkek, with the multi-hop coefficient ai lk indicating the number of different l-step paths that are available from 3HG i to the ROI k. If there is no l-step path between them, ai lk = 0, otherwise, ai lk will be a positive integer. In this work, we set l > 1; hence, the l hop features cover multiple hops of the 3HG graph and are also called multi-hop features. By organizing the multi-hop features in this manner, the hierarchical multi-hop relationships between 3HGs with all the 75 ROIs areencoded into FMH.  

---

 ## 翻译[2/12]：
 

# 树形行进

首先对脑回顶部区域进行距离变换，为每个顶点分配距离值并突出脑回中心。该距离是以从感兴趣的顶点到脑回顶部区域和沟回盆地之间边界的位移为基础定义的。因此，生成了一个从脑回中心到边界呈递减梯度的场。然后，应用一种树行进算法来连接从顶部中心到边界的顶点的距离图。在每个脑回中心放置一个树根，并沿着距离图的下降梯度逐步连接其他顶点，直到到达脑回顶部区域和沟回盆地之间的边界。在这个过程中，当两棵树相遇时，将在两棵树之间建立连接。通过这种方式，同一半球的脑回可以连接成一个图形结构。在图1a3中，黑色曲线显示了构建的图形结构，放大视图显示在图1a2和图1a3之间的圆圈区域。

# 脑回网络和3HGs的提取

在树行进过程中，将所有顶点连接到脑回顶部区域，并生成一些冗余分支。当它们的长度小于预定义的阈值时，我们修剪了这些冗余分支。保留图形结构的主干，这个修剪过的图形称为GyralNet（图1a4中的黑色曲线）。与GyralNet上有三个分支的相交点被定义为3HGs（图1a5中的绿色气泡）。

# 多跳特征编码

通过将3HGs作为节点，可以将GyralNet表示为无向图。假设G =（V，E）表示无向图，其中V = {v1，v2，···，vN}是N个3HGs组成的集合， E⊆{{vi，vj}|vi，vj∈ V}是连接3HGs的脑回顶部线条的无权边。其邻接矩阵由A = [ai，j]∈RN×N表示，其中ai，j = 1表示vi和vj之间有连接，ai，j = 0表示没有连接。我们通过Destrieux Atlas进行ROI标注，并将整个表面分为75个ROI。将每个3HG分配一个ROI标签作为节点特征（图1b1）。通过one-hot编码对ROI标签进行数字表示，即第k个标签由ek∈R75表示，其中第k个位置为1，其他位置为0。因此，第k个ROI中的第i个3HG可以表示为xi = ek。到目前为止，3HGs的无向图可以用邻接矩阵A和特征矩阵X = {x1；x2；···；xN}∈RN×75（图1b2）表示。基于这两个矩阵，多跳特征编码的两个关键组成部分定义如下：

第l次跳跃特征：在3HG图中，第l次跳跃邻域与3HG i通过l步最短路径（l-hop）相连的3HGs集合，表示为Nl（i）。对于vj ∈ Nl（i），其特征向量表示为xj，因此第l次跳跃特征的定义为 vj∈Nl（i）xj。给定邻接矩阵A和特征矩阵X，可以通过[AlX]i，*计算第i个3HG的第l次跳跃特征，其中Al是A的第l次幂。因为邻接矩阵A定义了图节点之间的直接连接，在反复相乘的过程中，就像图卷积操作一样，进一步的邻居连接沿着直接连接进行传播和聚集。相乘Al次时，可以汇集可以通过l步到达中心节点的邻居的特征。由于AlX的每一行对应一个3HG，因此第i个3HG的第l次跳跃特征可以表示为[AlX] i，*。

l跳跃特征（多跳距离特征）：根据第l次跳跃特征的定义，我们进一步定义第i个3HG的l跳跃特征如下（图1b3）：Fi ⎡⎢⎢⎢⎢⎢⎢⎢⎣[X]i，* AlX  ⎤ ⎥⎥⎥⎥⎥⎥⎥⎦(1)其中第0个跳跃特征[X]i，* = ek，表示3HG i在ROI k中，第l个跳跃特征[AlX]i，*=75lkek，其中ai lk表示可以从3HG i到ROI k的不同l步路径的数量。如果它们之间没有l步路径，则ailk = 0，否则ailk将是一个正整数。在此工作中，设置l> 1；因此，l跳跃特征覆盖了3HG图的多个跳跃，并且也被称为多跳特征。通过以这种方式组织多跳跃特征，将3HGs与所有75个ROIs之间的分层多跳关系编码到FMH中。

## 

---

 ## 原文[3/12]： 

 
 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

4 | Cerebral Cortex, 2022

 Learning-based Embedding Framework 

Our learning-based embedding framework (Fig. 1c) is designed in a self-supervised manner: it includes two-stage encoding to hierarchically map the input multi-hop features to a latent rep resentation, and a two-stage decoding that aims to hierarchically reconstruct the original input from the latent representation. The embedding learning process can be formulated as (2): Ei MH = σ  Fi (2) Ei F = σ  • Ei  E∼ MH = σ  WD1T • Ei  F∼ E∼ MH • WD2 where σ is the non-linear activation function, Fi is the multi-hop feature of 3HG i defined by (1); WEmbedding = {w1; w2; · · · ; w75} ∈ R75×d is the learnable embedding matrix. In our setting, there are 75 ROIs in total, we initialized a learnable embedding vector w ∈ Rd for each ROI and organized them as the same order as the one-hot encoding in (1) to form WEmbedding. The input multi-hop features are embedded via WEmbedding (hop by hop) to generate multi-hop embeddings Ei MH ∈ R(l+1)×d. To further fuse the multi-hop embeddings into a single embedding vector that contains the complete multi-hop information, we conducted the second encoding by learnable combination parameters WFusion ∈ R(l+1)×1 to integrate the multi-hop embeddings into one embed ding vector Ei F ∈ R1×d. In (2), the transpose of WFusion—(WFusion) was used for matrix multiplication at multi-hop dimension (rowsT, of Ei We used a symmetric design (as traditional autoencoder) for the two-stage decoding with the parameters WD1 ∈ R1×(l+1) and WD2 ∈ Rd×75, respectively. The first decoding reconstructs the hier archical multi-hop embeddings from the combined embedding vector, which ensures that the combined embedding vector Ei has captured the complete information to restore the embeddingsF for each hop. Then, upon the restored multi-hop embeddings Ei the second decoding was applied to recover the multi-hop input features Fi MH. We adopted the MSE loss to evaluate the two-stage decoding and the objective function can be defined by (3): L = α Ei MH −Ei  F + β Fi MH −Fi  (3) where α and β are the hyper-parameters to control the contribu tion of the two-stage decoding. The entire model was trained in a self-supervised manner, which avoids introducing any bias from supervised term when representing cortical folding as embed ding vectors. Through a two-stage encoding-decoding process, the intrinsic patterns buried in the complex and variable cortical folding can be effectively instilled into the embedding vectors.

 Evaluation of the Embedding Effectiveness 

In this work, there are two kinds of embeddings: the ROI embed dings learned by population—WEmbedding and the individual 3HG embeddings—Ei F. The ROI embeddings are learned by recovering the 3HG’s multi-hop features on population level and can be served as the basic elements to represent each 3HG. Therefore, they should have the capability to characterize regularity of the anatomical pattern of 3HGs shared by the population. We adopted the strength of 3HG’s multi-hop connection to describe the 3HGs’ connection patterns between ROI pairs at different hop levels, which is defined as: Strength of 3HG’s multi-hop connection (ground truth matrix): If 3HG i is in ROI k and 3HG j is in ROI m, and they are connected in the GyralNet via the shortest path with l steps, then ROI k and ROI m have one l-hop 3HG connection. In this work, we added up the connections between each pair of ROIs across the whole population and used the resulting matrix as the ground truth(Fig. 2a).

 In the ground truth matrix, if two ROIs have a larger number of l-hop 3HG connections, it means the two ROIs are closely related in GyralNet at the l-hop level. Thus, their l-hop embedding vec tors should capture this close relationship by possessing highersimilarity in the latent space.

 The individual 3HG embedding Ei F is used to represent a unique 3HG. As defined in (1) and (2), each 3HG embedding is a specific combination of ROI embeddings WEmbedding via a set of 3HG specific coefficients {ai lk} and the learned fusion parameters WFusion. An effective embedding vector Ei F is expected to be able to preserve the individuality of different 3HGs and provides reliable cross-subject 3HG anatomical correspondence. To evaluate this capability, we applied the generated 3HG embeddings to the anatomical correspondence task to infer the complicated many to-many cross-subject anatomical correspondence of 3HGs.

 Results   

---

 ## 翻译[3/12]：
 

#由德州大学阿灵顿分校用户于2022年12月10日下载自https：//academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883

4 | 大脑皮层，2022

#基于学习的嵌入框架

我们的基于学习的嵌入框架（图1c）采用自监督方式设计：包括两个阶段的编码，将输入的多跳特征层次化地映射到潜在表征，以及两个阶段的解码，旨在层次化地从潜在表征中重构原始输入。嵌入学习过程可以公式化为（2）：Ei MH = σ Fi（2）Ei F = σ • Ei （2）E∼ MH = σ WD1 T • Ei∼ F∼ E∼ MH • WD2其中，σ为非线性激活函数，Fi是由（1）定义的3HG i的多跳特征；WEmbedding = {w1; w2; · · · ; w75} ∈R75×d是可学习的嵌入矩阵。在我们的设置中，总共有75个ROI，我们为每个ROI初始化了可学习的嵌入向量w ∈ Rd，并按照（1）的独热编码顺序组织它们以形成WEmbedding。输入的多跳特征通过WEmbedding（一跳一跳）嵌入，以生成多跳嵌入Ei MH ∈ R（l + 1）×d。为了进一步将多跳嵌入融合成包含完整多跳信息的单一嵌入向量，我们通过可学习的组合参数WFusion ∈ R（l + 1）进行第二次编码，将多跳嵌入集成到一个嵌入向量Ei F ∈ R1×d中。在（2）中，WFusion的转置（WFusion）用于在多跳维度（Ei的rowsT）进行矩阵乘法。

我们采用对称设计（如传统的自编码器）进行两阶段解码，参数分别为WD1∈R1×（l + 1）和WD2∈Rd×75。第一次解码从组合嵌入向量重构分层多跳嵌入，确保组合嵌入向量Ei已捕获完整信息以恢复每个跳的嵌入F。然后，针对恢复的多跳嵌入ȜEi，应用第二次解码来恢复多跳输入特征ȜFi MH。我们采用MSE损失来评估两阶段解码，并且目标函数可以通过（3）定义：L = αEi MH −ȜEiF + βFi MH −ȜFi，其中α和β是超参数，用于控制两个阶段解码的贡献。整个模型采用自监督方式进行训练，避免在表示皮层折叠为嵌入向量时引入任何有监督的偏差。通过两个阶段的编码-解码过程，可以有效地将埋藏在复杂和可变的皮层折叠中的内在模式灌输到嵌入向量中。

#嵌入有效性的评估

在这项工作中，有两种嵌入：通过人口学习得到的ROI嵌入 - WEmbedding和单个3HG嵌入 - Ei F。ROI嵌入通过恢复人口水平上的3HG的多跳特征来学习，并且可以作为表示每个3HG的基本元素。因此，它们应该具备表征人口共享的3HGs的解剖模式的规律性的能力。

我们采用3HG的多跳连接强度来描述ROI对之间的3HG连接模式在不同跳级别上的情况，并定义为：3HG的多跳连接强度（真实矩阵）：如果3HG i在ROI k中，3HG j在ROI m中，并且它们通过长度为l的最短路径在GyralNet上连接，则ROI k和ROI m具有一个l-hop 3HG连接。在本工作中，我们对整个人口的每对ROI之间的连接进行了求和，并将得到的矩阵用作基础事实（图2a）。

在基本事实矩阵中，如果两个ROI具有更多的l-hop 3HG连接，则表示它们在跳l一级上在GyralNet中密切相关。因此，它们的l-hop嵌入向量应该通过在潜在空间中具有更高的相似性来捕获这种密切关系。

单个3HG嵌入Ei F用于表示独特的3HG。如（1）和（2）所定义，每个3HG嵌入是通过一组3HG特定系数{ailk}和学习的融合参数WFusion通过ROI嵌入WEmbedding的特定组合来获得的。期望有效的嵌入向量Ei F能够保留不同3HGs的个性，并提供可靠的跨主体3HG解剖对应关系。为了评估这种能力，我们采用生成的3HG嵌入向量进行解剖对应任务，以推断3HGs的复杂多对多跨主体解剖对应关系。

#结果

## 

---

 ## 原文[4/12]： 

 
We applied the proposed multi-hop feature encoding method (Section 2.3) and the learning-based embedding framework (Sec tion 2.4) to the identified 3HGs (Section 2.2). By training the model end-to-end in a self-supervised task, we learned a set of ROI embeddings—WEmbedding. The effectiveness of WEmbedding was evaluated by the strength of 3HG’s multi-hop connection (Section 2.5). Then we generalized the learned ROI embeddings and the well-trained model to a new dataset and generated an individual embedding vector for each 3HG. The effectiveness of the generated individual embedding vectors was evaluated in the anatomical correspondence task to infer the complicated cross subject anatomical correspondence of 3HGs. The result section is organized as follows: Section 3.1 introduces the experimental setting; Section 3.2 evaluates the effectiveness of the learned ROI embeddings; Section 3.3 and 3.4 show the inference results of the 3HGs’ anatomical correspondence task; Section 3.5 assesses the regression performance of the proposed two-stage decodingframework.

 Experimental Setting 

Data Setting. We randomly divided the 1064 subjects from HCP dataset into two datasets. Dataset-1 is used to train the model and learn the embeddings. Then the well-trained model and the learned ROI embeddings are applied to dataset-2 to infer the 3HGs’ anatomical correspondence. In dataset-1, there are 564 subjects, and 186,915 3HGs are identified. In dataset-2, there are 500 subjects and 169,923 3HGs are identified. Each 3HG is treatedas a data sample.

 Model Setting. For multi-hop features, we generated 1-hop features (l = 1 in (1) and (2)), 2-hop features (l = 2) and 3-hop features (l = 3). For each kind of feature, we trained the model to learn the corresponding ROI embeddings. In our experiments, the learnable ROI embeddings were initialized by identity matrix to ensure the initial distances between any two embedding vectors are the same. We adopted the embedding dimension d = 128. The fusion operation—WFusion, and the two decoder operations—WD1 and

 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

Lu Zhang et al. | 5 WD2 were implemented by fully connected layers and the param eters were initialized following the Xavier scheme. The entire model was trained in an end-to-end manner. The Adam optimizer was used to train the whole model with standard learning rate 0.001, weight decay 0.01, and momentum rates (0.9, 0.999).

 Effectiveness of ROI Embeddings 

In the experiments, we used dataset-1 to generate different multi hop features including 1-hop features, 2-hop features, and 3-hop features to train the model and learn the ROI embeddings. We evaluated the learned ROI embeddings via the strength of 3HG’s multi-hop connection (Section 2.5) and displayed the results in Fig. 2. Figure 2(a) shows the statistical results of the strength of 3HG’s multi-hop connections based on the whole population of 1064 subjects. Figure 2(b) shows the cosine similarity between the learned ROI embedding vectors. For each matrix, the order of the brain regions is the same as the order defined in Destrieux atlas (Fischl et al. 1999), where most of the first 44 regions are gyri and the last 31 regions are sulci. For the ease of better analyzing the results, we divided all the 3HG’s multi-hop connections into 3 dif ferent groups: gyri-gyri connections, gyri-sulci connections, and sulci-sulci connections. The gyri-gyri connections mean both two connecting 3HGs are in gyral regions. The gyri-gyri connections are located at the top left of each matrix in Fig. 2. The gyri-sulci connections and sulci-sulci connections are defined in the same manner. In the matrices, the gyri-sulci connections are located at top right and bottom left, sulci-sulci connections are located at the bottom right. It is worth noting that the ground truth matrix (Fig. 2(a)) reflects the strength of actual anatomical relationship between two ROIs on cortical surface, e.g. the number of GyralNet edges between these two ROIs. While the embedding similarity matrix is defined based on cosine similarity between the embed ding vectors of these two ROIs, which represents the relationship of these two ROIs in the latent (embedding) space. Therefore, if the two matrices have similar patterns, we can conclude that our ROI embeddings can effectively represent anatomical ROIs in the latent space including their relations. In Fig. 2, we can see that when only considering 1-hop features (the first column), the ground truth matrix shows strong gyri-gyri connections, weak gyri-sulci connections and almost no sulci-sulci connections. Our embeddings effectively capture this pattern, though some weak connections in the ground truth matrix are missing due to the sparsity of the input 1-hop features. When using 2-hop and 3 hop features (the second and third columns), the ground truth matrices show stronger connections in some regions (highlighted by red squares), and the same pattern can also be found in our embedding similarity matrices. Besides the visualization, we also adopted three measures, including structural similarity index measure (SSIM), Pearson correlation coefficient (PCC), and cosine similarity (CS), to quantitatively measure the similarity between the ground truth matrices and our embedding similarity matrices. The results are reported in Table 1. For all the three measures,  

---

 ## 翻译[4/12]：
 

我们采用了提出的多跳特征编码方法（第2.3节）和基于学习的嵌入框架（第2.4节）来处理已识别的3HGs（第2.2节）。通过自监督任务的端到端训练，我们学习了一组ROI嵌入——WEmbedding。WEmbedding的有效性是通过3HG多跳连接（第2.5节）的强度来评估的。然后，我们将学习到的ROI嵌入和经过良好训练的模型推广到新数据集，并为每个3HG生成一个单独的嵌入向量。生成的单个嵌入向量的有效性是通过解决解剖对应任务以推断3HGs的复杂跨受试者解剖对应来评估的。结果部分按以下方式组织：第3.1节介绍了实验设置；第3.2节评估了学到的ROI嵌入的有效性；第3.3和3.4节展示了3HGs解剖对应任务的推理结果；第3.5节评估了提出的两阶段解码框架的回归性能。

# 实验设置

数据设置。我们将HCP数据集中的1064个受试者随机分为两个数据集。数据集1用于训练模型和学习嵌入。然后将经过良好训练的模型和学习到的ROI嵌入应用于数据集2以推断3HGs的解剖对应关系。在数据集1中，有564个受试者，已识别出了186,915个3HGs。在数据集2中，有500个受试者，已识别出了169,923个3HGs。每个3HG都被视为一个数据样本。

模型设置。对于多跳特征，我们生成了1跳特征（l=1于(1)和(2)中）、2跳特征（l = 2）和3跳特征（l = 3）。对于每种特征，我们训练模型学习相应的ROI嵌入。在我们的实验中，可学习的ROI嵌入由身份矩阵初始化，以确保任意两个嵌入向量之间的初始距离相同。我们采用嵌入维度d = 128。融合操作WFusion和两个解码器操作WD1和WD2由完全连接层实现，参数采用Xavier方案进行初始化。整个模型都是以端到端的方式进行训练的。使用Adam优化器以标准学习率0.001、权重衰减0.01和动量率（0.9、0.999）来训练整个模型。

# ROI嵌入的有效性

在实验中，我们使用数据集1生成不同的多跳特征，包括1跳特征、2跳特征和3跳特征来训练模型和学习ROI嵌入。我们通过3HG的多跳连接的强度来评估学习到的ROI嵌入（第2.5节），并在图2中显示了结果。图2（a）显示了基于1064个受试者总体的3HG多跳连接强度的统计结果。图2（b）显示了学习到的ROI嵌入向量之间的余弦相似度。在每个矩阵中，脑区的顺序与Destrieux图谱（Fischl等人，1999年）中定义的顺序相同，其中大多数的前44个区域是回旋区，后面31个是小沟区。为了更好地分析结果，我们将所有的3HG的多跳连接分成3个不同的组：回旋区-回旋区连接、回旋区-小沟区连接和小沟区-小沟区连接。回旋区-回旋区连接意味着两个连接的3HGs都在回旋区。回旋区-回旋区连接在图2的每个矩阵的左上方。回旋区-小沟区连接和小沟区-小沟区连接以同样的方式定义。在矩阵中，回旋区-小沟区连接位于右上方和左下方，小沟区-小沟区连接位于右下方。值得注意的是，基础真值矩阵（图2（a））反映了大脑皮层上两个ROI之间实际解剖关系的强度，例如这两个ROI之间的回旋网络边的数量。虽然嵌入相似性矩阵是基于这两个ROI的嵌入向量之间的余弦相似度定义的，但它代表了这两个ROI在潜在（嵌入）空间中的关系。因此，如果两个矩阵具有类似的模式，我们可以得出结论，我们的ROI嵌入可以有效地在潜在空间中表示解剖ROI及其关系。从图2中，当仅考虑1跳特征（第一列）时，基础真值矩阵显示出强烈的回旋区-回旋区连接，较弱的回旋区-小沟区连接和几乎没有小沟区-小沟区连接。虽然一些基础真值矩阵中的弱连接由于输入1跳特征的稀疏性而缺失，但我们的嵌入有效地捕捉了这种模式。当使用2跳和3跳特征（第二和第三列）时，基础真值矩阵在某些区域显示出更强的连接（用红色方框突出显示），我们的嵌入相似度矩阵中也可以找到相同的模式。除了可视化外，我们还采用了三个度量标准，包括结构相似性指数度量（SSIM）、Pearson相关系数（PCC）和余弦相似度（CS）来定量测量基础真值矩阵和我们的嵌入相似度矩阵之间的相似度。结果在表1中报告。所有三个度量标准的结果表明，我们的ROI嵌入与基础真值矩阵之间具有相似的模式。

## 

---

 ## 原文[5/12]： 

 
 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

6 | Cerebral Cortex, 2022 Table 1. Similarity between ground truth matrix andembedding similarity matrix.

 the learned embedding similarity matrices show highly consis tent pattern to the ground truth matrices, especially for the 2-hop embedding whose SSIM measure is over 0.8. These results demonstrate that our learned embeddings can effectively encode the population-level common connection patterns in the data: if two ROIs have strong/weak connections on GyralNet (cortical space), they will also have large/small similarities in the latentspace (embedding space).

 In addition to evaluating the learned embeddings by the overall pattern, we also assessed it at a finer scale by selecting the top 9 connections (pair of ROIs) in the three ground truth matrices and three embedding similarity matrices in Fig. 2. The results are shown in Fig. 3. Most of the top-9 connections in the ground truth matrices can be found in the embedding similarity matrices (missed 1 in hop-1 embedding, missed 2 in hop-2 embedding and missed 3 in hop-3 embedding). In addition, we found that all the missing connections (top-9 connections in ground truth but not in embedding similarity matrices) or redundant connections (top 9 connections in embedding similarity matrices but not in the ground truth) can be found in the top 40 connections in embed ding similarity matrices and the ground truth matrices. In general, the learned embedding vectors can reliably represent 3HGs and their connections in a latent space, which can be used to construct their similarities and correspondences across individuals.

 Effectiveness of 3HGs Individual Embeddings 

After the model is well trained by dataset-1, we applied the learned ROI embeddings and the model to the new dataset 2 to generate individual 3HG embeddings—EiF (defined in (2)).

 According to the discussion in Section 3.2, the 2-hop features can provide the best embedding performance; hence, we adopted 2 hop embedding in this section. As discussed in Section 2.5, the individual 3HG embeddings are expected to be able to preserve the individuality of 3HGs and infer reliable cross-subject anatomical correspondence. Therefore, in this section we will evaluate the effectiveness of the individual embeddings from two aspects.

 Inferring Reliable 3HGs Cross-Subject Anatomical Correspondence 

To evaluate the effectiveness of 3HG individual embeddings on inferring cross-subject anatomical correspondence, we used the 3HGs from one randomly selected subject (sub-0 in Fig. 4) as the exemplars to find their corresponding 3HGs in other subjects by the learned embedding vector—Ei F. For each exemplar 3HG in sub 0, the correspondence inferring process is based on the following steps: (i) we examined all the 3HGs in different subjects and calculated the cosine similarity between the embedding vector of exemplar 3HG with each of the other 3HGs; (ii) for each subject, the 3HGs that have the cosine similarity of 1.0 to the exemplar 3HG will be identified as the corresponding 3HGs in this subject; (iii) if there is no 3HGs having the cosine similarity of 1.0, the one with the largest cosine similarity (above a threshold) will be identified as the corresponding 3HG. Following these steps, we obtained the corresponding 3HGs for each of the exemplar 3HGs in different subjects. In the sub-0, 190 and 175 3HGs have been identified in the left and right hemisphere by the 3HGs identifi cation pipeline (Section 2.2), respectively. For better visualization, we selected 60 3HGs on each hemisphere, which spread over the whole cerebral cortex and showed the corresponding 3HGs on 10 randomly selected subjects in Fig. 4. Bubbles indicate the locations of 3HGs. The corresponding 3HGs in different subjects were color-coded by correspondence indices. From the results we can see that the corresponding 3HGs identified on different indi viduals have consistent locations in terms of common anatomical landscapes: for example, 3HG 157 in left hemisphere and 3HG 96 in right hemisphere (marked by red arrows) are found in the middle of left front superior gyri and middle of the right precentralgyri, respectively, across all the subjects.

 Preserving Cross-subject Individuality   

---

 ## 翻译[5/12]：
 

本文下载自https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883，由得克萨斯大学阿灵顿分校的用户于2022年12月10日下载。

表1显示了基准矩阵和嵌入相似度矩阵之间的相似性。

学习到的嵌入相似度矩阵表现出高度一致的模式，特别是2-跳嵌入的SSIM（结构相似性）指标超过0.8，这些结果证明，我们学习到的嵌入能够有效地编码数据中的种群水平的共同连接模式：如果两个感兴趣的区域（ROI）在GyralNet（皮层空间）上具有强/弱连接，它们在潜在空间（嵌入空间）中也将具有较大/较小的相似性。

除了通过总体模式评估学习到的嵌入之外，我们还通过选择图2中三个基准矩阵和三个嵌入相似性矩阵中的前9个连接（ROI对）在更细的尺度上对其进行了评估。结果如图3所示。大多数基准矩阵中的前9个连接可以在嵌入相似度矩阵中找到（1个在1-跳嵌入中, 2个在2-跳嵌入中，3个在3-跳嵌入中未找到）。此外，我们发现，在嵌入相似度矩阵和基准矩阵的前40个连接中都可以找到所有缺失连接（出现在基准矩阵中但未出现在嵌入相似度矩阵中的前9个连接）或多余连接（出现在嵌入相似度矩阵中但未出现在基准矩阵中的前9个连接）。总之，学习的嵌入向量可以可靠地表示潜在空间中的3HGs及其连接，这可以用于构建它们在个体之间的相似性和对应关系。

# 3HGs个体嵌入的有效性

在通过数据集1训练好模型后，我们将学习到的ROI嵌入和模型应用于新的数据集2，生成个体3HG嵌入-Ei F（定义在（2）中）。

根据第3.2节的讨论，2-跳特征可以提供最佳的嵌入表现；因此，在本节中我们采用2-跳嵌入。如第2.5节所述，个体3HG嵌入被期望能够保留3HGs的个性并推断出可靠的跨学科解剖学对应关系。因此，在本节中，我们将从两个方面评估个体嵌入的有效性。

# 推断可靠的3HGs跨学科解剖学对应关系

为评估个体3HG嵌入在推断跨学科解剖学对应关系方面的有效性，我们使用来自一个随机选择的个体（图4中的sub-0）的3HGs作为例子，通过学习到的嵌入向量-Ei F，在其他个体中找到它们相应的3HGs。对于每个sub-0中的例子3HG，对应关系推断过程基于以下步骤：（i）我们检查不同个体中的所有3HGs，并计算例子3HG的嵌入向量与每个其他3HG之间的余弦相似度；（ii）对于每个个体，余弦相似度为1.0且与例子3HG相似的3HGs将被识别为该个体中相应的3HGs；（iii）如果没有余弦相似度为1.0的3HGs，在余弦相似度大于阈值的3HGs中，余弦相似度最大的将被识别为相应的3HG。遵循这些步骤，我们在不同个体中获得了每个例子3HG的相应3HGs。在sub-0中，3HGs识别流程（第2.2节）在左半球和右半球共识别出了190个和175个3HGs。为了更好地可视化，我们在每个半球上选择了60个3HGs，它们分布在整个大脑皮层上，并在图4中随机选取的10个个体中显示了相应的3HG。气泡表示3HGs的位置。不同个体中的相应3HGs由对应指数进行颜色编码。从结果中可以看出，在不同个体中识别出的相应3HGs在常见解剖景观方面具有一致的位置，例如，左半球的3HG #157和右半球的3HG #96（由红色箭头标记）均在所有个体中位于左侧前上额回的中央和右侧前中心回的中央。

# 保留跨个体个性

## 

---

 ## 原文[6/12]： 

 
As an essential characteristic of human cerebral cortex, the folds of different subjects have shown intensive variability. To illus trate that the learned 3HG individual embeddings can preserve individuality of different subjects, we randomly selected three 3HGs as exemplars to find their corresponding 3HGs in different subjects. That is, within each subject, all the 3HGs that have a cosine similarity over 0.9 to the exemplar 3HG will be identified as the corresponding 3HG of that exemplar. We randomly selected 12 subjects and showed the results in Fig. 5. The cerebral cortex of the 12 subjects displays distinct cortical folding patterns. For example, the first exemplar 3HG is located at the conjunction of precentral gyri and front middle gyri, and there is no other 3HGs in the neighborhood. However, the folding patterns of the fifth subject (highlighted in yellow) is more convoluted, as a result, multiple 3HGs are aggregated at the same location. For the 11th subject (highlighted in pink), there is no conjunction that can connect precentral gyri and front middle gyri, therefore, there is no 3HGs located at this location. More examples have been found and marked in exemplar 2 and 3. Despite the widely existed individuality of cortical folding patterns, our embedding method can provide a reliable way to identify the complex many to-many correspondences without mapping different individual brains to the same space, and thus, the variabilities of 3HGs can be preserved. Notably, the embeddings and models used in this correspondence task were trained in a different dataset—dataset 1 by a self-supervised regression task, but they can generalize well on the new dataset-2 that shows promising transferable capability in other datasets. Therefore, the proposed framework can provide an effective way to design practical pre-training paradigms and facilitate downstream tasks in brain anatomy studies.

 Commonality of 3HGs Correspondence 

In this section, we studied the commonality of 3HGs. For each 3HG, the commonality is defined as the percentage of the sub jects in which the corresponding 3HGs can be found. If a 3HG has higher commonality, that represents its corresponding 3HGs exist on a large population. Similarly, higher individuality (less commonality) means the 3HG can only be found in a small number of individuals. We used all the 169,923 3HGs in dataset 2 (testing dataset) to calculate the commonality. The calculation pipeline is as follows: (i) Taking the 169,923 3HG embeddings as input data, we adopted agglomerative hierarchical clustering algorithms (Murtagh and Contreras 2012) to conduct clustering. Agglomerative hierarchical clustering algorithms is a bottom-up approach that starts with many small clusters (each cluster is one

 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

Lu Zhang et al. | 7 data point at the beginning) and merges the closest ones gradually to create larger clusters, until the shortest distance between clus ters is beyond a predefine distance threshold. In this work, we took cosine similarity as the metric to calculate the distance between data points. Complete linkage was used as the linkage criterion to define the distance between two clusters: the longest distance between two points in the two clusters. Complete linkage can make sure that the distance between any two data points in the same cluster will smaller than the predefined distance threshold. (ii) After the clustering process, the 3HGs in the same cluster are treated as the corresponding 3HGs in different individuals. (iii) For each cluster, representing a distinct 3HG, we examined how many subjects on which we can find the correspondences. Thus, we obtained the percentage of the subjects (to the entire testing dataset) that have this specific 3HG. Note that the derived commonality may be affected by the threshold of cosine similarity used, therefore, we evaluated different cosine similarity values as the predefined distance to conduct clustering and reported the results in Table 2. We listed the number of common 3HGs (clusters) that can be found when using different thresholds and commonality levels. For example, when we set the cosine similarity threshold as 0.85, 49 common 3HGs (clusters) can be identified on 90% of the subjects in the testing dataset. From the results we can see that when the cosine similarity decreases from 1.0 to 0.9, the number of 3HGs at all commonality levels (from 40 to 90%) increases. When the cosine similarity further decreases (from 0.9 to 0.85), the number of common 3HGs keeps relatively stable. This situation suggests that when the cosine similarity threshold decreases below 0.9, the new common 3HGs identifiedincline to join the existing clusters.

 We further visualized the distribution of all 3HGs as well as the common 3HGs across brain regions and presented the results in Fig. 6. The distribution of all 3HGs was calculated by adding the number of 3HGs across populations for each region and then dividing by the number of individuals (Fig. 6(a)). For common 3HGs distribution, we presented the results of five commonality levels ranging from 50% to 90% with cosine similarity equal to 0.9 (Fig. 6(b)-(f)), which correspond to the results highlighted by the

 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

8 | Cerebral Cortex, 2022  

---

 ## 翻译[6/12]：
 

作为人类大脑皮层的基本特征，不同受试者之间的褶皱显示出强烈的可变性。为了说明学习到的三角测量矩阵（3HG）个体嵌入能够保留不同主体的个体性，我们随机选择了三个作为样本，以找到不同受试者中相应的3HGs。也就是说，在每个受试者内，在与样本3HG余弦相似度超过0.9的所有3HGs将被标识为该示例器的相应3HG。我们随机选择了12个受试者，并将其结果呈现在图5中。12个对象的大脑皮层显示出明显的皮层折叠模式。例如，第一个样本3HG位于前中回和中央前回的交会处，其邻近没有其他的3HGs。然而，第五个受试者（突出显示为黄色）的折叠模式更加复杂，因此多个3HGs汇聚在同一位置。对于第11个受试者（突出显示为粉色），没有交会可以连接前中回和中央前回，因此在此位置上没有3HGs。更多示例在样本2和3中标注。尽管皮层折叠模式存在广泛的个体差异，我们的嵌入方法可以提供一种可靠的方法，在不将不同个体的大脑映射到同一个空间的情况下识别复杂的多对多对应关系，从而可以保留3HGs的变异性。值得注意的是，此对应任务中使用的嵌入和模型是在不同的数据集（数据集1）上通过自监督回归任务训练的，但它们可以在新的数据集2上广泛适用，显示出良好的可转移能力。因此，提出的框架可以提供一种有效的方法来设计实际的预训练范式，并促进脑解剖研究中的下游任务。

在本节中，我们研究了3HGs的共性。对于每个3HG，共性被定义为在相应的主体中可以找到对应3HGs的百分比。如果3HG的共性更高，表示其相应的3HGs存在于较大的人口中。同样，更高的个体性（较低的共性）意味着只能在少数人中找到3HG。我们使用数据集2中的全部169,923个3HGs计算共性。计算流程如下：（1）将169,923个3HG嵌入作为输入数据，我们采用凝聚层次聚类算法（Murtagh和Contreras 2012）进行聚类。凝聚层次聚类算法是一种自下而上的方法，从许多小聚类（每个聚类都是一个数据点）开始，并逐渐地将最接近的聚类合并为更大的聚类，直到聚类之间的最短距离超过预定义距离阈值为止。在这项工作中，我们采用余弦相似度作为指标来计算数据点之间的距离。完全连接作为联动标准来定义两个聚类之间的距离：两个聚类中任意两个点之间的最远距离。完整连接可以确保在同一聚类中的任意两个数据点之间的距离小于预定义的距离阈值。（2）在聚类过程之后，将同一聚类中的3HGs视为不同个体中的相应3HGs。（3）对于每个聚类，表示不同的3HG，我们检查有多少个主体可以找到对应关系。因此，我们得到了每个3HG在整个测试数据集中有多少个主体（百分比）。注意，衍生的共性可能会受到所使用的余弦相似度阈值的影响，因此，我们评估了不同的余弦相似度值作为预定义的距离阈值进行聚类，并在表2中报告了结果。我们列出了当使用不同阈值和共性水平时，可以找到多少常见的3HGs（聚类）。例如，当我们将余弦相似度阈值设置为0.85时，在测试数据集的90%上可以识别出49个常见的3HGs（聚类）。从结果可见，当余弦相似度从1.0降至0.9时，所有共性水平（从40%到90%）的3HGs数量都会增加。当余弦相似度进一步降低（从0.9到0.85），常见3HGs的数量保持相对稳定。这种情况表明，当余弦相似度阈值降低到0.9以下时，新常见的3HGs倾向于加入现有的聚类。

我们进一步可视化了所有3HGs以及各个脑区的常见3HGs分布，并在图6中呈现了结果。所有3HGs的分布是通过计算每个区域中3HGs的人口数量并除以个体数量来计算的（图6(a)）。对于常见的3HGs分布，我们呈现了五个共性水平，分别从50%到90%，余弦相似度为0.9（图6(b)-(f)），这对应于本文中突出显示的结果。

## 

---

 ## 原文[7/12]： 

 
 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

Lu Zhang et al. | 9 Table 2. Commonality of 3HGs. red borders in Table 2. The distribution of common 3HGs under each commonality level was quantified by the number of com mon 3HGs of each brain region. To visualize the distribution, we color coded the 3HGs number and mapped it back to the cortical surface in Fig. 6. As shown in Fig. 6(a), 3HGs are found in all gyri regions and are more abundant in certain brain regions, such as G_front_sup (mean = 21), G_front_middle (mean = 10), G_orbital (mean = 8), and G_pariet_inf-Angular (mean = 7). The same pat tern is also found in the distributions of common 3HGs. Taking commonality ≥0.5 as an example, although 130 common 3HGs are spreading over all the gyri regions, the number of common 3HGs in different regions varies greatly. G_pariet_inf-Angular, for example, has eight common 3HGs, whereas G_occipital_sup hasonly one.

 Regression Performance 

The proposed framework was trained through a self-supervised regression task in a hierarchical two-stage decoding manner. The first stage (Stage-1) reconstructs the hierarchical multi-hop embeddings from the combined embedding vector, whereas the second stage (Stage-2) recovers the multi-hop features from the hierarchical multi-hop embeddings. In this section, we used four metrics to evaluate the regression performance of the two decod ing stages from various perspectives, including mean absolute error (MAE) and mean squared error (MSE) for magnitude, and Structural Similarity Index Measure (SSIM) and cosine similarity (CS) for overall pattern. In addition, we evaluated the regression performance of both the multi-hop embeddings/features and the embedding/feature vector of each single hop. The results were calculated using 169,923 3HGs in dataset-2 (independent testing dataset) and reported in Table 3. Our results show that the reconstructed embeddings/features in the first/second decoding stage have low MAE and MSE (<0.1/0.15) and high SSIM and CS (>0.9/0.6), indicating that the two-stage decoding framework performs well in the regression task. Furthermore, it is worth noting that the performance of stage-1 is slightly better than stage-2, with lower MAE and MSE, and higher SSIM and CS. This may be because the ground truths in stage-2 are highly sparse matrices—the multi-hop features created by one-hot encoding, whereas the ground truths in stage-1 are dense embedding

 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

10 | Cerebral Cortex, 2022 Table 3. Regression performance of 2-hop embedding. vectors. In the discussion section, we will compare the one-hot encoding and the learnable embedding vectors further.

 Discussion 

Self-supervised embedding. A common problem in deep learning models is limited samples: the huge architectures usually demand hundreds of millions of labeled data, which are often publicly inaccessible. Especially in neural imaging domains, where designing these labeled data can be a time-consuming and expensive process and impossible in some scenarios. In natural language processing (NLP), this appetite for data has been suc cessfully addressed by self-supervised pretraining, which enables training of generalizable NLP framework containing over one hundred billion parameters, such as BERT (Devlin et al. 2018) and GPT (Radford et al. 2018, 2019; Brown et al. 2020). Inspired by these successful models in NLP, in this study, we adopted an autoen coder architecture—a simple self-supervised method, to design the learning-based embedding framework. In our experiments, the proposed framework generalizes well on new datasets and shows promising transferable capability in downstream tasks. Disentangling the commonality and individuality. Our pro posed embedding framework was trained to indirectly encode brain anatomy using folding pattern derived landmarks—3HGs. Different from NLP methods in which the words are well defined, and the language vocabulary can be easily built up, there is no pre-exist “vocabulary” when representing cerebral cortex, since each brain has unique folding patterns. To solve this problem, we designed a new embedding strategy to disentangle the com monality and individuality of the 3HGs: instead of embedding the 3HG itself, we embedded the ROIs (from brain atlas) into a set of ROI embedding vectors, serving as basic blocks for representing commonality, and then used these ROI embedding vectors todistill individuality.  

---

 ## 翻译[7/12]：
 

本文摘自https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883，使用者为University of Texas at Arlington，时间为2022年12月10日。

张路等人| 表2.3HGs的共性。表2中的红色边框。每个共性水平下共同3HGs的分布由每个脑区的共同3HGs数量量化。为了可视化分布，我们将3HGs的数量进行了颜色编码，并将其映射回图6中的皮层表面。如图6(a)所示，3HGs存在于所有回脑区域，并且在某些脑区域中更加丰富，例如G_front_sup（平均= 21），G_front_middle（平均= 10），G_orbital（平均= 8）和G_pariet_inf-Angular（平均= 7）。相同的模式也在共同3HGs的分布中发现。以共性≥0.5为例，尽管130种共同的3HGs分布在所有回脑区域，但不同区域的共同3HGs数量差异很大。例如，G_pariet_inf-Angular有8个共同的3HGs，而G_occipital_sup只有1个。

回归性能。所提出的框架通过分层两阶段解码方式进行自监督回归任务的训练。第一阶段（阶段1）从组合嵌入向量中重构出分层多跳嵌入，而第二阶段（阶段2）从分层多跳嵌入中恢复出多跳特征。在本节中，我们使用四个指标从不同角度评估了两个解码阶段的回归性能，包括用于量级的平均绝对误差（MAE）和平均平方误差（MSE），以及用于总体模式的结构相似性指数度量（SSIM）和余弦相似性（CS）。此外，我们还评估了多跳嵌入/特征和每个单跳嵌入/特征向量的回归性能。结果是使用独立测试数据集中的169,923个3HGs计算得出的，并在表3中报告。我们的结果表明，第一/二解码阶段中的重构嵌入/特征具有低MAE和MSE（<0.1/0.15）和高SSIM和CS（>0.9/0.6），表明两阶段解码框架在回归任务中表现良好。此外值得注意的是，第1阶段的性能略优于第2阶段，具有较低的MAE和MSE，以及更高的SSIM和CS。这可能是因为第二阶段中的ground truths是高度稀疏的矩阵 - 单热编码创建的多跳特征，而第一阶段中的ground truths是密集的嵌入向量。在讨论部分，我们将进一步比较单热编码和可学习嵌入向量。

自监督嵌入。深度学习模型中的一个常见问题是样本有限：庞大的体系结构通常需要数亿个标记数据，这些数据通常是公共不可访问的。尤其是在神经成像领域，设计这些标记数据可能是一个耗时且昂贵的过程，在某些情况下不可能。在自然语言处理（NLP）中，自监督预训练成功解决了数据需求的问题。例如，BERT（Devlin等，2018年）和GPT（Radford等，2018年，2019年；Brown等，2020年）等超过一百亿参数的通用NLP框架。受到NLP中这些成功模型的启发，我们在本研究中采用了自动编码器架构 - 一种简单的自监督方法，设计了基于学习的嵌入框架。在我们的实验中，所提出的框架在新数据集上具有良好的推广性，并显示出有前途的下游任务的可传递能力。分离通用性和个性。我们提出的嵌入框架被训练成使用基于折叠模式的地标（3HGs）间接编码脑部解剖结构。与NLP方法不同，其中单词定义良好，语言词汇可以很容易地建立起来，当表示大脑皮层时，没有预定义的“词汇表”，因为每个大脑都具有独特的折叠模式。为了解决这个问题，我们设计了一种新的嵌入策略，以分离3HGs的通用性和个性：我们不是嵌入3HG本身，而是将ROIs（来自脑图谱）嵌入一组ROI嵌入向量中，这些向量作为表示通用性的基本单元，然后使用这些ROI嵌入向量来提炼个性。

## 

---

 ## 原文[8/12]： 

 
 Integrating multi-modal data. In this work, we limited our interest in the folding patterns of 3HGs and focused on the effec tiveness of the proposed method on anatomical correspondences. We did not include the white matter structure into our study. Although it has been widely reported that brain folding patterns are closely related to brain structural connectivity patterns (Van Essen 1997; Thompson et al. 2004; Hilgetag and Barbas 2006; Fischl et al. 2008; Honey et al. 2010; Xu et al. 2010; Nie et al. 2012; Budde and Annese 2013; Reveley et al. 2015), there is still no consensus about the relationship between them. For example, some studies suggested that the tension on the axon pulls the cortex closer and forms the gyri (Van Essen 1997; Hilgetag and Barbas 2006), while in other works gyri were reported to be connected by axons with greater density than sulci at different scales (Xu et al. 2010; Nie et al. 2012; Budde and Annese 2013). There are also some studies suggesting that there exists a superficial axonal system

 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

Lu Zhang et al. | 11 at the boarder of white matters and gray matters, which could impede the detection of axonal connections, especially in sulci regions (Reveley et al. 2015). In addition, the disease related alter ations of white matter structures make the situation even more complicated (Stewart et al. 1975; Landrieu et al. 1998; Sallet et al. 2003; Hardan et al. 2004; Harris et al. 2004; Thompson et al. 2004; Tortori-Donati et al. 2005; Nordahl et al. 2007; Pang et al. 2008; Bullmore and Sporns 2009; Barkovich 2010; Honey et al. 2010; Manzini and Walsh 2011; Wang et al. 2019; Wang et al. 2020; Zhang et al. 2019, 2020a; Yu et al. 2021). However, it is undeniable that white matter plays an important role in the formation of folding patterns,and we intend to include it in our future studies: (i) Inves tigating whether the corresponding 3HGs with similar anatom ical characteristics also have similar fiber connection patterns; (ii) Identifying a group of 3HGs with similar anatomical features and fiber connection patterns and investigating their functional homogeneity/heterogeneity; (iii) Incorporating fiber connection patterns and function into current frameworks to establish a more comprehensive 3HGs map with superb functional homo geneity and intrinsically established cross-subject cortical corre spondences. Then, based on this map of 3HGs, we can radiate the research scope to larger area and to include more landscapes,such as 2-hinges.

 One-hot encoding vs learned embeddings. The ROI features of 3HGs were initially represented by one-hot encodings and used as input features to learn anatomical meaningful embeddings. Compared to the learned embeddings, one-hot vectors cannot be directly used as an embedding vector to infer cross-subject correspondence for two reasons: (i) One-hot vectors are anatom ical meaningless and cannot provide reliable cross-subject 3HGs correspondences. Each one-hot vector contains a single one and N-1 zeros where N is the number of dimensions. As a result, ROIs are embedded in isolation and are equal distance apart, making it impossible to represent the underlying relationships between ROIs. In this way, the similarity of two 3HGs based on one-hot encoding is only related to the number of common ROIs shared by their multi-hop features, while the underlying connections between ROIs are ignored, rendering it powerless when inferring correspondences of 3HGs. For example, under one-hot encoding, the anatomical similarities of all 3HG pairs with no common ROIs of their multi-hop features are zero. In contrast, the learned embeddings can effectively encode the population-level com mon connection patterns between ROIs, with closely connected ROIs having similar embedding vectors, and thus can provide more reliable anatomical correspondences; (ii) One-hot vectors are sparse and grow with vocabulary size, which can easily lead to the curse of dimensionality, whereas embeddings are dense and low-dimensional, making them more computationally efficient. In general, the learned embeddings are more efficient than the one-hot vectors in both anatomical correspondence reliability andcomputational efficiency.

 Conclusion 

In this work, we proposed a learning-based embedding framework to embed the anatomically meaningful patterns of 3HGs, into a group of learnable embedding vectors. Each 3HG can be repre sented as a hierarchical combination of the learned embedding vectors via a set of multi-hop combination coefficients. By this way, the regularity of folding pattern is encoded into the embed ding vectors, and the variability is preserved by the individual specific coefficients. We evaluated the proposed method using HCP dataset and the experiment results show that the learned embeddings can successfully encode the cortical folding patterns and reliably infer the cross-subject complex many-to-many cor-respondences of 3HGs.

 Funding 

This work was supported by STARTs from UT system and par tially supported by National Institutes of Health (R01AG075582, RF1NS128534) and National Science Foundation (IIS-2011369). Conflict of interest statement: The authors declare that no com mercial or financial relationships that could be construed as a potential conflict of interest existed during the research.

 References 

Barkovich AJ. Current concepts of polymicrogyria. Neuroradiology.2010:52:479–487.  

---

 ## 翻译[8/12]：
 

集成多模态数据。本文中，我们限制了对3HGs折叠图案的兴趣，并重点关注所提出的方法对解剖学对应的有效性。我们并未将白质结构纳入我们的研究范畴。尽管大脑折叠图案与大脑结构连接图案密切相关为人们所广泛报道（Van Essen 1997; Thompson et al. 2004; Hilgetag and Barbas 2006; Fischl et al. 2008; Honey et al. 2010; Xu et al. 2010; Nie et al. 2012; Budde and Annese 2013; Reveley et al. 2015），但它们之间的关系还没有达成共识。例如，一些研究认为轴突张力会将皮质拉近并形成大脑回（Van Essen 1997; Hilgetag and Barbas 2006），而在其他研究中，据报道大脑回通过轴突在不同尺度上与沟连接，其密度高于沟（Xu et al. 2010; Nie et al.2012; Budde and Annese 2013）。还有一些研究表明，存在一个在白质和灰质之间的浅表轴突系统，这可能妨碍了轴突连接的检测，特别是在沟区域（Reveley et al. 2015）。此外，与疾病相关的白质结构的变化会使情况更加复杂（Stewart et al. 1975; Landrieu et al. 1998; Sallet et al. 2003; Hardan et al. 2004; Harris et al. 2004; Thompson et al. 2004; Tortori- Donati et al. 2005; Nordahl et al. 2007; Pang et al. 2008; Bullmore and Sporns 2009; Barkovich 2010; Honey et al. 2010; Manzini and Walsh 2011; Wang et al. 2019; Wang et al. 2020; Zhang et al. 2019, 2020a; Yu et al. 2021）。然而，不可否认的是，白质在大脑折叠图案的形成中发挥着重要的作用，并且我们打算在未来的研究中将其纳入考虑范围中：（i）调查具有相似解剖学特征的相应3HGs是否还具有相似的纤维连接图案；（ii）确定具有相似解剖学特征和纤维连接模式的3HGs组并研究它们的功能同质性/异质性；（iii）将纤维连接模式和功能纳入当前框架中，建立一个更全面的3HGs图，具有极好的功能同质性和内在建立的跨主体皮质对应关系。然后，基于这个3HGs图，我们可以扩大研究范围并包括更多的景观，例如2个铰链。

one-hot编码vs学习嵌入。3HGs的ROI特征最初由one-hot编码表示，并用作输入特征来学习解剖学明显的嵌入向量。与学习的嵌入向量相比，one-hot向量不能直接用作嵌入向量来推断两个主体之间的对应关系，原因有两点：（i）one-hot向量是没有解剖学意义的，不能提供可靠的主体间3HGs对应关系。每个one-hot向量包含一个1和N-1个零，其中N是维数。结果，ROIs独立嵌入且彼此距离相等，这使得无法表示ROIs之间的潜在关系。这样，基于one-hot编码的两个3HGs的相似性仅与它们的多跳特征所共享的常见ROIs数量有关，而忽略了ROIs之间的潜在联系，从而在推断3HG对应关系时无力为。例如，使用one-hot编码，所有没有多跳特征共享的ROIs的3HG对的解剖相似度为零。相比之下，学习的嵌入向量可以有效地编码ROIs之间的群体级别的共同连接模式，相互紧密连接的ROIs具有相似的嵌入向量，因此可以提供更可靠的解剖对应关系；（ii）one-hot向量是稀疏的，并随着词汇量的增加而增长，这很容易引起维数灾难，而嵌入向量是密集的和低维的，更具计算效率。总的来说，学习的嵌入向量在解剖对应可靠性和计算效率方面比one-hot向量更高效。

结论。本文提出了一个基于学习的嵌入框架，将3HGs的解剖学明显图案嵌入为一组可学习的嵌入向量。每个3HG可以通过一组多跳组合系数表示为已学习的嵌入向量的分层组合。通过这种方式，折叠图案的规则性被编码到嵌入向量中，并且个体特定系数保留变异性。我们使用HCP数据集评估了所提出的方法，实验结果表明，学习的嵌入向量可以成功地编码皮质折叠图案并可靠地推断3HGs的跨主体复杂多对多的对应关系。

资助。本研究得到了来自德克萨斯大学系统的STARTs的支持，部分资助来自美国国立卫生研究院（R01AG075582, RF1NS128534）和美国国家科学基金会（IIS-2011369）。利益冲突声明：作者声明在研究过程中没有商业或财务关系可能构成潜在利益冲突。

参考文献。Barkovich AJ. Current concepts of polymicrogyria. Neuroradiology.2010:52:479–487。

## 

---

 ## 原文[9/12]： 

 
 Bertrand G. On topological watersheds. Journal of Mathematical Imag-ing and Vision. 2005:22:217–230.

 Brown T, Mann B, Ryder N, Subbiah M, Kaplan JD, Dhariwal P, Nee lakantan A, Shyam P, Sastry G, Askell A, Agarwal S. Language models are few-shot learners. Advances in neural information pro-cessing systems. 2020:33:1877–1901.

 Budde MD, Annese J. Quantification of anisotropy and fiber orien tation in human brain histological sections. Front Integr Neurosci.2013:7:3.

 Bullmore E, Sporns O. Complex brain networks: graph theoretical analysis of structural and functional systems. Nat Rev Neurosci.2009:10:186–198.

 Chen H, Li Y, Ge F, Li G, Shen D, Liu T. Gyral net: A new representation of cortical folding organization. Med Image Anal. 2017:42:14–25. Chen, H., Perozzi, B., Hu, Y., & Skiena, S. (2018). Harp: Hierarchical representation learning for networks. In Proceedings of the AAAIConference on Artificial Intelligence. 32.

 Devlin J, Chang M-W, Lee K, Toutanova K. Bert: Pre-training of deep bidirectional transformers for language understanding. 2018:arXiv preprint arXiv:1810.04805.

 Dubois J, Benders M, Borradori-Tolsa C, Cachia A, Lazeyras F, HaVinh Leuchter R, Sizonenko S, Warfield S, Mangin J, Hu ¨lppi PS. Primary cortical folding in the human newborn: an early marker of later functional development. Brain. 2008:131:2028–2041. Fischl B. Freesurfer. NeuroImage. 2012:62:774–781. Fischl B, Sereno MI, Dale AM. Cortical surface-based analysis: Ii: inflation, flattening, and a surface-based coordinate system.NeuroImage. 1999:9:195–207.

 Fischl B, Rajendran N, Busa E, Augustinack J, Hinds O, Yeo BT, Mohlberg H, Amunts K, Zilles K. Cortical folding patterns and predicting cytoarchitecture. Cereb Cortex. 2008:18:1973–1980. Ge F, Li X, Razavi MJ, Chen H, Zhang T, Zhang S, Guo L, Hu X, Wang X, Liu T. Denser growing fiber connections induce 3-hinge gyralfolding. Cereb Cortex. 2018:28:1064–1075.

 Giedd JN, Rapoport JL. Structural mri of pediatric brain development: what have we learned and where are we going? Neuron. 2010:67:728–734.

 Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 855–864. Hardan AY, Jou RJ, Keshavan MS, Varma R, Minshew NJ. Increased frontal cortical folding in autism: a preliminary mri study. Psy-chiatry Res Neuroimaging. 2004:131:263–268.

 Harris JM, Whalley H, Yates S, Miller P, Johnstone EC, Lawrie SM. Abnormal cortical folding in high-risk individuals: a predictor of the development of schizophrenia? Biol Psychiatry. 2004:56:182–189.

 Hilgetag CC, Barbas H. Role of mechanical factors in the morphology of the primate cerebral cortex. PLoS Comput Biol. 2006:2:e22.

 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022 

12 | Cerebral Cortex, 2022 

Holland MA, Miller KE, Kuhl E. Emerging brain morphologies from axonal elongation. Ann Biomed Eng. 2015:43:1640–1653. Honey CJ, Thivierge J-P, Sporns O. Can structure predict function in the human brain? NeuroImage. 2010:52:766–776. Landrieu P, Husson B, Pariente D, Lacroix C. Mrineuropathologi cal correlations in type 1 lissencephaly. Neuroradiology. 1998:40:173–176.  

---

 ## 翻译[9/12]：
 

Bertrand G. 关于拓扑分水岭。《数学成像与视觉杂志》。2005:22:217-230。

Brown T, Mann B, Ryder N, Subbiah M, Kaplan JD, Dhariwal P, Neelakantan A, Shyam P, Sastry G, Askell A, Agarwal S. 语言模型是少样本学习者。《神经信息处理系统进展》。2020:33:1877-1901。

Budde MD, Annese J. 人类脑组织切片中各向异性和纤维定向的量化分析。《前沿整合神经科学》。2013:7:3。

Bullmore E, Sporns O. 复杂脑网络：结构与功能系统的图论分析。《自然神经科学评论》。2009:10:186-198。

Chen H, Li Y, Ge F, Li G, Shen D, Liu T. GyrAl Net：一种新的皮层折叠结构表示。《医学图像分析》。2017:42:14-25。Chen, H., Perozzi, B., Hu, Y., 和 Skiena, S. (2018). Harp: 面向网络的分层表示学习。《人工智能AAAI会议论文集》。32。

Devlin J, Chang M-W, Lee K, Toutanova K. Bert：深度双向变压器的预训练模型，用于语言理解。2018:arXiv预印本arXiv:1810.04805。

Dubois J, Benders M, Borradori-Tolsa C, Cachia A, Lazeyras F, HaVinh Leuchter R, Sizonenko S, Warfield S, Mangin J, Hu ¨ lppi PS. 人类新生儿的初级皮层折叠：晚期功能发展的早期标志。《大脑》。2008:131:2028-2041。Fischl B. FreeSurfer。《神经影像》。2012:62:774-781。Fischl B，Sereno MI，Dale AM。基于皮层表面的分析：II: 膨胀、平面化和一个基于表面的坐标系。《神经影像》。1999:9:195-207。

Fischl B, Rajendran N, Busa E, Augustinack J, Hinds O, Yeo BT, Mohlberg H, Amunts K, Zilles K. 皮层折叠模式和对细胞学结构的预测。《大脑皮层》。2008:18:1973-1980。Ge F, Li X, Razavi MJ, Chen H, Zhang T, Zhang S, Guo L, Hu X, Wang X, Liu T. 更密集的纤维连接诱导三铰拐点皮层折叠。《大脑皮层》。2018:28:1064-1075。

Giedd JN, Rapoport JL. 儿童脑结构MRI发展情况：我们学到了什么，我们的研究方向在哪里？《神经元》。2010:67:728-734。

Grover, A., & Leskovec, J. (2016). node2vec：适用于网络的可扩展的特征学习。《第22届ACM SIGKDD国际会议的论文集》。855-864。

Hardan AY, Jou RJ, Keshavan MS, Varma R, Minshew NJ. 自闭症患者额叶皮层折叠增多：一项初步的MRI研究。《精神病学研究》。2004:131：263-268。

Harris JM, Whalley H, Yates S, Miller P, Johnstone EC, Lawrie SM. 高风险个体异常的皮层折叠：精神分裂症发展的预测因素？《生物精神病学》。2004:56:182-189。

Hilgetag CC, Barbas H. 机械因素在灵长类大脑皮层形态学中的作用。《PLoS计算生物学》。2006:2:e22。

Landrieu P, Husson B, Pariente D, Lacroix C. 1型光滑脑性发育不良的MRI-神经病理相关性。《神经放射学》。1998:40:173-176。

## 

---

 ## 原文[10/12]： 

 
 Li K, Guo L, Li G, Nie J, Faraco C, Cui G, Zhao Q, Miller LS, Liu T. Gyral folding pattern analysis via surface profiling. NeuroImage.2010:52:1202–1214.

 Li G, Wang L, Shi F, Lyall AE, Lin W, Gilmore JH, Shen D. Mapping longitudinal development of local cortical gyrification in infants from birth to 2 years of age. J Neurosci. 2014:34:4228–4238. Li G, Wang L, Shi F, Lyall AE, Lin W, Gilmore Li X, Chen H, Zhang T, Yu X, Jiang X, Li K, Li L, Razavi MJ, Wang X, Hu X et al. Commonly preserved and species-specific gyral folding patterns across primate brains. Brain Struct Funct. 2017:222:2127–2141. Manzini MC, Walsh CA. What disorders of cortical development tell us about the cortex: one plus one does not always make two. CurrOpin Genet Dev. 2011:21:333–339.

 Mikolov T, Chen K, Corrado G, Dean J. Efficient estimation of word representations in vector space. 2013: arXiv preprintarXiv:1301.3781.

 Murtagh F, Contreras P. Algorithms for hierarchical clustering: an overview. Wiley Interdisciplinary Reviews: Data Mining and KnowledgeDiscovery. 2012:2:86–97.

 Narayanan A,Chandramohan M,Venkatesan R,Chen L,Liu Y,Jaiswal S. graph2vec: Learning distributed representations of graphs.2017: arXiv preprint arXiv:1707.05005.

 Nie J, Guo L, Li K, Wang Y, Chen G, Li L, Chen H, Deng F, Jiang X, Zhang T et al. Axonal fiber terminations concentrate on gyri.Cereb Cortex. 2012:22:2831–2839.

 Nordahl CW, Dierker D, Mostafavi I, Schumann CM, Rivera SM, Amaral DG, Van Essen DC. Cortical folding abnormalities in autism revealed by surface-based morphometry. J Neurosci.2007:27:11725–11735.

 Pang T, Atefy R, Sheen V. Malformations of cortical development.Neurologist. 2008:14:181.

 Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: Global vec tors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) 1532–1543. Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining.701–710.

 Peters ME, Neumann M, Iyyer M, Gardner M, Clark C, Lee K, Zettle moyer L. Deep contextualized word representations. 2018: arXivpreprint arXiv:1802.05365.

 Radford A, Narasimhan K, Salimans T, Sutskever I. Improving lan guage understanding by generative pre-training. 2018. Radford A, Wu J, Child R, Luan D, Amodei D, Sutskever I. Lan guage models are unsupervised multitask learners. OpenAI blog.2019:1:9.

 Reveley C, Seth AK, Pierpaoli C, Silva AC, Yu D, Saunders RC, Leopold DA, Frank QY. Superficial white matter fiber systems impede detection of long-range cortical connections in diffusion mr trac tography. Proc Natl Acad Sci. 2015:112:E2820–E2828. Roth G, Dicke U. Evolution of the brain and intelligence. Trends CognSci. 2005:9:250–257.

 Sallet PC, Elkis H, Alves TM, Oliveira JR, Sassi E, de Castro CC, Busatto GF, Gattaz WF. Reduced cortical folding in schizophrenia: an mri morphometricstudy.

 Am J Psychiatr. 2003:160: 1606–1613. Stewart RM, Richman DP, Caviness VS. Lissencephaly and pachy-gyria. Acta Neuropathol. 1975:31:1–12.  

---

 ## 翻译[10/12]：
 

Li K, Guo L, Li G, Nie J, Faraco C, Cui G, Zhao Q, Miller LS, Liu T. 基于表面剖析的脑回折叠模式分析。NeuroImage。2010:52:1202-1214.

Li G, Wang L, Shi F, Lyall AE, Lin W, Gilmore JH, Shen D. 从出生到2岁婴儿局部皮质脑回折叠的纵向发展映射。J Neurosci。2014:34:4228-4238。

Li G, Wang L, Shi F, Lyall AE, Lin W, Gilmore Li X, Chen H, Zhang T, Yu X, Jiang X, Li K, Li L, Razavi MJ, Wang X, Hu X 等。灵长类大脑中常见保留和物种特异性的脑回折叠模式。Brain Struct Funct。2017:222:2127-2141。

Manzini MC, Walsh CA. 关于大脑的皮层发育异常告诉我们的：一加一不总等于二。CurrOpin Genet Dev。2011:21:333-339。

Mikolov T, Chen K, Corrado G, Dean J. 向量空间中单词表示的高效估计。2013：arXiv预印本arXiv:1301.3781。

Murtagh F, Contreras P. 分层聚类算法概述。Wiley跨学科评论：数据挖掘和知识发现。2012:2:86-97。

Narayanan A, Chandramohan M, Venkatesan R, Chen L, Liu Y, Jaiswal S. graph2vec：学习图的分布式表示。2017:arXiv预印本arXiv:1707.05005。

Nie J, Guo L, Li K, Wang Y, Chen G, Li L, Chen H, Deng F, Jiang X, Zhang T 等。轴突纤维的终止集中于回。Cereb Cortex。2012:22:2831-2839。

Nordahl CW, Dierker D, Mostafavi I, Schumann CM, Rivera SM, Amaral DG, Van Essen DC. 表面分析形态学揭示孤独症患者的大脑皮层折叠异常。J Neurosci。2007:27:11725-11735。

Pang T, Atefy R, Sheen V. 皮层发育畸形。神经学家。2008:14:181。

Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: 全局单词表示。在2014会议工作坊中的论文。1532-1543。

Perozzi, B., Al-Rfou, R., &Skiena, S. (2014). Deepwalk：社交关系的在线学习。在第20届ACM SIGKDD国际会议上的论文。701-710。

Peters ME，Neumann M，Iyyer M，Gardner M，Clark C，Lee K，Zettlemoyer L. 深度上下文化的单词表示。2018：arXiv预印本arXiv:1802.05365。

Radford A, Narasimhan K, Salimans T, Sutskever I. 基于生成性预训练提高语言理解。2018。

Radford A, Wu J, Child R, Luan D, Amodei D, Sutskever I. 语言模型是无监督多任务学习者。OpenAI博客。2019:1:9。

Reveley C, Seth AK, Pierpaoli C, Silva AC, Yu D, Saunders RC, Leopold DA, Frank QY. 表浅白质纤维系统阻碍了扩散磁共振成像中长程皮层连接的检测。Proc Natl Acad Sci。2015:112:E2820-E2828。

Roth G, Dicke U. 大脑和智力的演化。Trends CognSci。2005:9:250-257。

Sallet PC, Elkis H, Alves TM, Oliveira JR, Sassi E, de Castro CC, Busatto GF, Gattaz WF. 患精神分裂症者皮层折叠减少: MRI形态计量研究。美国精神病医生。2003:160: 1606–1613。

Stewart RM, Richman DP, Caviness VS. 光滑大脑和厚大脑。Acta Neuropathol。1975:31:1-12。

## 

---

 ## 原文[11/12]： 

 
 Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015). Line: Largescale information network embedding. In Proceedings of the 24th international conference on world wide web 1067–1077. Thompson PM, Hayashi KM, Sowell ER, Gogtay N, Giedd JN, Rapoport JL, De Zubicaray GI, Janke AL, Rose SE, Semple J et al. Mapping cortical change in alzheimer’s disease, brain development, andschizophrenia. NeuroImage. 2004:23:S2–S18.

 Tortori-Donati P, Rossi A, Biancheri R. Brain malformations. In: Pedi atric neuroradiology. Springer, Berlin, Heidelberg; 2005. pp. 71–198 Van Essen DC. A tension-based theory of morphogenesis and com pact wiring in the central nervous system. Nature. 1997:385:313–318.

 Wang, D., Cui, P., & Zhu, W. Structural deep network embedding. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016:1225–1234. Wang L, Zhang L, Zhu D. Accessing latent connectome of mild cognitive impairment via discriminant structure learning. In: In 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019). IEEE; 2019. pp. 164–168 Wang L, Zhang L, Zhu D. Learning latent structure over deep fusion model of mild cognitive impairment. In: In 2020 IEEE 17th Inter national Symposium on Biomedical Imaging (ISBI). IEEE; 2020. pp. 1039–1043 Xu G, Knutsen AK, Dikranian K, Kroenke CD, Bayly PV, Taber LA. Axons pull on the brain, but tension does not drive cortical folding; 2010 Yu X, Scheel N, Zhang L, Zhu DC, Zhang R, Zhu D. Free water in t2 flair white matter hyperintensity lesions. Alzheimers Dement.2021:17:e057398.

 Zhang L, Zaman A, Wang L, Yan J, Zhu D. A cascaded multi modality analysis in mild cognitive impairment. In: International Workshop on Machine Learning in Medical Imaging. Springer; 2019. pp. 557–565 Zhang L, Wang L, Zhu D. Jointly analyzing alzheimer’s disease related structure-function using deep cross-model attention network. In: In 2020 IEEE 17th International Symposium on Biomedical Imaging (ISBI). IEEE; 2020a. pp. 563–567 Zhang L, Wang L, Zhu D. Recovering brain structural connec tiv ity from functional connectivity via multi-gcn based generative adversar ial network. In: International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer; 2020b. pp. 53–61 Zhang T, Huang Y, Zhao L, He Z, Jiang X, Guo L, Hu X, Liu T. Identifying cross-individual correspondences of 3-hinge gyri. MedImage Anal. 2020c:63:101700.

 Zhang T, Li X, Jiang X, Ge F, Zhang S, Zhao L, Liu H, Huang Y, Wang X, Yang J et al. Cortical 3-hinges could serve as hubs in cortico-cortical connective network. Brain imaging and behavior.2020d:14(6):2512–2529.

 Zhang L, Wang L, Gao J, Risacher SL, Yan J, Li G, Liu T, Zhu D, Initiative ADN et al. Deep fusion of brain structure-function in mild cognitive impairment. Med Image Anal. 2021:72:102082. Zhang L, Wang L, Zhu D, Initiative ADN et al. Predicting brain structural network using functional connectivity. Med Image Anal.2022:79:102463.

 Zilles K, Armstrong E, Schleicher A, Kretschmann H-J. The hu man pattern of gyrification in the cerebral cortex. Anat Embryol.1988:179:173–179.

 Downloaded from https://academic.oup.com/cercor/advance-article/doi/10.1093/cercor/bhac465/6880883 by University of Texas at Arlington user on 10 December 2022  

---

 ## 翻译[11/12]：
 

2015年，唐杰、曲梅、王敏、张明、闫俊和梅清发表了一篇题为“Line: Largescale information network embedding”的学术论文，收录在第24届国际万维网会议的论文集中，论文页码为1067-1077。Thompson PM、Hayashi KM、Sowell ER、Gogtay N、Giedd JN、Rapoport JL、De Zubicaray GI、Janke AL、Rose SE、Semple J等人在2004年发表了一篇以“Mapping cortical change in Alzheimer’s disease, brain development, and schizophrenia”为题的论文，发表于期刊NeuroImage，论文页码为S2-S18。

Tortori-Donati P、Rossi A、Biancheri R在2005年发表了一篇章节为“Brain malformations”的论文，收录于专著《Pediatric neuroradiology》中，由Springer出版社发行，页码为71-198。Van Essen DC在1997年发表了一篇题为“A tension-based theory of morphogenesis and compact wiring in the central nervous system”的研究论文，发表于期刊Nature，页码为313-318。

2016年，王鼎、崔鹏和朱伟发表了一篇名为“Structural deep network embedding”的学术论文，收录在第22届ACM SIGKDD国际知识发现与数据挖掘会议的论文集中，论文页码为1225-1234。王亮、张玲和朱东巍在2019年发表了一篇题为“Accessing latent connectome of mild cognitive impairment via discriminant structure learning”的论文，收录于2019年IEEE第16届国际生物医学成像研讨会(ISBI 2019)论文集中，页码为164-168；在2020年，他们又发表了一篇题为“Learning latent structure over deep fusion model of mild cognitive impairment”的论文，收录于2020年IEEE第17届国际生物医学成像研讨会(ISBI 2020)论文集中，页码为1039-1043。徐刚、Knutsen AK、Dikranian K、Kroenke CD、Bayly PV和Taber LA在2010年发表了一篇题为“Axons pull on the brain, but tension does not drive cortical folding”的论文。

于希、Scheel N、张玲、朱东巍、张锐和朱东巍在2021年发表了一篇题为“Free water in t2 flair white matter hyperintensity lesions”的论文，发表在期刊《Alzheimer's & Dementia》上，页码为17:e057398。

张玲、Zaman A、王亮、闫俊和朱东巍在2019年发表了一篇名为“ A cascaded multi modality analysis in mild cognitive impairment”的学术论文，收录于国际医学影像机器学习研讨会论文集中，由Springer出版社发行，页码为557-565；同年，她们还发表了一篇题为“Jointly analyzing Alzheimer's disease related structure-function using deep cross-model attention network”的论文，收录于2020年IEEE第17届国际生物医学成像研讨会(ISBI 2020)论文集中，页码为563-567。张玲、王亮和朱东巍在2020年发表了一篇题为“Recovering brain structural connectivity from functional connectivity via multi-gcn based generative adversarial network”的论文，收录于医学图像计算与辅助干预国际会议论文集中，页码为53-61。

张涛、黄勇、赵磊、何增进、姜肖瑶、郭力、胡潇、刘涛在2020年发表了一篇题为“Identifying cross-individual correspondences of 3-hinge gyri”的论文，发表在期刊《Medical Image Analysis》上，页码为63：101700。张涛、李旭、姜肖瑶、葛峰、张升、赵磊、刘浩、黄勇、王旭、杨洁等人在2020年发表了一篇题为“Cortical 3-hinges could serve as hubs in cortico-cortical connective network”的论文，发表在期刊《Brain Imaging and Behavior》上，页码为14（6）：2512–2529。

张玲、王亮、高君、Risacher SL、闫俊、李根、刘涛、朱东巍和追踪阿尔茨海默病神经影像学研究倡议（ADNI）等人在2021年发表了一篇题为“Deep fusion of brain structure-function in mild cognitive impairment”的论文，发表在期刊《Medical Image Analysis》上，页码为72：102082。张玲、王亮、朱东巍和ADNI等人在2022年发表了一篇题为“Predicting brain structural network using functional connectivity”的论文，发表在期刊《Medical Image Analysis》上，页码为79：102463。

Zilles K、Armstrong E、Schleicher A和Kretschmann H-J在1988年发表了一篇名为“The human pattern of gyrification in the cerebral cortex”的学术论文，发表于期刊《Anatomy and Embryology》上，页码为179：173-179。

