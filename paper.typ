#import "template.typ": *

#show: project.with(
  title: "支持动态图分析的图存储研究与实现",
  author: "杨世蛟",
  abstract_zh: [
    TODO
  ],
  abstract_en: [
    TODO
  ],
  keywords_zh: ("关键词1", "关键词2", "关键词3"),
  keywords_en: ("Keyword 1", "Keyword 2", "Keyword 3"),
  school: "计算机科学与工程学院",
  id: "20194652",
  mentor: "张岩峰",
  class: "计算机科学专业 1905 班",
  date: (2023, 6, 1)
)

= 绪论

== 选题意义

图数据，用来表达实体及其之间的关系，在许多领域中都有广泛的应用，如社交网络、Web、语义网、路线图、通信网络、生物学和金融等等。这些只是使用的一小部分例子而已。在科学研究和实际操作中，图处理的普及率已经显著提升，特别是用于管理和处理图的商业和研究软件数量剧增。比如图数据库，RDF引擎，可视化软件，查询语言，以及分布式图处理系统。在学术界，则是发表了大量的与图处理相关联的论文。

VLDB2018年的一篇文献 #cite("sahu2017ubiquity") 中，通过问卷的方式收集了图相关的一些工作负载以及图数据产生的方式，并同时收集了用户在使用图相关软件的过程中遇到的问题。其中首先值得关注的有图的大小，用户的图数据大小在小于100MB到大于1TB上都有分布；其次是绝大多数用户使用的图都是动态图，即图本身的拓扑结构，点边上的属性，都会随着时间不断变化；然后是图计算中使用频率较高的算法为查找连通块，邻居查询（2跳查询），查找最短路径，子图匹配，三角计数，可达性检查等。在ML领域，则是近几年比较火的图神经网络（GNN） #cite("scarselli2008graph") ，比如图卷积神经网络（GCN） #cite("kipf2016semi") ；绝大多数的用户都是在图数据库上进行的图查询以及图存储，比如Neo4j #cite("neo4j") ，部分用户则是使用一些分布式的迭代计算引擎，比如Hadoop，Spark等，还有部分用户则是靠传统的关系型数据库来做数据的存储以及查询，比如MySQL，PostgreSQL；最后值得关注的则是用户认为图查询上的挑战，绝大多数的用户认为图处理和图存储的可拓展性（Scalability）是最主要的问题，部分用户认为图的可视化也是比较关键的问题，即如何将图数据清晰的展示给用户，以体现出其中的数据特征，还有部分用户认为图查询语言是一个有挑战的问题，根本原因在于目前图查询语言不像是关系型数据库那样统一使用SQL或者其变种，图查询语言比较多样并且不用语言写法差距较大，比如OpenCypher，Gremlin，不过近期推出的GQL标准应该是一个解决这个问题的方法。

VLDB2019年的一篇文献中 #cite("ozsu2019graph") 中，则进一步描绘了图处理领域的全貌，给出了若干个图处理领域相关的问题以及例子，并在最后提出了一些开放性的问题。

最近的一个工作#cite("tian2023world") 中，则是发现了目前大多数图相关的工作都是围绕学术界展开，而缺少了工业界的视角，所以给出了在工业界的视角中，图数据库目前的发展状况。在模型上，将图模型分为两大类，RDF模型，以及属性图模型。在图数据库上，将其分为2大类，分别是图原生数据库，例如TigerGraph #cite("tigergraph") ，Neo4j #cite("neo4j") ，以及混合图数据库，比如Microsoft Cosmos DB Graph #cite("cosmosdb") ，Oracle Spatial & Graph #cite("spatialandgraph") 。最后还提出了图数据库发展的方向以及一些机会，比如关注端到端的解决方法，利用云厂商拥有全套数据栈的优势来提升端到端的服务能力，针对拓展性强的分布式图数据库进行研究，优化图数据库的导入导出（ETL）能力，针对动态图进行优化等。

上述工作都在对现有的图处理系统进行总结，尝试识别出图数据特有的工作负载，并针对这些特有的工作负载进行优化。在这个背景下，图原生的数据库则是变得越来越重要，因为相较于一些多模数据库是基于NoSQL系统作为底座构建，上层提供一个图查询相关的API来说，图原生数据库则会针对图的工作负载对底层存储结构，计算引擎等关键模块进行特化的设计，从而达到更优的性能。并且上述图计算负载下一些特殊的多跳查询则会加剧图原生数据库和多模数据库的性能差距。在用户更加关注的端到端服务方面，图原生数据库直接负责了数据的存储以及计算，相较于离线计算中用户从关系型数据库中将数据导入到HIVE中，再通过ETL将其转入到图计算引擎中，节省的数据的导入导出过程，使得整个过程更加高效的同时，还保证了数据的新鲜度。

综上，图数据库，尤其是图原生数据库，在目前更加关注深数据，图数据的后关系型数据库时代中扮演着越来越关键的角色。而在数据库中，计算和存储则是最为核心的两个概念，主要关注快速计算的计算引擎在OLAP数据库中一般扮演较为核心的角色，其主要目的是加速算子的执行。而主要关注快速存储的存储引擎则是在OLTP中扮演较为核心的角色，其主要目的则是提供高性能的事务语义的读写接口，为上层的计算引擎服务。作为针对图工作负载优化的图原生数据库，无论是计算引擎还是存储引擎都需要重新审视历史的设计思路，并给出对于图数据库更优的设计方案。本次毕业设计的选题则是针对图原生数据库，设计并实现一个高性能的图存储引擎。

== 国内外研究现状

在近几年的数据库相关领域的顶会中，有一些对于图存储系统的研究，这里给出一个简单的介绍。Teseo #cite("de2021teseo") 整体是一个类似CSR的结构，核心的数据结构为Fat Tree，在非叶节点的索引部分为ART #cite("leis2013adaptive") ，而在叶节点部分则是支持重平衡的稀疏数组(Packed Memory Array)#cite("bender2000cache") 。LiveGraph #cite("zhu2019livegraph") 是首个提出的可以支持磁盘存储的图存储系统，整体是类邻接表的结构，核心思路是通过TEL(Transactional Edge Log)来支持MVCC以及顺序写。GraphOne #cite("kumar2020graphone") 的核心思路是前台写入EdgeLog，然后异步的持久化到磁盘中，后台线程负责将EdgeLog合并到紧凑的邻接表中，邻接表是块状链表，写入每次都是追加写，在无用数据过多的时候进行Compaction。Kuzu #cite("fengkuzu") 则是提出了一种针对图数据库的列存格式，实现了基于列存的存储引擎。Kuzu还进一步提出了A+Index #cite("mhedhbi2021a+") ，针对图查询特性而做的二级索引，并且物化了两个特殊的二级索引来加速查询。Sortledton #cite("fuchs2022sortledton") 中总结了以往的图存储系统，并提出了三个划分指标，分别是是否支持高性能的扫描，是否支持事务语义的读写，以及是否支持用于高效进行模式匹配的数据交集查询，并识别出了为了针对这些工作负载，对于图存储结构的需求是什么。最后提出了一个类似邻接表结构有序跳表来进行图的存储。

目前针对图存储系统的研究中，有几个主要的问题，一个是认为图更新频率较低，所以都采用类似CSR的结构来加速读取性能，但是忽视了CSR结构更新开销非常困难的问题。第二个是认为数据能够放到内存中，所以绝大多数的系统没有考虑数据规模超出内存后的处理方法，在数据规模较大的时候只能靠虚拟内存和交换区，进而导致性能的极速下降。在支持磁盘存储的系统LiveGraph中，使用的也是简单的MMAP管理内存，并没有考虑缓冲区等问题。并且在数据的组织方面，LiveGraph使用了简单的追加写入的方式，并在预分配的空间不够的时候，分配一个更大的磁盘空间并将其搬运过去。这种方式对于点查请求效率极低，并且也无法支持高性能的模式匹配查询。

ByteGraph #cite("li2022bytegraph") 是一个针对磁盘设计的分布式图数据库，用BTree来组织数据，分为查询层，存储层，磁盘层三层。整体性能较为出色，但是其主要的缺点有两个，磁盘层作为共享存储，由分布式KV实现，底层原理是LSM-Tree，在KV较大的时候会引入较大的写放大，同时KV不能感知上层语义，磁盘层无法针对图数据进行优化存储。第二点是BTree的写入放大比较严重，进而会加剧磁盘层的写入放大以及磁盘带宽压力。

Neo4j #cite("neo4j") 作为图原生数据库，磁盘中采用类似链表的方式组织数据，通过特殊的ID分配方式来规避索引的使用（免索引邻接）。这种方式的缺点在于随机访问过多，在内存不够的情况下，会导致出现大量的随机IO，进而导致性能退化严重。

Nebula #cite("nebulagraph") 是基于KV的分布式图数据库，底层KV是通过Raft + RocksDB构建，将每一个点和每一条边作为单独的KV写入。缺点是扫描顶点出边时需要走LSMTree的Merge Read流程，性能不稳定。

Dgraph #cite("dgraph") 是基于一个支持KV分离的KV引擎构建的图数据库，一个顶点的同类型出边会被存到一个KV中，扫描出边的局部性比较好。缺点是出度较大时写入放大过于严重，并且并发写入的能力较差。

Igraph #cite("igraph") 是基于定制的Indexlib构建的针对图分析的图数据库。Indexlib支持倒排索引，KV索引，以及KKV索引。其中KKV分为PKey，SKey，Value三个文件。单个PKey下的SKey连续存储。扫描出边的局部性比较好。聚合邻接表的缺点是写入放大比较高，出度较大时写入放大问题较为严重。

AWS Neptune #cite("neptune") 整体架构类似Aurora #cite("aurora_web") ，数据组织格式目前不详。

综上，现有的图存储系统的主要缺点有：没有针对磁盘设计，对于数据集规模超过内存的情况下性能下降较为严重；认为图更新的频率较低，大多数系统只考虑OLAP类查询；部分系统是基于多模数据库上实现图语义相关接口，存储引擎和计算引擎无法针对图的工作负载进行优化。

== 主要研究内容

本次毕业设计所研究的内容就是针对图的工作负载，设计一个支持事务语义的高性能单机图存储引擎。其关键的研究点有：

1. 针对可拓展性进行设计，从而保证避免阻塞该存储引擎作为分布式图数据库存储引擎的可能性。

2. 针对图的工作负载进行设计，识别图工作负载对于存储结构的需求，并根据需求针对存储结构进行优化。

3. 针对磁盘进行设计，不对用户的数据集以及硬件环境有所假设。

4. 针对云端一体进行设计，不对存储底座有所假设，使得该存储引擎可以灵活的部署到云上环境（S3 #cite("s3") ，EBS #cite("ebs") ），以及单机环境（Ext4 #cite("ext4") ）。


#pagebreak()

= 相关理论与技术

== 图工作负载

在客户用例方面，图数据库已经应用于许多垂直行业，包括金融、保险、医疗、零售、能源、电力、制造、政府、营销、供应链、交通等。这种图在许多领域具有多样化和广泛适用性的现象也在 #cite("sahu2017ubiquity") 中得到观察。关于图数据库的一些具体用例已经在 #cite("top10usecases", "17usecases", "customersuccessstories", "tian2019synergistic") 中提供。也许，图数据库应用的最常见示例是欺诈检测。例如，#cite("tian2019synergistic") 展示了一个详细的示例场景，通过遍历包含保险理赔信息和患者医疗记录的图来检测欺诈理赔。

与关系数据库中的不同类型工作负载类似，图数据库也有两种不同类型的工作负载。第一种类型侧重于低延迟的图遍历和模式匹配，通常被称为图查询。这些查询只涉及图的小局部区域，例如查找顶点的 2 跳邻居，或者查找两个顶点之间的最短路径。由于低延迟要求和图查询的交互式特性，人们也将它们称为图 OLTP。图 OLTP 通常用于探索性分析和案例研究。图数据库工作负载的第二种类型是图算法，通常涉及对整个图进行迭代、长时间运行的处理。典型的例子是 Pagerank 和社区检测算法。图算法通常用于类似 BI 的应用。因此，人们也称之为图 OLAP。最近，一种将图和机器学习结合在一起的新趋势出现了，称为图 ML。例如，图嵌入或顶点嵌入用于将图结构转换为向量空间，然后将这些空间作为 ML 模型训练的特征。图神经网络（GNN）是图 ML 的另一个例子。通常情况下，图 ML 与图 OLAP 工作负载一起归类。

然而，随着图数据库的普及，各大公司也在追求一站式服务，为了减少数据的ETL，追求极致的性能，通常拥有全部数据栈的云服务厂商会通过一些其他的方式避免数据的转化。如一些常见的使用方法是用户通过RDBMS将数据写入到传统数据库中（MySQL等），然后通过ETL将数据导入到数据仓库进行AP查询，或者将数据转化成图格式，来进行图相关的处理。而拥有全部数据栈的云厂商可以通过多模数据库来减少数据的转化，或者直接用图查询语言来进行数据的写入，实现图数据，分析的一站式处理。

== 数据模型

虽然在关系世界中存在一个明确的关系数据模型和一个明确的事实标准(“SQL模型”)，但在图领域的情况则不太明确。

在讨论图数据库时，我们首先需要了解它所支持的图模型。大多数商业图数据库支持的两种主要图模型是 RDF 模型和属性图模型。

#img(
  image("./1.png"),
  caption: "RDF模型"
) <img1>

RDF 模型。RDF 是支持链接数据和知识图谱的 W3C 标准套件之一 #cite("w3rdf") 。RDF 图是一个有向边标记图，由主题-谓语-宾语三元组表示。@img1 展示了一个用 RDF 模型表示的示例图。该图包含以下信息：患者 Alice Brown（病人 ID 为 19806）在 2020 年 3 月 24 日被诊断为 2 型糖尿病（疾病 ID 为 64572326）；2 型糖尿病是糖尿病的子类型，其疾病 ID 为 6472345。

以 (Patient 1) -[hasName]-> (Alice Brown) 三元组为例，Patient 1 是主体，hasName 是谓语，Alice Brown 是对象。RDF 模型通常用于知识表示、推理以及语义 Web 应用。例如，DBPedia #cite("dbpedia") 和 YAGO #cite("yago") 都使用 RDF 来表示他们的知识图，并使用 SPARQL #cite("sparql") 支持对知识库的查询。

#img(
  image("./2.png"),
  caption: "属性图模型"
) <img2>

属性图模型。相比之下，属性图是一个有向图，其中每个顶点和边都可以具有任意数量的属性。顶点/边还可以用标签标记，以区分图中不同类型的对象/关系。@img2 展示了如何在属性图模型中表示 @img1 中 RDF 图中捕获的相同信息。在这里，与将患者或疾病的 ID 和名称表示为单独的节点不同，属性图模型可以将它们作为患者和疾病节点的属性。类似地，诊断时间可以表示为 diagnosedWith 边的属性，从而无需创建单独的诊断节点及其连接到患者和疾病节点的边。总的来说，如本示例所示，属性图模型可以用比 RDF 模型更少的节点和边捕捉相同的信息。这是因为在 RDF 模型中，信息只能作为节点或边表示，而在属性图模型中，它还可以定义为现有节点或边的属性，从而导致图中节点和边的数量减少。属性图模型通常用于需要图遍历、模式匹配、路径和图分析的应用。

根据 #cite("tian2023world") 所述，尽管图数据库行业支持这两种模型，但实际上，属性图模型已经得到了压倒性的支持。在该文献中调查的所有主要产品都支持属性图模型，其中两个还支持 RDF 模型。在 #cite("hartig2014reconciliation") 中，Hartig 提出了 RDF 和属性图模型之间的转换，希望将这两种模型进行整合。

总结一下，当谈论图数据库时，我们首先需要了解其所支持的图模型。RDF 模型和属性图模型是目前大多数图数据库支持的两种主要模型。RDF 模型通常用于知识表示、推理和语义 Web 应用，例如 DBPedia 和 YAGO。与之相比，属性图模型可以用更少的节点和边来捕捉相同的信息，并且通常用于需要图遍历、模式匹配、路径和图分析的应用。尽管 RDF 是一个更老的模型，但属性图模型在图数据库行业中获得了更广泛的支持。

== 图查询语言

在图 OLTP 方面，对于 RDF 图，有标准的 SPARQL 查询语言 #cite("sparql")。对于属性图，有许多正在使用和拟议的语言，但没有明确的赢家。其中最具竞争力的之一是 Tinkerpop Gremlin #cite("gremlin")，目前大约有 30 家图数据库供应商支持，也可能是当今使用最广泛的图查询语言。另一个强有力的竞争者是 openCypher #cite("opencypher")。Cypher #cite("francis2018cypher") 最初是 Neo4j 的专有声明性图查询语言，于 2015 年开源。大约有 10 家图数据库供应商支持 openCypher。除了这两种更广泛采用的语言之外，许多供应商还提出了自己的图查询语言。Oracle 提出了一种基于 SQL 的声明性语言，称为 PGQL #cite("pgql")。GSQL #cite("gsql") 是 TigerGraph 采用的类 SQL 图查询语言。Microsoft SQL Graph 扩展了 SQL，用 MATCH 子句进行图模式匹配 #cite("sqlserver_graph")。LDBC #cite("ldbc") 图查询语言任务组（成员包括学术界和产业界）提出了 G-Core #cite("angles2018g")。为了减少图查询语言方面的混乱，2019 年，ISO/IEC 的联合技术委员会 1 批准了一个创建标准图查询语言的项目，称为 GQL #cite("gql")。这项工作还得到了另一个扩展 SQL 的项目的补充，该项目通过图视图定义和图查询构造来扩展 SQL，称为 SQL/PGQ。GQL 和 SQL/PGQ 共享一个通用的声明性图模式匹配语言组件。这个通用组件整合了 openCypher、Oracle 的 PGQL、TigerGraph 的 GSQL 和 LDBC G-CORE 的想法。GQL 和 SQL/PGQ 的标准化工作得到了学术界的大力参与，它是图研究社区对图行业产生重大影响的领域之一。然而，考虑到当前图查询语言的状况，即使 GQL 和 SQL/PGQ 标准发布之后，供应商采用这些标准还需要时间，因为大量的图应用程序已经用这些现有的语言编写。标准化的确立仍将需要很多年的时间。

在语言特性方面，Gremlin 更多地是一种命令式图遍历语言（尽管 Gremlin 的最新版本也具有一些声明性语言特性），而其他语言则是声明性的。因此，Gremlin 相对较低级别且不太用户友好。但在表达能力方面，Gremlin 是图灵完备的 #cite("rodriguez2015gremlin")，而大多数声明性对应物（包括 openCypher）则不是。这意味着有些图算法或操作无法用这些非图灵完备的语言表示。在所有声明性语言中，TigerGraph 的 GSQL 是唯一图灵完备的语言 #cite("gsql2")。

在图 OLAP 方面，同样没有标准语言或 API，但大多数供应商支持类似 Pregel 的 API 变体 #cite("malewicz2010pregel")。与机器学习类似，内置图算法库对用户更有用，因此对于图 OLAP 来说，缺乏标准并不是一个很大的问题。

== 现有图数据库系统

#img(
  image("./3.png"),
  caption: "Major Graph Database Offerings"
) <img3>

图数据库领域在行业中非常拥挤，新项目和初创企业层出不穷。不可能列举出所有当前的图数据库产品。因此，本节仅重点介绍三个类别中的一些主要产品：仅提供图数据库的供应商、具有图支持的数据公司以及具有内置图数据库支持的企业云供应商。他们的图产品不同特性在 @img3 中总结。在下一节中，我们将讨论各种供应商采用的不同架构解决方案。

在纯粹的领域中，Neo4j 和 TigerGraph 是两个最强大的竞争者。他们在本地和主要云（AWS、Azure 和 GCP）上都提供解决方案。他们对图 OLTP 和 OLAP 工作负载的支持非常好。纯粹的参与者还完善了可视化和工具的技巧，以及对大量内置图算法的支持。

DataStax 和 Databricks 是两家拥有广泛数据产品组合的数据公司。图组件也与系统的其他组件紧密集成。例如，DataStax Enterprise Graph（DSG）是基于 DataStax 的主要 NoSQL 数据引擎 Cassandra 构建的。对于 Databricks 的图支持，GraphX 是基于 Spark 的 RDD 构建的，而 GraphFrames 是基于 DataFrames 的。由于这两家公司都致力于更通用的数据系统，因此他们对图的支持并不像仅针对图的供应商那样全面。DataStax 对图 OLAP 的支持非常基本（仅依赖 Gremlin 中的 SparkGraphComputer API，并只有 3 个内置图算法）。Databricks 的图 OLTP 支持仅来自 GraphFrames 中的简单图案查找支持。这种支持不仅受到非常简单的图案查找 DSL 的限制，而且由于图 OLTP 查询处理在底层使用 DataFrames（最初设计用于分析目的），因此性能可能不佳。

图数据库供应商的最后一个类别是大型云公司，包括 Amazon、Microsoft、Oracle 和 IBM。他们在云平台上提供大量的数据服务，内置图数据库服务是其中之一。Microsoft、Oracle 和 IBM 之前都是大型关系数据库商店，因此他们的图数据库解决方案基于他们的关系数据库并不令人惊讶：Microsoft SQL Graph 基于 SQL Server（本地）和 Azure SQL 数据库（云端），Oracle Spatial 和 Graph 基于 Oracle 数据库（本地和云端），IBM Db2 Graph 基于 Db2（本地和 Cloud Pak for Data）。此外，Microsoft 还提供了另一种图数据库解决方案，Cosmos DB Graph，它是基于 NoSQL 数据库 Azure Cosmos DB 构建的。另一方面，Amazon 使用与其他 AWS 平台（如 Aurora 和 DynamoDB）相同的后端存储构建 Neptune。作为纯粹的云公司，Amazon 不提供本地图数据库解决方案。除了 Oracle Spatial 和 Graph 拥有大量内置算法的出色图 OLAP 支持外，此类别中的大多数图数据库都专注于图 OLTP 工作负载。

现在，让我们来看一下 @img3 中表格的不同维度。除了 Amazon Neptune 和 Microsoft Cosmos DB Graph 仅支持云端部署外，大多数供应商都支持本地和云端部署。在图模型方面，所有供应商都支持属性图模型。Amazon Neptune 以及 Oracle Spatial 和 Graph 还额外支持 RDF 模型。对于图 OLTP 工作负载，不同供应商使用的语言反映了上一节中讨论的语言混乱，但 Gremlin 似乎得到了最广泛的支持。由于图 OLTP 工作负载的探索性质，可视化对客户尤为重要。大多数图供应商确实提供了可视化支持。与关系数据库相比，事务支持一直是图数据库的痛点。对单个节点的更新往往会影响其边和连接的节点，例如，删除节点需要删除与其连接的所有边。因此，图数据库中的事务通常更复杂，尤其是在分布式环境中。一些图数据库设法提供完整的 ACID 支持，但其他数据库要么没有支持，要么对事务支持较弱。与图 OLTP 相比，图 OLAP 支持总体相对较弱，但由于内置算法众多，TigerGraph、Neo4j 和 Oracle 表现出色。VLDB 调查 #cite("sahu2017ubiquity") 观察到了大规模图（具有超过十亿条边）的普遍存在，并指出可扩展性是许多用户面临的挑战。因此，主要的图供应商努力解决这个挑战。所有图数据库解决方案都可以在一定程度上很好地扩展，这可以满足很多客户的需求，而且大多数还为那些无法容纳在单个节点上的巨大图提供了扩展解决方案。正如 #cite("fan2022big") 正确指出的，尽管分布式并行化可以处理更大的图，但并不总是提供理想的性能。由于图的连接特性，在分布式环境中几乎不可能实现访问局部性。因此，分布式图计算经常访问图的许多分区，这会产生很多通信成本。如果大型图可以容纳在单个节点中，与相同系统的扩展版本相比，扩展解决方案可能会提供更好的性能。正如 #cite("tian2019synergistic") 中所示，单个节点系统在合适的机器配置上可以轻松处理具有数十亿条边的大型图。然而，与 #cite("sahu2017ubiquity") 一致，高效查询和处理大规模图（远远超过数十亿条边）仍然是一个挑战。

== 图数据库的解决方案空间

#img(
  image("./4.png"),
  caption: "Graph Solution Space"
) <img4>

一种对解决方案空间进行分类的方法是将其划分为原生图数据库和混合图数据库，如 @img4 所示。顾名思义，原生图数据库是专为图而从头构建的专用查询和存储引擎。Neo4j 和 Tigergraph 是原生图数据库的两个典型例子。这类图数据库针对支持的图工作负载进行了高度优化。但缺点是工程成本高，因为它们必须重新发明用于支持事务、访问控制、可扩展性、高可用性（HA）、灾难恢复（DR）等方面的技术。相比之下，混合图数据库具有专用的图查询引擎，但依赖现有的数据存储来处理数据持久化，无论是 SQL 数据库、键值存储还是文档存储。如 @img4 所示，更多的图数据库属于这个阵营。由于混合图数据库将其存储引擎委托给现有的数据存储，因此其开发时间更短。此外，它还可以从后端存储中免费获得许多功能，如事务支持、访问控制、可扩展性、HA 和 DR 等。但潜在的缺点是混合图数据库的性能可能无法与高度优化的原生图数据库相匹敌。当然，单个图数据库的性能也高度依赖于实现细节。

将解决方案空间的另一种分类方式是仅支持图数据库的数据库与融合数据库，也称为多模型数据库。如图3所示，所有原生图数据库都是仅支持图的数据库，而大多数混合图数据库是融合数据库，但其中一些是仅支持图的数据库。仅支持图的数据库只支持图工作负载。实际上，这也可能是这些数据库的一个根本限制。实际上，Neo4j和TigerGraph的用户手册中都有专门关于数据导入和导出的章节。相比之下，融合数据库或多模型数据库支持在共享数据上使用多种查询语言/API。这也是融合数据库架构的一个优势。我们在下面详细阐述了一些优势。

从根本上讲，融合数据库解决方案解决了数据库碎片化问题。真实应用很少只有同质化的工作负载，仅包含图分析。通常图分析与SQL、ML和其他分析相混合。为了支持异构工作负载，开发人员必须在不同的系统之间移动数据，这是数据库碎片化的世界。通过在共享数据上支持多种语言/API，融合数据库解决方案实质上允许用户以所需的方式查看数据！ SQL、图和ML可以协同处理相同的数据。也不需要数据传输或转换。这是巨大的节省。尽管原生图数据库针对图工作负载进行了高度优化，但如果我们考虑异构工作负载端到端管道的性能，融合图数据库实际上可能具有优势。

此外，一些融合数据库解决方案（如IBM Db2 Graph）甚至允许在操作数据库中的原始数据上执行图查询。它带来的额外优势是在不影响大量现有关系型应用程序的情况下具有图查询功能，以及对操作数据的事务更新可以在实时图查询中可见。

融合数据库解决方案的其他优势来自现有的后端数据存储，例如事务支持、访问控制、符合审计和法规要求、时间支持、可扩展性支持、高可用性和灾难恢复支持等。

正如上面所讨论的，每种类型的图解决方案都有其优缺点。选择合适的架构在很大程度上取决于实际应用需求，例如工作负载是仅针对图还是异构的，延迟和吞吐量要求，更新频率，结果的实时性要求等。

#pagebreak()

= 系统设计与实现

本次毕业设计系统名为ArcaneDB，代码，设计文档，以及测试代码都已经开源 #cite("arcanedb")

== 设计目标

数据模型为有向属性图，即用户可以定义点边上属性的Schema，ArcaneDB的定位是图数据库的存储引擎，提供对于点边的读写原语。

ArcaneDB对外提供的接口为：
- GetEdgeIterator -> 获取一个顶点的所有出边的的迭代器。
- GetVertex -> 读取一个顶点
- InsertEdge/UpdateEdge/DeleteEdge -> 插入/更新/删除一条边
- InsertVertex/UpdateVertex/DeleteVertex -> 插入/更新/删除一个点

#indent()ArcaneDB可以作为分布式图数据库的存储层，计算层可以构建在存储层之上，并基于存储层提供的原语实现：
- 图查询语言的解析（OpenCypher，Gremlin）
- 生成分布式执行算子（多跳查询等）

整体目标为实现高性能的单机存储引擎，并提供基本的读写原语。并且在设计与实现的过程中，保持演进的能力，以便将来的拓展。

== 整体架构

#img(
  image("./5.jpg"),
  caption: "ArcaneDB"
) <img5>

整个存储引擎分为4层5个模块，分别是磁盘层，对应LogStore，PageStore，Btree层，事务层，以及图语义层。
下面介绍一下各层中涉及到的模块，以及每一个模块具体的职责：
- 磁盘层：
  - LogStore：日志相关模块，负责写入WAL
  - PageStore：负责存储Btree的节点，也就是Page
- Btree层：
  - BufferPoolManager：负责控制Btree Page的换入换出
  - SchemaManager：负责管理数据的Schema
  - Flusher：负责在后台将修改过的Btree Page刷回到磁盘中
  - Btree：将数据组支撑Btree的形式存储，包含支持多版本语义的叶节点，以及非叶节点。
  - SubTable：和RDBMS中的表模型类似，提供表语义。SubTable的作用是协调多个Btree的读写，如ClusterIndex和SecondaryIndex
- Transaction层：
  - TransactionManager：负责启动事务，以及负责事务上下文的回收
  - TransactionContextOCC：一种基于多版本的乐观并发控制实现。
  - TransactionContext2PL：两阶段封锁并发控制实现。
  - SnapshotManager：负责追踪可以进行无锁快照读事务的快照管理器。
- Graph层：
  - WeightedGraph：一种Schema特化后的图语义层，负责将用户的图语义相关请求转化成非图语义的对子表的请求。

== 关键设计

=== 图划分

由于图结构不像关系表一样结构工整，可以水平划分或者垂直划分来将数据分布到多台机器上，或者分配到不同的NUMA node上以获取更好的multi-core scalability，所以图的划分一直是一个活跃的研究方向。在图存储引擎的视角来看，目前前沿的图存储系统划分防止主要有两种：

1. CSR，以Teseo #cite("de2021teseo") 为代表，将所有数据都放入到一个偏中心化的数据结构中（比如一颗Btree）。好处是一般涉及到全图扫描的Workload在这个数据结构上表现会更好一些。
2. 邻接表，以Sortledton #cite("fuchs2022sortledton")， LiveGraph#cite("zhu2019livegraph") 为代表，将边数据根据顶点进行划分，比如一个顶点的所有出边存到一个数据结构中（Skiplist，EdgeBlock等），然后将顶点到其出边的数据结构地址的映射关系维护成一个内存中的哈希表。这样的好处是利用了图结构的局部性，每个顶点的出边在更新的时候不会影响其他顶点的出边。

在 Sortledton#cite("fuchs2022sortledton")中总结了图处理相关Workload对于图存储结构的需求：
1. 顶点的顺序扫描
2. 一个顶点出边的顺序扫描，并且要求扫描是有序的
3. 随机访问数据的属性，不同算法对于属性访问的需求不同
4. 顶点的随机访问
5. 事务支持

事务的支持其实隐含了快速点查的需求，因为在一些多版本系统的实现上 #cite("diaconu2013hekaton", "larson2011high", "neumann2015fast")，并发控制协议要求先读取这个数据的老版本并做一些修改。

上述需求主要是图计算的需求，然而在一些线上的图数据库工作负载中，还包含了一些其他的工作负载，如：
1. 获取某个顶点的出边数量
2. 根据属性进行过滤，只获取满足条件的点/边
3. 查询某个点是否有到另一个点的出边

综上，我认为选择类似邻接表的结构作为图数据的存储结构是更加符合需求，更加高效的方案。

=== 邻接表

#img(
  image("./6.png"),
  caption: "Adjancency List"
) <img6>

我们称第一步为Seek，第二步为Scan。

Seek邻接表的做法分为两大类：
- 内部分配点边ID，然后返回给用户，通过数组可以直接索引到邻接表头
- 不做编码，通过索引定位到邻接表头。

第一种方式需要用户层，或者我们自己额外维护索引，通过属性定位到点边ID。相较于第二种方式并没有节省一次索引，并且还不灵活。所以不选择第一种方案。

然而这两种方案主要的问题是在访问具体数据的时候都需要先走一个索引，然而这个索引本身的开销可能较大，是O（点的数量），所以将该索引设计成全内存的并不可行（对用户的环境假设较强）。

#img(
  image("./7.png"),
) <img7>

如 @img7 所示，这里解决的思路是通过编码来省略掉定位顶点的索引，我们可以直接通过对PointID进行编码得到具体的PageID，从而定位到BTree的根结点。访问数据的时候，原本的路径是先查找顶点的索引，得到物理地址，然后查找buffer pool看这个数据是否在缓存中。这样无论数据是否在缓存中，都需要两次索引的查找。而修改后，对于在缓存中的数据，只需要查找缓存中的索引，而不在缓存中的数据，则需要额外查找一下物理地址，再进行IO得到具体的数据。

=== Btree

数据组织的形式使用Btree，在现代SSD的加持下，BTree可以提供高性能的读，以及Scan的能力。

传统的Btree的并发控制手段一般是使用latch coupling #cite("graefe2010survey")，即子节点先上锁，然后父节点放锁。对于同一个Page的访问，读写是互斥的。这就导致高并发的情况下，写操作会影响读操作的性能。之前也有一些相关的工作是优化Btree的上锁方式从而提高并发度，比如BwTree #cite("wang2018building", "levandoski2013bw")，BlinkTree #cite("lehman1981efficient")。

在大多数的场景下，读的数量都是远大于写的，并且图存储引擎希望提供高性能的读（Scan），所以我们希望针对读操作来优化。

写不阻塞读是一个比较常见的话题了，在事务的并发控制手段中，多版本并发控制（MVCC） #cite("wu2017empirical") 通过写请求创建新版本的方式来避免读请求被阻塞。一般这里多版本的粒度就是行级。

我们可以使用相似的思路，在写一个Btree Page的时候，创建一个新的副本出来，在上面执行写操作，而读操作则可以直接读取现有的副本。通过CopyOnWrite的方式来避免读请求被阻塞。这样，在一个Btree上的读操作都是永不阻塞的。

对于Btree的SMO操作，我们认为一般是小概率事件，所以会将其和Btree的写操作阻塞，从而简化实现的复杂度。

所以在正常的写操作下，我们只会修改Btree中的一个Page。而在SMO的情况下，可能修改若干个Page。

为了防止读性能受到影响，我们允许SMO和读操作的并发，这里的要求是每次SMO会生成新的PageID，老的Page在失去引用后会被删掉。并且要保证子节点先完成SMO，再去完成父节点的SMO。

SMO作为一个系统级别的事务，为了性能以及简化实现，不为SMO操作记录undo日志，即SMO是redo only的。可以类比innodb的mini transaction #cite("innodb")。

这里希望通过类似BwTree #cite("levandoski2013bw")的方式来优化写入能力，即借用BwTree的Delta的思想。将LSM嵌入Btree的节点中。

这里的设计空间有两块，一个是内存中是否选择用Delta，一个是磁盘中是否选择Delta。其中内存中Delta的思路来源于原始的BwTree，而磁盘中Delta的思路则来自于BwTree的缓存/存储子系统 #cite("levandoski2013llama")。

对于磁盘来说，使用Delta的好处是用读放大换写放大。因为都是log structured，所以是随机读的iops放大换顺序写的带宽。在现代SSD上，随机读性能不差，但是顺序写可以优化SSD内部的GC。也有一些其他的工作是结合SSD的性质去给用户提供更加适合SSD的抽象，比如OpenChannelSSD #cite("picoli2020open")，ZNS #cite("bjorling2021zns") 等，都是暴露了SSD顺序写的特性，进而优化性能。所以一般认为磁盘的Delta是一个比较好的选择，只要iops放大不是太严重就可以。

写盘写入Delta的思路其实在很多系统中都有体现，比如最近比较火的存算分离数据库Aurora #cite("verbitski2017amazon"), Socrates #cite("antonopoulos2019socrates")，是通过写入Log的方式来优化网络带宽。ArkDB #cite("pang2021arkdb") 则是直接借鉴了BwTree的思路来优化写入放大。追溯到早期的单机存储引擎，最知名的就是LSMTree #cite("o1996log") 和LFS #cite("rosenblum1992design") 了。

对于内存来说，使用Delta的好处是写操作可以不用重新写原本的Page，缺点就是读需要Merge-on-read，需要遍历若干个delta来读取数据。一个特别的点在于这个delta的粒度是可以动态控制的，比如一个page是热点写入的情况，我们就可以允许delta多一些，而对于希望优化读性能的场景下，则可以让delta的数量变少，甚至是0个，这样读性能就是最优的。

在任意一种情况下，读写都是相互不阻塞的。在bwtree原始的论文中，允许在同一个page上做无锁的并发写入，虽然lock-free本身性能很高，但是当写入失败的时候，就需要本次写入对应的WAL被abort掉。在高争用的情况下，abort log buffer的开销变得不可忽视(具体问题详见基于BwTree实现的开源项目Sled #cite("sled"))，并且bwtree原始的论文中，每次写入只能prepend一个delta的限制使得读操作的间接跳转变得更加严重。所以这里不选择使用bwtree的乐观协议，而是不允许写写并发，并发的写操作会通过GroupCommit聚合起来，由单线程负责写入数据。这种聚合带来两个好处，一个是批量写入log buffer，降低对log buffer模块的压力，第二个是聚合delta使得一个delta中的数据更多，从而减少delta chain的长度。（photondb #cite("photondb") 是通过partial consolidation来减少delta chain的长度）

内存中的consolidation和磁盘中的consolidation的时机是解耦的，缺点是引入一定的内存空间放大，好处则是允许上述的delta单独处理自己的逻辑（比如consolidation）

=== 事务

这里有两种思路：
1. 单机下用ReadView做SI，提供XA接口做分布式提交。好处是ReadView的并发度更高，但是分布式场景下需要中心化的节点来维护活跃事务的视图。
2. 每个事务分配StartTs和CommitTs。其中Ts的分配最简单就是TSO #cite("peng2010large")，或者用ClockSI #cite("du2013clock")，并且用HLC #cite("kulkarni2014logical") 来避免读等待的问题。好处是分布式事务性能会高一些，但是其中ClockSI只能提供GeneralizedSI #cite("elnikety2005database")，不能提供最新的快照。但是单机情况下会有一些不必要的阻塞。

由于上述的Btree中的数据是不可变的，而分布式事务中不可避免的会出现2PC，就导致intent会先被写进去，后续被提交的时候会再写一次进去，造成写放大。而为了维护global snapshot，我们还必须保证Intent是先被写进去的，这样读者在读到intent的时候才能等待并检查intent的CommitTs，从而判断可见性。

目前希望的是解耦的架构，不太希望事务层去感知数据存储的格式，显式的区分开Transaction Component和Data Component。

这样的话可能会造成一些suboptimal，比如bwtree自己本身不是原地更新的结构，对于多版本的处理会复杂不少。比如一般的mvcc都是在事务提交的时候再写txn ts，需要更新到原本的数据上。在bwtree上要想实现这一点就需要为bwtree的delta加上特殊的commit以及abort标记。并且bwtree的compaction也会和事务的语义耦合起来。

bwtree本身有多版本的语义，和事务相关的语义耦合起来处理复杂度也会高很多。可以参考一下rocksdb的transaction db是怎么做的。

同时考虑到page也不仅仅有bwtree page这一种，所以可能出现对这种page比较好的方法，对另一种page就不太行了。这里我还是更希望去实现一个契合RUM #cite("athanassoulis2016designing") 假设的系统。

和数据耦合程度最小的并发控制方法就是2PL了。所以这里第一版的设计思路是借鉴google spanner #cite("corbett2013spanner")，对于读写事务用2PL来做并发控制。而多版本只用来提供给只读事务。

这里只读事务的ts获取方法参考cicada #cite("lim2017cicada")，每个线程写新版本是通过本地时钟wts，所有线程中wts的最小值作为rts可以为只读事务提供服务。这样只读事务就是完全不阻塞的。

基于2PL可以简化一些实现，比如如果实现的是OCC的话，需要先写intent，做validation，再commit write，这时候如果abort write对于bwtree来说不能覆盖写，处理起来较为复杂。（不希望引入特殊的abort delta，然后让读者去merge，这样会降低读的效率）。

在实现了2PL来处理读写事务之后，遇到的直接问题是2PL本身不感知多版本中的TS，就导致版本链上的ts不是单调递增的。虽然仍然可以保证读写事务的serializable，以及只读事务的一致性快照，但是对于版本的回收等其他细节问题可能会出现一定的影响。

借鉴PG，引入特殊的timestamp。比如0是无效版本，即被abort掉的事务。1是上锁的版本，读者遇到这种版本需要等待。

因为被上锁的版本一定是最新版本，（不允许WW覆盖写），所以abort以及commit的时候不需要写delta以及（page级别的）日志，直接找到对应SortKey的最新版本，更新ts即可。（不过不写日志的话，恢复的时候可能比较困难，无法并行回放page）

为了防止只读事务遇到intent阻塞，这里还是决定追踪所有活跃事务，并且只读事务的read ts只会小于所有活跃事务，从而在遇到intent的时候也会直接跳过。

还有一个点就是，如果提供的是SI，不能简单的选取一个Ts作为读写事务的read ts，因为这时候仍然可能有write ts小于read ts的事务在更新，就导致read ts读到的不是一个一致性的快照。

目前看到的实现（PG，Innodb）都是通过read view来做的。而之前通过实验发现简单的去追踪最小的ts不太靠谱，因为并发度很高。并且在协程的背景下，read view的大小可能是远超core num的，导致read view本身开销较大。

目前来看较为靠谱的方法是用MVOCC。这里就有很多选择了：Hekaton #cite("larson2011high")，Cicada #cite("lim2017cicada") 等。

这里选择魔改一下Hekaton，让他适配到bwtree中。思路如下：
读请求记录读的sortkey，以及读到版本的ts。
写请求会请求写锁，然后写入到btree中，并且带有Locked标识的ts。对于读后写的处理，有几种，要么是locked ts为txn id，读取的时候判断一下。要么是读取的时候看一下有没有锁，有的话就读上锁的版本。要么是写的时候写到缓存中，读可以直接读写缓存
Hekaton的原始做法是直接写到btree中，读请求遇到锁会去txn manager中找事务状态。这里不希望再多引入一个记录事务状态的全局组件。那么遇到intent就只能spin wait。
如果是发起写请求的时候就要执行写入的话，后续其他事务的读很容易被阻塞。所以决定写入的时候先上锁，然后写缓存。
提交的时候，先将intent写入到btree中，然后拿到commit ts。然后用commit ts去验证读集中的元素是否有变化（版本是否相同）。如果不相同，释放写锁，abort。如果相同，则写入commit log，然后在btree上回写commit ts。
这里写先intent，再拿commit ts，是为了保证所有read ts大于commit ts的人，都可以看到intent，从而serial after当前事务。即保证WR依赖。
对于WW依赖，先上锁的人一定commit ts更小。所以也可以保证。
对于RW依赖，读者如果在写者之前提交，没有问题，如果读者在写者之后提交，那么他可以看到intent，并spin wait，直到写者成功。然后会发现两次读到版本不同，则会abort。
至此可以证明这个算法的正确性。如果spin wait的时间过长，则可以考虑引入speculative read #cite("larson2011high")等技术，通过pipeline增强性能。

== 系统实现

在本节中将会按照ArcaneDB的实现过程来描述各个模块的实现细节。

=== 协程框架

由于云上存储相较于本地SSD访问延迟高出一个数量级 #cite("chen2022cloudjump")，所以会导致整个请求的延迟会有大幅度上升，根据little’s law #cite("littlelaw")，延迟乘以QPS为并发数，这个数量远大于机器的核数。所以为了维持高QPS，必须开非常多的线程，导致无论是调度开销，还是线程创建开销都非常高。这时候我们需要异步编程模型，最常见的就是通过回调函数来实现异步，但是这样代码复杂度会上升很多，协程则是解决这个问题的一个很好的办法，可以让用户用同步的方式写异步的代码，在编程复杂度和性能之间有一个很好的平衡。

协程本身是一个比较老的概念，是函数的一种更通用的表达手段，目前其主要的作用是可以让开发者可以利用协程将原本的回调流异步代码变成同步代码，使得开发效率更高，代码维护成本更低。协程本身也根据特性不同有很多种，最主要的一个划分点就是有栈协程和无栈协程。

1. 有栈协程可以简单看作是将内核的线程调度/并发控制等逻辑都搬到了用户态，使得线程上下文切换开销减少。典型的有栈协程就是Go中的Goroutine。
2. 无栈协程目前主流的实现方式都是通过感知异步代码的挂起点，将代码转化成状态机的模式，挂起点前后需要用到的变量会被编译器所捕获并存放到协程帧上。

目前由于C++20的协程库不太成熟，以及本人在开发时对于无栈协程不太熟悉，故选择了有栈协程。有栈协程一个比较好的实现就是brpc #cite("brpc")，brpc实现了一套用户态的线程，对应的线程调度，以及同步原语，称做bthread。
这里简单介绍一下bthread提供的API：

```cpp
// 用来在后台启动一个bthread
int bthread_start_background(bthread_t tid, const bthread_attr_t* attr, void *(*fn)(void*), void* args);
// 等待一个bthread结束
int bthread_join(bthread_t bt, void** bthread_return);
// 唤醒一个等待在条件变量上的bthread
int bthread_cond_signal(bthread_cond_t* cond);
// 唤醒所有等待在条件变量上的bthread
int bthread_cond_broadcast(bthread_cond_t* cond);
// 等待在条件变量上
int bthread_cond_wait(bthread_cond_t* __restrict cond,
                             bthread_mutex_t* __restrict mutex);
// 对一个mutex上锁
int bthread_mutex_lock(bthread_mutex_t* mutex);
// 释放mutex的锁
int bthread_mutex_unlock(bthread_mutex_t* mutex);
```

可以看到bthread提供的API和pthread库提供的是类似的。由于是纯C风格，所以在实际使用的过程中还需要一定的封装，比如提供一些现代C++的同步原语，如 `std::future` 等。

=== Env

ArcaneDB针对云端一体进行设计，不对底层存储底座有所假设，希望使得ArcaneDB可以灵活的部署到任意地方，如云上环境（S3，EBS），或者单机环境（Ext4），再或者一些私有化的部署（如NAS）。所以一个比较通用的写入接口就是AppendOnly的写入方式。
所以这里的实现方式是抽象一个Env出来，然后在部署到各个环境的时候，只需要实现Env就可以直接将ArcaneDB部署到对应的环境中。
Env主要提供的API为：

```cpp
Status NewRandomAccessFile(const std::string& fname,
                                     std::unique_ptr<RandomAccessFile>* result,
                                     const EnvOptions& options);
                                     
Status NewWritableFile(const std::string& fname,
                                 std::unique_ptr<WritableFile>* result,
                                 const EnvOptions& options);
                                 
class RandomAccessFile {
public:
    Status Read(uint64_t offset, size_t n, Slice* result, char* scratch) const;
};

class WritableFile {
public:
    Status Append(const Slice &data);
    Status Flush();
    Status Sync();
};
```

即提供的主要功能为顺序写入，以及随机读取。

=== PageStore

根据关键设计中所述，PageStore提供的是存储Btree Page的能力。同时我希望借鉴BwTree的思路，通过写入Delta的方式来减少Btree的写入放大。这时这里的PageStore的接口就有两种选择，即是否需要感知Delta的语义。

- 如果PageStore不感知Delta的语义，那么PageStore本身的实现就变得较为简单，而Delta则需要由上层处理。
- 如果PageStore感知Delta语义，好处是PageStore可以根据DeltaPage和BasePage的更新频率的不同，为数据进行冷热分离，从而降低写入放大。

这里我选择让PageStore感知上层的Delta语义，并且借鉴了微软的LLAMA #cite("levandoski2013llama") 中提供的接口，PageStore的接口如下：

```cpp
// 写入一个全量的BasePage
Status UpdateReplacement(const PageIdType &page_id, const WriteOptions &opts, const std::string_view &data);
// 写入一个DeltaPage
Status UpdateDelta(const PageIdType &page_id, const WriteOptions &opts, const std::string_view &data);
// 逻辑的删除一个Page
Status DeletePage(const PageIdType &page_id, const WriteOptions &opts);
// 读取一个逻辑Page下所有的BasePage和DeltaPage
Status ReadPage(const PageIdType &page_id, const ReadOptions &opts, std::vector<RawPage> *pages);
```

提供一个高性能的基于AppendOnly Env的PageStore实际上是相对困难的一件事，在开发时间有限的情况下，我这里选择复用已有的系统来加速开发，并且可以在后续发现瓶颈的时候再去实现一个定制化的PageStore。

目前的选择是使用LevelDB #cite("leveldb") 作为存储Page的系统，PageStore中包含三个组件，分别是BasePage Store，用来存储BasePage，DeltaPage Store，用来存储DeltaPage，以及PageIndex，用来存储逻辑PageId到其BasePage以及DeltaPage的映射。由于LevelDB不提供类似RocksDB #cite("rocksdb") 的ColumnFamily机制，因为不同类型数据的大小和更新时间不同，为了防止不同类型的数据相互影响，我选择将三个组件各由一个LevelDB构建。

#img(
  image("./8.jpg"),
) <img8>

如@img8 所示，这里三个组件的交互关系有两种选择：

- 一种是PageIndex记录了所有的Base和Delta，Base和Delta上没有相互联系。这样的好处是在读取PageIndex之后可以发起MultiRead来并发的读取所有的Base和Delta，降低读取延迟。
- 一种是PageIndex只记录了最新的Page，每个Page会有一个指针指向前一个Page。这样在读取的时候需要一个一个读Page直到读到最后一个BasePage，读性能会较差。好处是PageIndex只需要存储一个指针，空间较小，并且每次更新的时候不需要覆盖写，不会产生写放大。

这里因为PageIndex本身的数据就偏小，写放大本身并不大，然而读取性能的裂化会使得读取Page Cache Miss的时候无法确定要执行多少次IO，进而影响p99延迟。因为对于一个存储系统来说，性能的稳定性至关重要。所以这里选择了第一种方案，让PageIndex记录全量的索引。

对于写入的流程来说，每次会先生成一个以PageId为前缀的UUID，然后根据写入类型将数据写入到对应的KVS中。然后读取老的PageIndex，更新索引，并将其写回到PageIndex中。之所以要生成以PageId为前缀的UUID是因为每次读取Page都会将其所有的DeltaPage都读上去，所以聚合DeltaPage，使其尽可能的放入到一个SST/DataBlock中，进而可以提高读取的局部性。

=== LogStore

WAL，全称是 Write-Ahead Logging，是许多数据库系统（例如 PostgreSQL、SQLite 等）中用于保证数据一致性和恢复能力的关键技术。

在进行任何修改数据库状态的操作（例如更新、插入或删除数据）之前，数据库系统会先将这些操作写入到日志中。这些日志会在操作实际生效之前写入到持久化存储介质（通常是磁盘）中，这就是“预写日志”的概念。

WAL的主要优点包括：
1. 恢复能力：由于所有的修改操作都先写入到了日志中，因此即使在数据库系统崩溃或者其他错误情况下，数据库系统也能通过回放日志来恢复到一个一致的状态。
2. 数据一致性：WAL也是数据库事务的一个关键组成部分。在一个事务中的所有操作都被写入到日志中，并且在事务提交时一起生效。因此，WAL可以保证即使在并发或故障的情况下，数据库状态也是一致的。
3. 性能：与直接将每个操作写入到数据库文件相比，将操作写入到WAL通常可以提高性能。这是因为WAL的写入可以是顺序的，并且可以批量进行。另外，WAL的写入通常可以异步进行，不需要等待每个操作的持久化完成。

然而，WAL也有一些潜在的缺点，例如可能会增加I/O负载，因为每个操作需要写两次（一次到日志，一次到数据库文件）。此外，WAL也需要定期的管理和维护，例如需要定期的日志清理和日志压缩。

日志组件在数据库系统中一直是一个广受研究的组件，因为他影响了两个非常关键的位置：
1. 一个是日志写入的延迟，因为每个事务在提交返回给用户之前，都需要将事务对应的日志落盘，所以日志组件的写入延迟会影响前台事务的提交延迟，然而为了优化吞吐/减少IOPS，日志组件无法每次同步将数据落盘，需要有一些GroupCommit的逻辑来优化性能。最近的一些研究中，为了减少前台写入的延迟，会将日志先写入到PMem中，然后后台dump到成本较低的持久化存储中。
2. 第二个则是数据库的恢复时间，这里涉及到整个数据库系统多个组件之间的联动，日志组件需要能够快速解析日志，将其重放到对应的Page上，上层也需要及时控制Checkpoint的时间，防止日志过长影响恢复时间。

实际上除此之外，日志组件还是一个中心化的组件，因为目前绝大多数数据库系统沿用了ARIES #cite("mohan1992aries") 中的LSN概念，有一个中心化的日志组件负责分发LSN，并负责将数据落盘。在现在的多核环境下，全局共享的数据结构会导致并发冲突较高，工作线程会不断因为无法抢到锁进而陷入内核态休眠，导致上下文切换开销过高。在MySQL8.0中，就为了解决这个问题，引入了无锁的日志组件 #cite("mysql_lockfree_wal")，并取得了很好的性能收益。

这里ArcaneDB实现了一套高性能的无锁日志组件，接口如下：

```cpp
class LogReader {
public:
  bool HasNext() noexcept = 0;
  
  LsnType GetNextLogRecord(std::string *bytes) noexcept = 0;
};

class LogStore {
public:
    // 写入若干条日志，并返回其对应的lsn
    void AppendLogRecord(std::vector<std::string> log_records, std::vector<LSN> *result>);
    // 获取当前持久化的LSN
    LSN GetPersistentLSN();
    // 等待直到lsn的数据都被持久化
    void WaitForPersist(LSN lsn);
};
```

LogStore包含的组件主要是一个轮转写入的日志文件，若干个LogSegment用来作为LogBuffer使用，以及一个后台线程负责将LogSegment写入到磁盘中，并更新PersistentLSN。

LogSegment是实现无锁WAL的关键点，其核心的结构为一个 `std::atomic<uint64_t>`，用来控制当前Segment的状态。

每个Segment的状态有四种：
1. Free，表示当前Segment没有被使用
2. Open，表示当前Segment正在被写入
3. Seal，表示当前Segment不再可以被写入，等待所有并发的写者离开临界区后就可以进行IO了
4. IO，表示当前Segment正在等待或进行IO中

控制位的格式如下：
- 最高两位，存储当前Segment的状态
- 48到62位，共14位存储当前Segment的并发写者的数量
- 低48位，用来存储当前Segment下一次要分发的LSN是多少

每个Segment的状态转移如@img9

#img(
  image("./9.jpg"),
) <img9>

然后阐述一下每个状态转移的时机，以及状态转移的动作：
- Free -> Open：
  - 最开始的第一个LogSegment是Open的。
  - 当一个线程在尝试将前一个LogSegment转化为Seal并成功的时候，他就负责将下一个Segment置为Open
- Open -> Seal：
  - 当一个前台的线程发现当前LogSegment空间不够写入本次要写入的日志的时候，就会将当前的LogSegment转化为Seal
  - 当一个后台的线程等待需要IO的Segment等待的时间过久的时候，他就会主动Seal一个LogSegment
- Seal -> IO：
  - 当一个写者结束写入的时候，他会检查当前LogSegment是否已经Seal了，如果是，并且自己是最后一个写者，那么他负责将当前Segment置为IO
- IO -> Free：
  - 后台的线程会按顺序扫描LogSegment，并等待其状态置为IO，当其状态为IO的时候，进行IO，将数据写入到文件中，并将该Segment的状态设置为Free

写入日志的代码如下：

```cpp
void AppendLogRecord(std::vector<std::string> log_records, std::vector<LSN> *result) {
    // 计算本次写入的日志占用的空间
    size_t total_size = LogRecord::kHeaderSize * log_records.size();
    for (const auto &record : log_records) {
        total_size += record.size();
    }
    do {
        LogSegment *segment = GetCurrentLogSegment();
        auto [succeed, should_seal, raw_lsn] = 
            segment->TryReserveLogBuffer(total_size);
        if (succeed) {
            SerializeLogRecords(log_records, result);
            segment->OnWriterExit();
            return;
        }
        if (should_seal && SealAndOpen(segment)) {
            continue;
        }
        backoff.Sleep();
    } while (true);
}
bool SealAndOpen(LogSegment *log_segment) {
    std::optional<LSN> lsn = log_segment->TrySealLogSegment();
    if (!lsn.has_value()) {
        return false;
    }
    OpenNewLogSegment(lsn.value());
    return true;
}
void OpenNewLogSegment(LSN start_lsn) {
    AdvanceSegmentIndex();
    while (!GetCurrentLogSegment()->IsFree()) {
        backoff.Sleep();
    }
    GetCurrentLogSegment()->OpenLogSegment(start_lsn);
}

std::tuple<bool, bool, LSN> TryReserveLogBuffer(size_t length) {
    auto current_control_bits = control_bits_.load(std::memory_order_acquire);
    uint64_t new_control_bits;
    LSN lsn;
    do {
        if (!IsOpen(current_control_bits)) {
            return {false, false, kInvalidLSN};
        }
        lsn = GetLSN(current_control_bits);
        if (lsn + length > size_) {
            // seal current segment
            return {false, true, kInvalidLSN};
        }
        int current_writers = GetWriterNum(current_control_bits);
        if (current_writers + 1 > kMaxWriterNum) {
            // too much writers, backoff and sleep
            return {false, false, kInvalidLSN};
        }
        new_control_bits = IncrWriterNum(current_control_bits);
        new_control_bits = BumpLSN(new_control_bits, length);
    } while (!control_bits_.compare_exchange_weak(
        current_control_bits, new_control_bits, std::memory_order_acq_rel));
    return {true, false, lsn};
}
void OnWriterExit() {
    uint64_t current_control_bits =
        control_bits_.load(std::memory_order_acquire);
    uint64_t new_control_bits;
    bool is_sealed;
    bool is_last_writer;
    do {
      is_sealed = IsSeal(current_control_bits);
      new_control_bits = DecrWriterNum(current_control_bits);
      is_last_writer = GetWriterNum(new_control_bits) == 0;
    } while (!control_bits_.compare_exchange_weak(
        current_control_bits, new_control_bits, std::memory_order_acq_rel));
    if (is_last_writer && is_sealed) {
      if (CasState_(LogSegmentState::kSeal, LogSegmentState::kIo)) {
        // notify io thread
        waiter_.NotifyAll();
      }
    }
}
std::optional<LSN> TrySealLogSegment() {
    uint64_t current_control_bits =
        control_bits_.load(std::memory_order_acquire);
    uint64_t new_control_bits;
    LSN new_lsn;
    bool should_schedule_io_task = false;
    do {
      if (!IsOpen(current_control_bits)) {
        return std::nullopt;
      }
      if (GetLSN(current_control_bits) == 0) {
        // don't seal when segment is empty
        return std::nullopt;
      }
      if (GetWriterNum(current_control_bits) == 0) {
        should_schedule_io_task = true;
      }
      new_control_bits =
          SetState(current_control_bits, LogSegmentState::kSeal);
      new_lsn = static_cast<LSN>(GetLSN(new_control_bits));
    } while (!control_bits_.compare_exchange_weak(
        current_control_bits, new_control_bits, std::memory_order_acq_rel));
    if (should_schedule_io_task) {
      if (CasState(LogSegmentState::kSeal, LogSegmentState::kIo)) {
        // notify io thread
        waiter_.NotifyAll();
      }
    }
    valid_size_ = new_lsn;
    return new_lsn + start_lsn_;
  }
```

TryReserveLogBuffer的作用就是在当前LogSegment中预留一段空间出来，即去CAS控制位，增加当前并发写者的数量，以及增加LSN的偏移量，并在检测到不满足条件的时候及时返回。

如果在预留空间的时候成功了，那么写者就会将日志序列化到预留的空间中，并在退出的时候调用OnWriterExit，作用是减少并发写者的数量，并在发现自己是最后一个写者，且当前LogSegment已经被Seal的时候，将其状态设为IO，并唤醒后台线程将当前Segment落盘。

如果在预留空间的时候失败了，原因可能是空间不足，或者是并发的写者太多了。如果是因为写者过多，当前写者会回退，并在睡眠一段时间过后再重新去预留空间。如果是因为空间不足，当前写者会尝试将状态置为Seal，并打开下一个LogSegment。

在Seal LogSegment的时候，也需要检查，如果写者的数量为0，那么我们就需要负责将LogSegment的状态置为IO。

=== BufferPoolManager

缓冲区管理器，属于一个在数据库系统中比较关键的组件，为了支持数据量超过内存容量，并且避免操作系统自行进行换页 #cite("crotty2022you")，很多数据库都选择自己实现一个缓冲区管理器。

一般的缓冲区管理器都是通过Page这个概念来为上层屏蔽磁盘中的细节，对外提供的接口是：
```cpp
class Cache {
    HandleHolder Insert(const std::string_view &key, void *value, size_t charge,
                        void (*deleter)(const std::string_view &key, void *value));
    HandleHolder Lookup(const std::string_view &key);
}
```
即插入一个缓存项，以及查找一项是否在缓存中。ArcaneDB提供的接口也是如此。

在前人的研究中 #cite("stonebraker2008oltp")，有发现实际上数据库系统有很多的开销都浪费在了缓冲区管理器中，因为每次访问数据都涉及到访问一个磁盘上的Page，就都需要走到这个组件中查询是否这个Page在缓存中，如果不在则执行IO，然后再将其插入到缓存中这种逻辑。所以有很多工作都在希望尝试减少缓冲区管理器的开销，比如近几年的Umbra #cite("neumann2020umbra")，LeanStore #cite("leis2018leanstore")，通过pointer swizzling #cite("white1994pointer") 的方式来实现去中心化的缓冲区管理器。除此之外，在早期大家发现了磁盘数据库的开销严重，再加上内存价格下降，引发了内存数据库的研发热潮，而为了投入到生产使用，内存数据库的领域也希望支持存储数据规模超出内存的场景，VoltDB的AntiCache #cite("debrabant2013anti") 和微软的Siberia #cite("eldawy2014trekking") 从这个角度出发，为他们的纯内存数据库支持了Larger-than-memory的能力，其核心思路放弃了Page的概念，而是以Tuple/Row为粒度，去找到运行时比较访问频率比较低的Tuple，然后将其写入到磁盘中。

ArcaneDB对于缓冲区管理器实现的较为朴素，就是常见的LRU链表 + 哈希表。虽然数据结构本身比较简单，不过这里还是存在一些工程上的优化的：
- 为了减缓并发冲突，会给缓冲区管理器做一下分片，根据PageID哈希到对应的分片，然后执行操作即可。
- 为了防止一些扫描的场景污染LRU链表，会对LRU链表分区，新加入的Page会先加入到old-list，访问频率变高之后才会加入到yong-list。
- 通过intrusive-linked-list来减少内存分配
- 如果发生了CacheMiss，通过维护当前缓存项的状态来避免多线程并发的读取相同的Page，避免对磁盘的IOPS放大。

这里简述一下关键接口GetPage的代码：
```cpp
Status BufferPool::GetPage(const std::string_view &page_id, PageHolder *page_holder) {
    HandleHolder handle = cache_->Lookup(page_id);
    if (handle) {
        // cache hit
        *page_holder = PageHolder(std::move(handle));
        return Status::Ok();
    }
    // load group保证同一时间只会有一个人调用callback
    Status s = load_group_.Do(
        page_id, &handle,
        [&](const std::string_view &key, HandleHolder *value) {
            HandleHolder handle = cache_->Lookup(page_id);
            if (handle) {
                // cache hit
                *value = std::move(handle);
                return Status::Ok();
            }
            auto page = std::make_unique<BtreePage>(key);
            std::vector<PageStore::RawPage> pages;
            auto s = page_store_->ReadPage(key, &pages);
            if (s.ok()) {
                s = page->Deserialize(pages);
            }
            if (!s.ok() && !s.IsNotFound()) {
                return s;
            }
            handle = cache_->Insert(
                key, page.get(), page.GetCharge(), &PageDeleter);
            *value = std::move(handle);
            page.release();
            return Status::Ok();
        });
    if (!s.ok()) {
        return s;
    }
    *page_holder = PageHolder(std::move(handle));
    return Status::Ok();
}
```

=== Type Subsystem

类型子系统也属于数据库系统中一个比较常见的组件了，在一些开源的RDBMS中都可以看到他们会有一套管理用户Schema，处理不同类型数据写入的系统。

目前ArcaneDB支持6种类型，分别为int32_t，int64_t，float，double，string，bool。用户可以通过创建自己的Schema来决定每一行中要存储的数据的类型，以及每一个Btree中的排序键是什么。

类型系统以及子表语义使得ArcaneDB拥有更大的灵活性，比如用户可以指定Btree的排序键为一条边插入的时间，而非每一条边的顶点ID和顶点类型，这样用户可以根据插入时间进行查询，如最近3天的浏览历史等。

每一个Schema存储了从ColumnID到Column的映射，用户可以根据ColumnID获取一个特定的Column，然后读取该Column的类型等信息。

ArcaneDB有一个全局的实例去管理所有的Schema，一般的使用场景是，用户首先根据表中的元信息获取SchemaID，然后根据SchemaID的到对应的Schema，在读取数据的时候就用Schema去解析对应的一行数据，就可以取出指定位置的数据了。

=== Tuple/Row

这一节会介绍一下行的概念，行就是存储到数据库中的最小单位，用户写入的点/边都会被转化为一行数据存储到Btree中。有的系统也称Row为Tuple。

Tuple在内存中的格式和在磁盘上的格式对于系统的性能影响较为关键，因为每一次写入和读取都涉及到对于Tuple的读取，并且每次将数据从磁盘中读出/写回到磁盘的时候，也涉及到Tuple的序列化/反序列化。

一些内存中结构化的数据结构，如现在常见的Protobuf/Thrift，数据在内存中表示就是一个简单的结构体，并提供一系列的序列化方法，使用起来较为简便。但是其缺点是内存结构较为松散，比如访问一个字符串需要额外的跳转，并且序列化/反序列化开销较大，所以适合一些元数据的存储，并不适合存储数据库中常用的Tuple。

现在开源的一些常用的数据库都有自己的格式，其目的就是为了提高性能，比如现在最常用的MySQL的存储引擎innodb，就有多种Tuple的格式。这里简单介绍一下Compact格式：

#img(
  image("./10.png"),
) <img10>

Compact格式将所有的数据压缩到一起，故称为Compact格式，第一个区域为变长字段的长度，第二个区域用来标识哪一列数据是NULL，第三个区域用来记录一些元信息，比如当前Tuple是否被删除掉等，第四个区域用来记录真实的数据。这种格式的好处就是内存紧密，磁盘和内存中是相同的格式，无需额外的反序列化/序列化操作，从磁盘中读上来就可以用，并且由于所有数据都在一起，缓存命中率比较友好，经过实测发现，Compact格式的数据相较于用结构体来存储数据，IPC可以有显著的提高。

Tuple另一个比较常用的地方就是用来排序了，因为是存储到Btree中，所以每次在搜索的时候都涉及到Tuple的大小比较，Innodb中的实现，是按照Column的顺序一个一个取出来做比较，这种做法较为直观，但是缺点是每次读取一个Column的时间相较于排序操作本身占的比例还是比较高的。

在MyRocks #cite("matsunobu2020myrocks") 中，Facebook的工程师们将RocksDB对接上了MySQL，从而可以利用一些RocksDB的特性来降低成本。在对接的过程中，由于RocksDB会根据他的Key做排序，而MySQL提供的是表语义，其排序键可能有多个，所以在MyRocks中提出了一种新的格式，用来将多个用于排序的列编码成一种特殊的格式，这种格式可以直接用memcmp进行比较，并且不需要每次都将其中的列取出，极大程度的节省了CPU，这种格式叫做MemComparable格式。

ArcaneDB中融合了这些格式的优点，选择了一种既可以快速比较，也拥有紧密内存结构，可以快速读取的数据格式。

#img(
  image("./11.jpg"),
) <img11>

最开始有2byte的total length，用来将row快速的转化为slice。接着是2byte的排序键的长度，用来指导后面SortKey的大小。SortKey就是上面提到的MemComparable的格式。后面的Columns中是不需要进行排序的列，对于定长类型，会原地存储对应的数据，而对于变长类型，则会存储一个长度和偏移量，用来指向VarLen Area中的位置，而真正的数据则会存到VarLen Area中。

提供的接口如下：
```cpp
class Row : public RowConcept<Row> {
public:
    std::string_view as_slice();
    Status GetProp(ColumnID id, ValueResult *value, const Schema *schema);
    SortKeysRef() GetSortKeys();
    static Status Serialize(const ValueRefVec &value_ref_vec,
                            util::BufWriter *buf_writer,
                            const Schema *schema);
}
```

=== Page

本节会介绍ArcaneDB中一个非常关键的模块Page，Page有两种类型，一种是InternalPage，一种是LeafPage。其中InternalPage中存储的是LeafPage的PageID，以及对应区域的排序键。而LeafPage则存储具体的数据。

ArcaneDB针对不同类型的Page的访问模式，设计了不同的并发控制手段以及更新策略。

对于InternalPage来说，其更新较为低频，因为只有在进行SMO的时候才会涉及到对于InternalPage的修改，而读取则较为高频，所以在并发控制手段上选择Copy On Write，这样读者永远不会阻塞，对于读更加友好。而在更新的策略上选择原地更新，避免间接访存，优化读取性能。

对应LeafPage来说，更新较为高频，所以这里选择BwTree的思路来优化更新，即每次更新都会写入一段Delta，在Delta链比较长的时候会进行Compaction，将其合并为一个比较大的LeafPage。

#img(
  image("./12.png"),
) <img12>

如@img12 所示，读者可以直接定位到Delta去读，是无锁的。而和BwTree有一定区别的点在于，为了避免乐观写入在失败的时候需要Abort带来的复杂度和开销，ArcaneDB选择的是悲观写入，即同一时间只能有一个写者写入，其他的写者会等待在锁上。为了解决热点写入的问题，通过GroupCommit来增加热点写入的性能。这里写操作为悲观模型在协程的背景下开销较小，因为不会出现一个持锁的线程突然被换出，基于锁的并发控制的风险被进一步降低 #cite("faleiro2017latch")，所以这里认为悲观写入是一个更优的选择。

BtreePage提供的接口如下：
```cpp
class VersionedBtreePage : public PageConcept<VersionedBtreePage> {
public:
    Status SetRow(const Row &row, TxnTs write_ts, const Options &opts, 
                  WriteInfo *info);
    Status DeleteRow(SortKeysRef sort_key, TxnTs write_ts, 
                     const Options &opts, WriteInfo *info);
    Status GetRow(SortKeysRef sort_key, TxnTs read_ts,
                  const Options &opts, RowView *view);
    void SetTs(SortKeysRef sort_key, TxnTs target_ts,
               const Options &opts, WriteInfo *info);
    Status GetChildPageID(const Options &opts, SortKeysRef sort_key,
                          InternalRowView *view);
    Status Split(const Options &opts, SortKeysRef old_sort_key,
                 std::vector<InternalRow> new_internal_rows);
    RowIterator GetRowIterator();
}
```

一次BwTreePage的写入流程：
1. 首先根据本次的写入类型，构造一个DeltaNode出来，并将数据放入到DeltaNode中
2. 将本次变更的日志写入到LogStore中，并得到本次变更对应的LSN
3. 将本次DeltaNode的指针指向BwTreePage中保存的最新的DeltaNode
4. 将BwTreePage中的DeltaNode指针指向本次的DeltaNode
5. 根据条件判断是否需要触发一次Compaction

一次BwTreePage的读取流程：
1. 读取到BwTreePage中最新的DeltaNode指针
2. 进入到DeltaNode中进行二分查找，尝试读取对应的版本
3. 如果读取成功，返回。
4. 如果在DeltaNode中没找到该排序键，则读取下一个DeltaNode，并回到步骤二
5. 如果下一个DeltaNode为空，则返回NotFound

DeltaNode需要能够提供快速的多版本读取的能力，为了避免旧版本影响到最新版本的读取，这里选择将旧版本和最新版本分离存储。同时因为这里选择了通用的Schema，可能每一个Row的排序键都有不同的长度，所以这里选择额外引入了一个间接的Offset来解决这个问题。

DeltaNode的格式如@img13

#img(
  image("./13.jpg"),
) <img13>

每个DeltaNode中有四个数据结构，分别是：
1. 一个Entry的数组，用来记录最新版本的数据。每个Entry中有control_bit和write_ts，其中control_bit的最高位代表当前版本是否是一个删除版本，低31位代表当前版本在Data Buffer中的偏移量。
2. Data Buffer，用来存储最新版本的数据。
3. 一个OldVersion Entry的数组套链表，每一个逻辑上相同的版本都可能存在多个老版本。
4. Version Data Buffer，用来存储老版本的数据。

这种分离的好处是老版本不会影响读取最新版本的缓存命中率，并且由于Entry本身可以被压缩的比较小，一个Cache line中可以存储多个Entry的控制位，这样在二分的时候不会受到额外间接层的影响。一个后续可以做的优化就是区分排序键是否位定长类型，如果是的话就可以去除控制位，然后将排序键和非排序键分离，这样可以做到只读取排序键就可以定位一个数据，进而增加了缓存友好性。

在进行Compaction的时候，ArcaneDB会遍历DeltaNode中的所有数据，并将其加入到一个map中来收集所有的旧版本，对于一些不在被需要的旧版本，如被Abort掉的数据，或者根据可见性判断不可能再被读到的数据，就会被丢掉。最后会在map中区分来最新版本和老版本，并将其重新序列化为上述DeltaNode的格式。

Compaction是一个用来控制写放大和读放大的关键策略点，目前ArcaneDB的策略较为简单，这里用代码简述一下Compaction的流程：

```cpp
void VersionedBwTreePage::Compaction(VersionedDeltaNode *current_ptr) {
    if (current_ptr < kBwTreeDeltaChainLength) {
        return;
    }
    VersionedDeltaNodeBuilder builder;
    builder.AddDeltaNode(current_ptr);
    auto current = current_ptr->GetPrevious();
    while (current != nullptr && 
        (current->GetSize() == 1 || 
         builder.GetRowSize() * kBwTreeCompactionFactor > current->GetSize())) {
        builder.AddDeltaNode(current);
        current = current->GetPrevious();
    }
    auto new_node = builder.GenerateDeltaNode();
    new_node->SetPrevious(current);
    UpdatePtr(new_node);
}
```

ArcaneDB只有在Delta链过长的时候才会触发Compaction。如果当前节点只有一行数据，我们会强行把它加入到Compaction集合中。如果Compaction集合中的数据大小乘上Factor超过了下一个DeltaNode的大小，我们就会将下一个DeltaNode加入到Compaction集合中。目前的CompactionFactor设置为2，即每一次新生成的DeltaNode都不会超过前一个DeltaNode的1/2。

BwTree的一个好处就是可调整性比较强，比如某些Page是写入热点，我们就可以动态增加DeltaChain的阈值，以及增加CompactionFactor，用来减少写放大。而如果某些Page是读取热点，就可以减少DeltaChain的阈值，加速Compaction，用来减少读放大。

因为BwTree可以看做是Btree内部的LSMTree，所以他可以在Page内部采用LSMTree的策略去做Compaction来平衡读写放大，而在整棵树的角度来看，他还是Btree，所以他可以在树结构上采用Btree的策略去做分裂合并来平衡读写空间放大。所以我认为BwTree是最适合RUM #cite("athanassoulis2016designing") 假设的数据结构，可以根据工作负载动态变化，随意调整。

=== Versioned Btree

VersionedBtree负责组织不同类型的Page，对外提供的接口和Page基本类似，核心逻辑就是负责执行Btree的搜索操作，以及负责触发SMO。

由于ArcaneDB组织Btree的方式是一个点的出边为一个Btree，所以相对于RDBMS来说Btree的数量更多，但是规模更小，所以SMO操作也更为低频。为了简化处理，ArcaneDB的SMO操作会与写者互斥，但不会与读者互斥。SMO操作的流程如下：
1. 写入操作发现某个Page的大小超过阈值，触发一次后台的SMO操作
2. 后台线程会先获取写锁，用来阻塞所有前台的写者
3. 定位到需要进行SMO的位置，执行分裂。为了不阻塞读者，SMO的方式有两种，一种是BlinkTree的思路，维护RightLink，这样并发的读者在读到SMO的中间状态时，可以根据RightLink移动到下一个Page中。另一种则是放弃原地修改，比如Split不是去生成一个新的Page，另一个Page原地修改，而是生成两个新的Page，然后再将两个新Page的ID注册到InternalPage中。这样无论读者读到前后哪个Page都可以正常处理数据。
4. SMO结束后，释放写锁

SMO操作一个复杂的点在于如果在执行SMO的时候，系统重启，这时候Btree的状态就是不一致的，我们需要有一些手段维护Btree的结构一致性。一种解法是ARIES-IM #cite("mohan1992ariesim")，在Btree中维护一些特殊的标记，用来保证在未完成的SMO操作日志后不会有依赖Btree结构一致性的操作，这样未完成的SMO操作就可以像正常操作一样被undo。另一种则是Nested Transaction #cite("rothermel1989aries")，Innodb中使用了这种解决方法，通过引入MiniTxn的概念，来将整个SMO操作都放入到一个MiniTxn中，这样整个SMO操作就是原子的，未完成的SMO操作不需要被Undo，也不会被刷入到磁盘中。ArcaneDB就借鉴了Innodb的方法，通过MiniTxn来解决这个问题。

=== 2PhaseLocking

通过两阶段封锁来实现可串行化事务在不考虑扫描的情况下较为简单，如果考虑一些带谓词的扫描，就需要引入Gap Lock/Next key locks #cite("mysql_lock") 等手段来保证可串行化事务了。

目前ArcaneDB不打算支持带扫描的读写事务，只支持点读点写的读写事务，所以暂时不需要考虑谓词锁等问题。

两阶段封锁整体来说实现较为简单，需要维护一个全局的锁表，在读取的时候，获取读锁，在写入的时候，获取写锁。在事务提交的时候会释放所有的锁。

为了避免死锁，一种常见的手段是死锁检测，检查等待图中是否构成环，如果有，则需要终止环上的事务。ArcaneDB的实现较为简单，上锁的时候会在锁上记录事务TS，在获取锁的时候，如果发现当前事务的事务TS小于锁上事务的TS，就会Abort，并重新开启一个事务。

为了避免锁表的并发冲突较高，同样会按照锁的Key为锁表进行分区。锁的Key是由一个子表的Key和Tuple上的排序键拼接而成的。

=== MVOCC

ArcaneDB还支持了一种基于多版本的乐观并发控制算法，用来支持可串行化事务，本节会介绍一下该算法的细节。

==== 读写事务

- 事务开始时，会获取一个Timestamp，是全局递增的，称为BeginTS。
- 事务执行的过程中：
  - 对于读请求，会使用BeginTs来读取数据
    - 会先查找写缓存中是否存在，如果存在则返回写缓存中的数据
    - 如果不存在，则读取Btree，并将读到的数据记录到读缓存中
  - 对于写请求
    - 会首先获取该数据的写锁，为了提前探测到写写冲突。
    - 然后将写的数据加入到写缓存中
- 事务提交时：
  - 写入一条事务开始的日志
  - 将写缓存中的所有数据写入到Btree中，这时候数据的TS为LockedTs，代表当前数据被锁住了。
    - 对于并发的读者可能读到LockedTs，则他应该重试读取，或类似Hekaton一样进行预测读
  - 再次获取一个Timestamp，是全局递增的，称为CommitTS
  - 用CommitTS，对读缓存中的所有元素再次执行读取，如果读到的TS和记录在读缓存中的TS不同，说明出现冲突，终止当前事务
  - 验证成功，写入Commit日志
  - 对于刚才写入到Btree中的所有数据，将其TS设置为CommitTS。这里的CommitTS就是当前事务在串行化顺序中的位置。
  - 等待所有写入的日志都落盘
  - 释放写锁
  - 返回用户

这里可以简单证明一下这个算法的正确性：
- 对于WW依赖，会通过锁表来解决，两个事务如果会有并发的对同一行有写操作，则一个会被Abort。
- 对于RW依赖，即一个事务A先读取，另一个事务B再写入，需要保证事务A在串行化顺序中会被排在事务B之前。因为CommitTS代表了事务的顺序，所以只需要保证CommitTS A小于CommitTS B。如果CommitTS A大于CommitTS B的话，那么事务A一定在B写入Intent之后才获取CommitTS，那么A在验证阶段一定可以看到事务B的Intent，在B提交后A会发现他读取的TS和第一次读到的TS不同，则会Abort。
- 对于WR依赖，即一个事务A先写入，后续另一个事务B读到了他的写入，那么事务B的CommitTS一定大于事务A的CommitTS。这是通过多版本保证的，即事务A会通过BeginTS读取数据，如果读到了事务B的写入，那么事务A的BeginTS一定大于事务B的CommitTS。
- 为了保证一致性快照，这里还需要保证的是一个事务可以读到所有CommitTS小于该事务BeginTS的事务的写入。因为算法流程中是先写入Intent再获取CommitTS，所以如果一个事务的BeginTS大于某一个事务的CommitTS，那么他一定可以看到该事务写入的Intent，那么他就可以等待在该Tuple上直到LockedTS被置换为CommitTS。

==== 只读事务

之所以引入只读事务的概念，是因为只读事务实际上不需要维护读集等数据，只需要读取一个一致性快照即可。所以只读事务是不需要进行验证的。为了提供更好的性能，只读事务可以选择一个旧一些的BeginTS，用来保证在读取的过程中永远不会看到有并发的事务在写入，即不需要读Intent。这样只读事务可以做到无等待，性能较高。

为了追踪这个没有并发写入事务的安全的BeginTS，ArcaneDB引入了SnapshotManager，他会追逐目前系统中所有活跃的TS，并找到这个活跃TS的最小值，在这个TS之前，可以保证没有任何的活跃事务，所以用这个TS去读取可以保证不会被阻塞。

简单来想，SnapshotManager就是一个并发的Map，每个事务在申请TS的时候会将这个TS加入到Map中，在事务结束的时候会将自己的TS从Map中取走。在实现上，这里ArcaneDB复用了Innodb的LinkBuf #cite("linkbuf")，LinkBuf的功能完全覆盖了前面所说的需求，并且是一个高性能的无锁数据结构。

=== WeightedGraph

WeightedGraph是ArcaneDB对外提供的接口，即一个图存储引擎。每一个顶点通过一个int64_t类型的顶点ID标识，每一条边由其起点和终点所标识，顶点和边上都可以存储一个string类型的数据。

WeightedGraph接口如下：

```cpp
class WeightedGraphDB {
public:
    static Status Open(const std::string &db_name,
                       const WeightedGraphOptions &opts,
                       std::unique_ptr<WeightedGraphDB>> *db);
    std::unique_ptr<Transaction> BeginROTxn(const Options &opts);
    std::unique_ptr<Transaction> BeginRWTxn(const Options &opts);
}
class Transaction {
public:
    Status InsertVertex(VertexID vertex_id, Value data);
    Status DeleteVertex(VertexID vertex_id);
    Status InsertEdge(VertexID src, VertexID dst, Value data);
    Status DeleteEdge(VertexID src, VertexID dst);
    Status GetVertex(VertexID vertex_id, std::string *data);
    Status GetEdge(VertexID src, VertexID dst, std::string *data);
    void GetEdgeIterator(VertexID src, EdgeIterator* iterator);
    Status Commit();
    Status Abort();
}
```

WeightedGraph的作用就是将图语义的请求转化为表语义的请求并发送到下层。这里的转化规则为，每一个顶点自身是一个子表，同一个顶点的所有出边是一个子表。

以写入一条边为例子：
```cpp
Status InsertEdge(VertexID src, VertexID dst, Value data) {
    ValueRefVec vec;
    vec.push_back(dst);
    vec.push_back(data);
    BufWriter writer;
    Row::Serialize(vec, &writer, &kWeightedGraphSchema);
    Row row(writer.data());
    return txn_context_->SetRow(EdgeEncoding(src), row, opts_);
}
```
这里会先将终点和数据转化成一个Tuple，然后将起点ID编码成子表的Key，然后调用子表的写入接口将数据写入即可。

=== Recovery

本节会简述ArcaneDB的恢复流程。数据库的崩溃恢复算法，自从ARIES #cite("mohan1992aries") 出现后，绝大多数的数据库系统或多或少的都借鉴了其中的一些思路，其中最有名的就是ARIES提出的LSN的概念了，LSN代表的是日志序列号，用来唯一的标识一段日志。ARIES提出的seal，no-force的算法可以最大程度的发挥数据库其他组件的性能，而不受到崩溃恢复组件的影响。

ArcaneDB同样借鉴了ARIES的恢复算法，下面简述一下ArcaneDB的恢复流程:

Redo阶段：
1. 从LogStore获取LogReader，开始读取日志
2. 对于Page级别的数据写入日志：
  1. 先读取磁盘中的Page，然后对比磁盘中的Page的LSN和当前日志的LSN
    1. 如果当前日志的LSN较大，则根据日志中的信息将数据写入到Page中。
    2. 如果当前日志LSN较小，则放弃重放日志
  2. 将本次写入记录到活跃事务表中的对应事务上
3. 对于TxnBegin日志，将该事务记录到活跃事务表中
4. 对于TxnEnd日志，将该事务从活跃事务表中移除

Finalize阶段：
1. 对于所有成功写入TxnCommit，但是没有将CommitTS写回到Btree中的所有事务，帮助他完成CommitTS的写回。

Undo阶段：
1. 对于没有成功写入TxnCommit的事务，执行Undo，将其所有写入的数据的TS设置为AbortTS
2. 后续在BwTree Compaction的时候会自动将Aborted Tuple删除。

ArcaneDB的日志恢复算法之所以相较于ARIES多出了一个Finalize阶段，核心原因是ArcaneDB的并发控制算法导致的。在传统数据库(pg, innodb)中，一般都是基于锁进行并发控制，事务在结束的时候，只需要写入一条TxnCommit即可，不需要回写CommitTS。而ArcaneDB的并发控制算法借鉴自Hekaton，一个内存数据库，在写入TxnCommit日志后还需要回写CommitTS。实际上，有正常的数据写入日志，以及Commit日志其实足够将数据库恢复到一个一致的状态。但是这样的缺陷是，SetTS操作本身对Page进行了修改，但是没有记录日志，虽然这个操作本身是幂等的，不需要LSN来帮助做幂等性，但是会影响未来系统的拓展性。比如在ArcaneDB的设计思路中，抽象PageStore和LogStore的目的是可以做成类似Socrates #cite("antonopoulos2019socrates")/Aurora #cite("verbitski2017amazon")一样的存算分离数据库，为此，ArcaneDB需要记录每个Page上的变动，故存在SetTS日志。

=== Flusher

Flusher是数据库提供Larger-than-memory语义的核心组件，其作用就是将被修改过的数据页重新写回到磁盘，在成功写回到磁盘后，我们便可以将数据从缓存中移除，从而可以容纳更新的缓存页。

Flusher本身的策略就很大程度上影响了整个系统的性能，innodb就为此添加了很多根据当前工作负载以及硬件环境进行自适应刷脏页的策略。

Flusher在写入较高的场景下，为了维持脏页率，会耗费很多CPU用在刷脏页上，比如执行序列化逻辑，等待RedoLog落盘等。Aurora等存算分离数据库对于Innodb的改造就是去除了Flusher，让计算层的全部资源都来处理用户的写入请求，将数据的存储，序列化，重放日志等逻辑下放到了存储层。

ArcaneDB的Flusher是一组后台线程，为了维持吞吐，每个线程负责一个分片的脏页。在前台写入的时候，如果数据被修改了，就会将该数据发送到Flusher中。Flusher会在后台将其写入到磁盘中，并更新一些元数据用来指导后续的行为。

FlushPage的代码如下：
```cpp
void FlushPage(PageHolder page_holder) {
    PageSnapshot snapshot = page_holder->GetPageSnapshot();
    LSN lsn = snapshot->GetLSN();
    std::string binary = snapshot->Serialize();
    // wAL protocol
    log_store_->WaitForPersist(lsn);
    Status s = page_store_->UpdateReplacement(page_holder->GetPageId(), binary);
    bool need_flush = page_holder->FinishFlush(s, lsn);
    if (need_flush) {
        InsertDirtyPage(std::move(page_holder));
    }
}
```
在CloudJump #cite("chen2022cloudjump")中有提到，在云上的IO延迟会比本地的SSD高出一到两个数量级，所以我们不能在持有锁的情况下进行IO，这样对前台写入影响过大。PolarDB这里的解决方法是在刷脏的时候拷贝一份Page出来，不阻塞前台写入 #cite("cao2021polardb")。而ArcaneDB这里的解决方法是利用了BwTree数据不可变的特性，刷脏的时候读取一个快照，在写入完成后，对比刷入磁盘的Page的LSN和当前内存最新的LSN，如果在刷脏过程中Page被修改了，那么我们需要将其重新插回到队列，等待继续刷脏。

=== RangeScan

为了适配图的工作负载，ArcaneDB需要支持扫描一个顶点的所有出边，也就是扫描一颗Btree。因为BwTree在每个Page上可以简单看作是一个小的LSMTree，所以在扫描的时候的算法也和LSMTree是类似的。参考RocksDB的实现，ArcaneDB会读取所有的DeltaNode，组成一个MergeIterator，其内部就是一个最小堆，每次从多个DeltaNode中找到排序键最小的数据返回给用户。

一个实现的细节是，ArcaneDB内部需要识别排序键相同的Tuple，并只返回最新的版本给用户。

#pagebreak()

= 性能优化

ArcaneDB在实现的过程中，在不断的进行性能优化。基本策略是在开发的早期就写出了一组用来测试性能的工具，测试的场景包括随机写入，顺序写入，随机点读，顺序点读，顺序扫描。在完成一些重要模块的编写之后，会用这些工具进行性能测试，并采集运行中的火焰图，针对一些瓶颈点进行性能优化。

ArcaneDB希望能够提供比较好的Scalability，即尽可能减少中心化的数据结构，提高ArcaneDB在众核场景下的性能，也为未来拓展ArcaneDB为分布式图存储引擎打下基础。

本节会详细阐述一下ArcaneDB做过的性能优化。

== Btree锁优化

ArcaneDB虽然使用了BwTree作为参考来实现Btree，但是其写写操作仍然会互斥。并且由于使用了shared_ptr来管理内存，所以读写操作和读读操作并不是完全的无锁，而是会有一个短时间的临界区。

ArcaneDB中的一个Btree节点中具有如下的结构：
```cpp
class BtreePage {
private:
    std::mutex mu_;
    std::shared_ptr<DeltaNode> node_;
}
```

每次读者需要先获取锁，然后获取到DeltaNode的指针，接着释放锁，最后才开始读取里面的数据。

由于ArcaneDB使用了协程框架brpc，为了避免阻塞协程框架内部的pthread worker，所以这里的锁需要用bthread::Mutex来替换。而bthread::Mutex的一个缺点是他没有做和pthread_mutex类似的优化，针对短程锁会先在用户态去自旋，超过一定时间后才会陷入内核态等待在futex上。这带来的问题是在热点读取的场景下，虽然每个锁的临界区都很短，但是每个线程都会发现目前存在竞争者，然后将当前bthread挂起，进而导致大量的bthread的挂起和唤醒，从而影响性能。

然后ArcaneDB使用了类似RCU的技术来优化读取性能。其更新思路类似于CopyOnWrite，下面会简述一下流程：

受到临界区保护的数据会被冗余存储一份，分别称其为foreground_data和background_data。每个线程有一个thread_local的锁。

每次在写入的时候：
1. 先更新background_data
2. 更新完成后，将background_data和foreground_data对调
3. 写者去遍历所有读者的thread_local的锁，他会都获取一次锁。
4. 更新background_data，也就是之前的foreground_data

读者每次只需要在获取自己thread_local的锁的时候进行读取即可。

在这个流程下，读者与读者之间不会相互冲突。并且每个读者只会获取自己本地的锁，不会涉及到读取远端的锁，减少了cacheline invalidation的次数。在一些高并发的场景下，这种算法具有远比读写锁更好的性能。

在后续的测试中，发现虽然类似RCU的这种思路对于读性能优化了很多，但是对于写性能却有一定的影响，比如在32线程的背景下，每次写入需要锁住32个pthread_mutex，虽然没有任何的冲突，但是更新原子变量本身却带来了一定的开销。后续我认为，虽然用简单的自旋锁会导致在热点情况下cacheline invalidation增加，但是类似RCU #cite("mckenney2001read") 的方法会影响Scalability，并且自旋锁导致的问题不一定在热点情况下是瓶颈，故后续使用了absl::Spinlock #cite("absl")，这种自旋锁会在自旋一段时间后进行backoff，来缓解争用。

C++20引入了`std::atomic<shared_ptr<T>>`，可以非常好的嵌入到ArcaneDB中，其基本原理是将shared_ptr中的控制块嵌入到了数据中，从而可以实现将整个智能指针嵌入到一个64位的变量中。

在原始的BwTree论文中，他没有使用shared_ptr来管理内存，因为引用计数本身在热点情况下也会带来一定的开销，他使用的是一种无锁的内存回收技术，叫做Epoch Based Reclaimation #cite("brown2015reclaiming")，这个算法通过追踪每个线程的Epoch，来确认每个线程对于一些被回收对象的可见性，然后计算出一个可以安全回收对象的水位并回收。这种相当于粗力度的对象追踪技术拥有很好的Scalability。但是他有一个缺点是需要统计所有线程的Epoch，在协程的背景下，并发的活跃线程数量非常高，进而导致统计所有线程的Epoch这个操作开销过大。也是因为这个原因，ArcaneDB放弃使用了EBR技术。

对于上述提到的自旋锁的优化，一些近期的工作中已经提出了一些比较好的锁的方式，比如针对ART的optimistic lock coupling #cite("leis2019optimistic", "leis2016art", "bottcher2020scalable")，通过乐观读取的方式来减少对于锁中原子变量的CAS操作，进而提升锁操作的Scalability。然而这些工作中对于访问的对象有一定的假设，他们会先读取对象再进行锁的验证，进而要求这个对象本身在并发读写的场景下不能出现损坏。ArcaneDB可以通过一些手段来使用这个技术，比如将shared_ptr对齐到缓存行中，但是这些实现方法并不通用，所以目前还没有做这个优化。

== 内存分配优化

内存分配属于是一个系统中比较占用资源的点了。ArcaneDB的内存分配的优化手段也比较通用：
- 链接一些三方的内存分配库，比如jemalloc, tcmalloc，通过一些thread local的手段来减少临界区的争用，以及池化等手段将内存块缓存在用户态，减少陷入内核态的次数。
- 针对一些特殊的对象，在应用层对一些数据分配进行池化。因为jemalloc等库都是按照页分配，并且不能很好的感知应用层的语义。比如在ArcaneDB中，可以通过对象池等技术来减少一些常用对象的内存分配，比如bthread栈的分配，BwTree DeltaNod的分配。
- 减少内存的分配次数：
  - 比如常用的std::vector需要在堆上分配空间，可以通过一些inline vector的手段，在vector长度比较短的时候，直接在栈上分配对象。另一个比较常见的例子就是std::string，一般的编译器都会对一些短字符串进行内联，减少内存分配的次数，基本思路也和inline vector相同。
  - 在ArcaneDB中，会有一些将数据序列化传给下层的操作。比如刷入脏Page的时候，会先将Page序列化，然后传给PageStore。比如写入WAL的时候，也需要先序列化WAL，然后将日志传给LogStore。对于WAL来说，其占用空间较小，一般一条日志基本小于100byte，所以为了减少内存分配的次数，ArcaneDB会直接在栈上分配一个足够大的空间，在栈上序列化，然后将栈上地址传给LogStore，从而减少关键路径上的一次内存分配开销。

== 编码优化

因为ArcaneDB为了避免邻接表索引的开销，使用了特殊的编码，从而可以在查询的时候，直接根据顶点的ID得到对应Btree根节点的PageID。

测试的时候发现，在随机读场景下，因为整体操作非常少，只有编码，定位到BtreePage，读取BtreePage中的行这几步。而在之前ArcaneDB使用了std::to_string进行编码，底层是snprintf，效率比较低。在定位到这个问题后，改用了fmtlib的一个比较特化的编码函数，解决了编码速度过慢的问题。并且为了提高速度，这里还提前进行了一次强制转化为uint来避免format_int去处理符号位，因为内部会走format_unsigned这段函数。

改进后的编码方式如下：
```cpp
static std::string VertexEncoding(VertexId vertex) noexcept {
  // cast to uint is more faster since it doesn't need to handle minus mark
  return fmt::format_int(static_cast<uint64_t>(vertex)).str() + "V";
}
static std::string EdgeEncoding(VertexId vertex) noexcept {
  return fmt::format_int(static_cast<uint64_t>(vertex)).str() + "E";
}
```

== 事务锁优化

ArcaneDB在第一版的事务实现中，用了一个中心化的锁管理器来处理写锁。并做了分区来减少并发冲突。但是其操作内存的效率还是较低，因为在基本没有冲突的场景下，每个线程虽然同一时间只会访问一个分区，但是因为哈希分区以及工作负载的原因，每个线程会连续访问不同的分区，并且由于锁管理器是持续写入的，导致相关的内存的cacheline invalidation次数较多。在测试中发现锁管理器的访问虽然指令不多，但是其IPC较低，主要原因就是缓存局部性较差。

为了避免中心化的锁管理器影响ArcaneDB的Scalability，我后续使用了每个Btree/GraphStorage一个的锁管理器，这样在访问的数据没有冲突的时候，每个线程要上锁的时候，访问的内存就是不同的，在工作负载有局部性的场景下性能会有一定的提升。

然而每个Btree一个锁管理器会严重增加Btree的内存开销，进而影响整个系统可以缓存的数据数量以及缓存命中率。为了解决这个问题，ArcaneDB参考了Hekaton的方法，在更新数据后将新的数据项上写入一个特殊的timestamp，即使用64位timestamp中的最高位代表当前行是否被锁住。后续的读者在读到上锁的行的时候会等待，直到锁被释放位置。写者在提交的时候，如果遇到了已经上锁的行，则认为发生了写写冲突，则会Abort。

通过将锁内联到数据的方法，在不引入额外空间开销的前提下，ArcaneDB的写性能的单核QPS提升了2万左右。

== 日志组件优化

在打开了WAL之后，因为目前的LogStore是一个中心化的组件，导致所有线程都会在LogStore中产生竞争。在上面LogStore一节中讲到，每次写入需要先预留LogBuffer，其中涉及到一个原子变量的CAS操作。在一个简单的写入事务中，比如插入一条边，ArcaneDB会访问日志组件四次，分别为写入事务开始日志，写入SetRow日志，写入事务提交日志，写入SetTS日志。而在高争用的环境下，每次修改原子变量预留日志缓存在测试中发现需要约1us，夹杂上其他的写入操作，每次写入日志延迟约为2us。相当于在打开WAL之后，不算落盘时间，对日志组件的操作就让插入一条边的延迟增加了约8us。而在全内存场景下，ArcaneDB写入的延迟约为3到5us。相当于打开WAL后使得写入延迟高了1到2倍。

这里的开销主要来自于多个线程对于单个原子变量的并发访问。之前已经有一些工作来解决日志组件全局LSN的问题。如SiloR #cite("tu2013speedy", "zheng2014fast") 放弃LSN来提高整个系统的Scalability，而这篇文章 #cite("haubenschild2020rethinking") 则是引入了Local LSN，用类似逻辑时钟的方法来解决问题。

ArcaneDB利用了图数据库系统的一个特点，即写入的工作负载绝大多数情况下都是为了保证正反向边的原子性。比如插入一条边，可以保证原子的插入正向边和反向边。或者是为了保证单个顶点的出边的写入原子性，比如原子的插入同一个顶点的若干条出边。可以发现这两种工作负载对于不同顶点的写入原子性没有要求，所以我们可以按照顶点对LogStore进行分区，保证一个顶点的正向边和反向边会被划分到同一个分区中。这样在不同顶点写入的时候访问的就是不同的日志流，相互之间没有任何干扰。

而对于一些少数情况下有多个顶点的原子写入需求，通过2PC保证即可。即在多个顶点对应的日志流中先写入Prepare日志，决议之后写入Commit日志即可。

引入WAL后，还存在的一个问题是每个事务提交前需要等待数据落盘。根据Cloudjump #cite("chen2022cloudjump") 中所述，现在的本地SSD落盘延迟约为20us。为了减少事务的提交延迟，ArcaneDB需要一个手段用来在等待落盘的时候高效的将事务挂起，并在数据落盘的时候将事务唤醒。

早期ArcaneDB使用的方案较为简单，在发现需要落盘的时候，会直接调用sleep，唤醒后对比当前持久化的LSN和当前事务的LSN。如果已经落盘则返回，否则继续sleep。这种方法的缺点是sleep的时间控制的不够准确，导致每个事务的延迟抖动较大，为了保证足够的吞吐，必须开足够多的线程。而更多的线程会带来更多的开销，并且对bthread的TimerThread有比较大的压力，进而导致抖动更大。

因为bthread提供了bthread_yield接口，用来挂起当前的bthread，让出执行线程给其他bthread。在发现需要落盘时，直接调用bthread_yield将当前bthread挂起。这种方法减少了对TimerThread的压力，但是唤醒时间不确定的问题仍然存在。在测试中发现，一些高并发的场景下，火焰图中大多数都是bthread_yield，说明协程的挂起和恢复耗费了绝大多数的CPU。

在阅读LeanStore的源码时发现他使用了C++20的std::atomic的wait和notify来解决这个问题。其实现方案也较为简单，可以让当前线程挂起到futex中，等到IO结束后落盘的时候，IO线程更新当前持久化的LSN，并唤醒futex中的线程。bthread中也提供了这样的同步原语，叫做butex，其作用基本类似于futex。

下面是对比的火焰图，可以看到效果还是比较明显的：

#img(
  image("./14.png"),
  caption: "优化前"
) <img14>

#img(
  image("./15.png"),
  caption: "优化后"
) <img15>

在改进前，bthread_yield占用了约80%的CPU。改进后，butex_wait只占用了约7%的CPU。

== 扫描优化

ArcaneDB使用了BwTree，在单个Page的扫描上需要进行DeltaNode之间的合并，类似LSMTree。扫描的性能较差。为了优化ArcaneDB在扫描上的性能劣势，这里我针对了不同的工作负载提供了不同的接口。比如之前提到的求交集，需要扫描出来的数据是有序的，所以这里进行DeltaNode之前的合并，再提供一个Iterator给用户。而对于一些不需要顺序的扫描场景，比如最短路径，PageRank等，只要求访问一个顶点的所有出边，但对顺序没有依赖。这时候我们可以选择不合并DeltaNode，而是直接按照DeltaNode中的顺序将数据返回给用户。这样的访问性能类似于访问块状链表，性能远高于原本的Merge Iterator。

所以ArcaneDB提供的接口如下：
```cpp
EdgeIterator GetEdgeIterator(VertexID src) noexcept;
UnsortedEdgeIterator GetUnsortedEdgeIterator(VertexID src) noexcept;
```

== 协程框架适配优化

在使用一些用户态的调度框架的时候，最好了解他的内部实现，否则在某些地方可能会遇到协程框架不兼容的问题，或者错误的使用一些系统调用导致卡住整个调度器，影响性能。

bthread的调度框架较为简单，每个工作的线程是一个pthread，每个工作线程有一个自己的队列，用户启动bthread的时候会将相关的上下文丢到队列中，工作线程从队列中获取要执行的bthread，并执行。

这里的一个问题是，如果我们访问了某些阻塞的系统调用，如read等，会导致bthread底层的工作线程被阻塞，而非这个协程被阻塞。进而导致在阻塞期间有一个工作线程无法执行任务，导致整体系统的CPU使用率上不去。

比如ArcaneDB在写日志落盘，以及写入Page的地方都会有IO操作，为了避免bthread的工作线程被阻塞，ArcaneDB实现了基于bthread同步原语的bthread future/promise，再结合一些异步IO方式从而保证了工作线程不会被阻塞。

ArcaneDB目前复用了LevelDB作为PageStore，而LevelDB没有异步接口，所以ArcaneDB为LevelDB套了一个线程池来模拟异步接口。基本框架如@img16

#img(
  image("./16.jpg"),
) <img16>

异步任务会被先扔到一个队列中，线程池中的工作线程会从队列中获取任务，发起对LevelDB的同步调用，调用结束后执行Callback。Callback执行结束后前台的bthread就会被唤醒，代表IO执行完成了。

== 插入场景性能优化

在ArcaneDB插入边的时候，为了避免重复边问题，需要先去读一下Btree中是否已经存在这条边，在测试中发现约有9%的CPU花费在了查找重复边的操作上。而对于这种查找空数据的场景对于LSMTree/BwTree等数据结构不是很友好。因为他们需要遍历所有的DeltaNode，才能确定这个边不存在。对于LSMTree来说，就是需要遍历所有的Level，开销较大。所以为了缓解这个问题，大家普遍的解决方法是通过bloom filter。读取的时候可以先读一下bloom filter，如果不存在就可以直接插入边，不需要额外的读取了。

对于ArcaneDB来说，每一个DeltaNode中的bloom filter代表的是这个DeltaNode以及他之前的所有DeltaNode的数据之和。每次创建新的DeltaNode的时候，会先读取当前DeltaNode的bloom filter，复制一份出来，并将本次DeltaNode新加入的数据添加进去。随着数据的删除，这种方法的False Positive的概率也会上升，所以在Compaction的时候会重建bloom filter。

#pagebreak()

= 实验

本章对ArcaneDB进行了测试。首先是系统之间的横向测试，这里对比的系统都是一些轻量级的存储系统，分别有本次毕业设计的ArcaneDB，一个前沿的图存储系统LiveGraph，以及基于LevelDB实现的图存储系统。然后是ArcaneDB的优化测试，针对第四节中的几个优化点，做了优化前后的对比测试。

== 测试环境

硬件环境：2个Socket的`Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz`，每个Socket共有16个物理核，没有超线程，一共32个物理核。64G内存。

数据集：因为本次毕业设计是图存储系统，并不关注计算相关内容，所以数据集唯一影响的就是每个工作线程任务的偏斜程度。为了测试系统的性能上限，以及时间因素考虑，我在测试的时候屏蔽了数据集偏斜带来的影响，给每个线程分配的任务较为均衡。所以本次实验的数据集是我通过代码生成出来的，并可以控制整个图的点边，以及出度的数量。

工作负载：实验所测试的工作负载主要为随机点写和顺序扫描。

== 系统横向对比

本节对比了ArcaneDB，LiveGraph，以及基于LevelDB实现的图存储系统，三个系统。

随机点写场景。实验结果如@img18 所示

#img(
  image("./18.png"),
) <img18>

可以看到ArcaneDB在多线程并发写入的场景下具有良好的Scalability。在线程数比较少的时候，ArcaneDB吞吐低于LiveGraph和LevelDB的原因是目前ArcaneDB不会主动触发落盘，而是需要等待后台线程定时落盘，进而导致事务提交的延迟比较高，所以在并发数不高的情况下，ArcaneDB的吞吐会偏低。

顺序扫描，无排序，单线程，小出度场景实验结果如@img19 所示

#img(
  image("./19.png"),
) <img19>

这里的ArcaneDB with Compaction指的是ArcaneDB读优化的模式，即调整BwTree的Compaction参数，用更大的写放大来减少读放大。

在出度较少的场景下，LiveGraph的扫描性能较高，这是因为LiveGraph扫描数据类似数组，整体局部性比较好，并且在全内存场景下LiveGraph的访问路径相较于ArcaneDB较短，所以会性能较高。随着出度增加，扫描操作本身的瓶颈由定位顶点变成了扫描本身，所以ArcaneDB在Compaction模式下具有和LiveGraph相近的性能。

顺序扫描，无排序，单线程，大出度场景实验结果如@img20 所示

#img(
  image("./20.png"),
) <img20>

顺序扫描，有排序，单线程，小出度场景实验结果如@img21 所示

#img(
  image("./21.png"),
) <img21>


顺序扫描，有排序，单线程，大出度场景实验结果如@img22 所示

#img(
  image("./22.png"),
) <img22>

这里的LiveGraph with sort指的是让LiveGraph读取数据后，重新进行排序的版本，因为LiveGraph本身并不支持有序读取。

可以看到在出度较少的场景下，LiveGraph的性能更优，因为局部性较好，并且排序的开销更低。随着出度增加，排序的开销逐渐变大，LiveGraph本身的读性能优势也在降低。在出度超过50的时候，LiveGraph的读取性能就会下降，而ArcaneDB由于本身就是有序结构，没有排序的开销，所以只会受到局部性的影响。

== 系统优化实验

本节针对第四节中的一些性能优化点做了对比实验，来看在这些优化分别会在那个阶段产生瓶颈点，以及优化后的性能收益。

ArcaneDB，未打开WAL模式下优化结果如@img23 所示

#img(
  image("./23.png"),
) <img23>

这里的InlineLock指的就是事务锁优化，可以看到在纯内存模式下，锁的开销还是比较大的，使用inline lock可以减少内存争用，提高缓存友好性。

ArcaneDB，打开WAL，异步落盘模式下优化结果如@img24 所示

#img(
  image("./24.png"),
) <img24>

打开WAL之后，性能瓶颈点会转移到日志组件中，此时优化事务锁效果不大，在优化日志组件后吞吐才得以提升，并会转化到事务锁中。

ArcaneDB，打开WAL，同步落盘模式下优化结果如@img25 所示

#img(
  image("./25.png"),
) <img25>

同步模式下，性能瓶颈点还是在等待日志落盘，以及日志组件中。而最后的事务锁优化之所以效果不大，是因为测试机器的磁盘带宽已经被打满，瓶颈点转移到了磁盘中，所以优化内存操作意义不大。

#pagebreak()

= 结论与展望

== 本文工作总结

本文通过对现有图存储引擎的总结，识别出了目前图存储引擎的不足点，然后针对这些不足点设计并实现了一个高性能的基于磁盘的图存储引擎ArcaneDB。

本文的工作主要分为一下几个方面：
1. 简单介绍了现有图数据库的现状
2. 针对图工作负载，以及现有图存储引擎的不足点，设计并实现了ArcaneDB
3. 针对Scalability对ArcaneDB进行优化，进一步提高了ArcaneDB的性能
4. 通过系统性能对比实验，证明了ArcaneDB的高效性，并且验证了ArcaneDB的设计思路

== 未来工作展望

本文提出了一个高性能的支持图分析的图存储系统ArcaneDB，为之后的研究打下基础。由于系统开发时间有限，ArcaneDB在设计时埋下的一些伏笔并没有完全实现，ArcaneDB仍然具有高度演化的能力，主要有：
1. 划分LogStore/PageStore的目的是做存算分离，思路参考自Socrates，PageStore，LogStore以及ArcaneDB计算层可以独立拓展，提高系统的可用性以及性能。
2. PageStore目前实现基于LevelDB较为简单，之后可以针对图的工作负载对PageStore做更加特化的设计，比如针对图结构对数据进行冷热分离，减少写放大。
3. ArcaneDB支持子表语义，可以提供二级索引的能力，这样在一些查询带有过滤器的多跳场景下具有更好的性能。也可以参考SQL Server，为二级索引支持列村，提高AP场景的查询能力。
4. ArcaneDB事务层与图工作负载深度结合设计，放弃对长事务的支持，转向支持内存多版本，减少多版本回收的开销。
5. ArcaneDB Btree层实现为BwTree，在Compaction到最后一层的时候转化为列存，提高内存中数据结构的读取性能。
6. PageStore可以在底层将数据转化为列存格式，RO节点可以读取列存数据，提高读性能。
7. PageStore支持列存并感知上层Schema，读取Page的时候按需读取，从而达到Skip IO的效果。

#pagebreak()

#references("./ref_sheep.bib")