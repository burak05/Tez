from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



sentences = [
    "Biochemistry isolates the protein of the functional organ and studies the chemical composition of the puree. Genetics study the population and observe how traits are being transmitted from parents to offspring through genes. Molecular Biology isolates the gene (DNA) and arrives at the protein form for biochemistry.",
    "theoriticl thinking less than other science vitalist the organic from inorganic all of them work and explain  cell , atoms , genetics",
    "The most important relationship is  all of them interested biology. Using the biochemistry we can improve molecular biology and improve molecular biology we can study on genetics. ",
    "Before all these terms (chemistry , biochemistry , molecular biology and genetics) we use to study on different things like light or math etc. We didn't know anything about these terms once we started to discover other things it lead us to found these terms . Newton and other scientist studied on light and we finally found telescope and similar tools and we found elements and chemistry has born with chemistry we could find microscobic compounds it lead us to molecular biology and once we collect more information about all these molecular biology and genetics etc. used all these information and they grow together.",
    "biochemistry studies on dna part. genetics studies on gene part. molecular biology studies on both of them. ",
    "its all about change something without breaking genetic code biochemistry declare how change, molecular biology which change,genetics why happen if its change",
    "biochemistry contains organic and inorganic chemistry.chemistry and biology always have  connection .for instance animal die and when  deadth body convert to land by biology and chemistry combining.",
    "They are all realated to themselves. Biochemistry, molecular biology and genetics have an unseperatable relationship. If a scientist wants work for genetics he/she has to involved with moleculer biology and biochemistry.",
    "biochemistry focuses on the relationship between life function and protein(chemical compound). genetics focuses on living beings and how they inherit their trait without knowing the chemical functions of the process. molecular biology combines biochemistry and genetics and investigate both aspects.",
    "Genetic is looking for the what kind of qualification of the living things impress the reaction or some genotopical changes. Biochemistry is looking for its reactions and molecular biology tries to explain the mathematical way of it.",
    "They have the same point. That is organic chemistry. ",
    "Biochemistry works on  to things like nutrition and enzyme deficiency and hemical reactions that happen within the body Molecular Biology is a Genetics focused course Molecular Biology works on Microbiology and PCR so Molecular Biology and Genetics  pretty looks like each other",
    "Biochemistry is about functions of proteins. It explains how proteins affect the genes. Genetics is about functions of genes. It compares the genes. Molecular biology integrates those two. Take the gene and arrive the protein.",
    "these scientific branches examine common values in a cyclical relationship. Such as molecular biology uses protein and genes when genes uses by genetics and protein uses by biochemistry.",
    "all of them work area humans,animals,plants. all of them study area: live all of them study area: World of lives.",
    "Biology need genetics to explain transformation of the human nature, mutation and the differences between different people. Genetic need molecular biology to understand cells, nucleus and their mutations",
    "in Biochemistry work area is compounds(how compunds bond with each other, H2O...) it don't care organisms. in Moleculer biology  organism are important(What is Dna ,What is Chromosome...) and it don't care compounds. But in genetics , moleculer biology and biochemistry are integrated. So with help of this diciplines we can study Genetics.",
    "biochemistry search the structure and actions of proteins and how genes are expressed in terms of cells. molecular biology takes genes by considering genetic approaches to things and genetics execute most if the same genetic parts of molecular biology and include a nonmolecular part genetics",
    "these fields is generally talk about people s cell and   other organic things . and these fields  are composed of atomic things  .and all of these field fundamentally decribe the nature and try to  understanding its",
    "Biochemistry processes the chemical structure of matter in a biological way. Molecular biology processes the molecular state of matter in a biological way. Genetics processes the purely biological structure of matter. Their basis is to contribute to the development of biology.",
    "Molecular biology workswith cells. Genetics works with the genetical codes in the cell core. And biochemistry works what can we do with that geneticaly changed cell.",
    "Biochemistrists are interested in only proteins, they are not interested in genes. Genetics are interested in genes. They are not interested in  proteins , they make gen isolation and study on it. They observe the genes on different cells.  Molecular biology's study area consist of both of them(genetics and biochemistry).",
    "Biochemistry, molecular biology, and genetics have relationship because actually they try to find similar result (not same but similar)  and also one of their subject can be other's subject. ",
    "Biochemistry focuses on the protein part of life functions. Genetics focuses on the gene part. Molecular Biology integrates those two science. For example: gen > protein",
    "biochemistry, molecular biology and genetics; they all affect each other. A new scientific improvement changes the other scientific knowledge.",
    "Biochemistry works on the chemical parts of a living thing. It takes carbonhydrates, fats and proteins (aminoacids) as its field. Genetics works on the DNA and tries to understand what does DNA do and how does it control the cell or the body of the living thing. Molecular biology works on molecules of a living thing. All living things are a composition of molecules. Molecular biology tries to understand how the molecules gather together. ",
    "Biochemistry study on changing for  chemical property. It study between Biology and Chemistry Molecular biology study on relationship of moleculs affenity each other. Genetics just study on genetics. For example; DNA,RNA ",
    "Biochemistry study function of protein, molecular biology study molecule of protein ; amino acid , genetics study gene.",
    "They explain each other. For example Darwin's study explaine by Mendel's study in genetics. In moleculer biology Darwin explain natural selection, variation then they became more understanding with Mendel' study.",
    "Biochemistry and Molecular Biology deals with the structure and function of proteins and how genes are expressed in cells. Molecular Biology takes genes further by considering genetic approaches to things (like genetic engineering and how to approach genes).",
    "Biochemistry is about biology and chemistry and it works on part of living in chemistry. Molecular biology works on human, foods etc. Genetics is basically about main elements of things who are alives. Actually , if we look at their ingredients they are all about mainly biology and nature.",
    "Biochemistry,molecular biology,genetics all of them so important for our lifes.They have same subject for example organic chemistry.They are related each other.Also they are searching quite same things.Our lives can be harder for us withouth biochemistry,molecular biology,genetics.",
    "Biochemistry , molecular biology and genectics are dependent each other. Genetics couldn't understand as don't knowing chemistry.  Understanding  biology possible to good knowledge of chemistry. Development modern chemistry lead to discovered the organizmas. ",
    "biochemistry involves more about organic molecules molecular biology is about more central dogma, genes to proteins genetics is about genes and uses mutant cells the relationship between these 3 is that biochemistry produce information about organic molecules which helps molecular biology because proteins are organic molecules. Genetics helps molecular biology beacuse it has the information.",
    "Biochemistry searchs proteins in living things. Genetics searchs DNA in living things. Molecular Biology connects those two areas' information. That means Central Dogma. ",
    "Biochemistry, molecular biology and genetics are linked each other.If an improvement happened one of these, It will help the others. If you want to learn about genetics then you have to know a lot about biochemistry and molecular biology.",
    "Biochemistry focused on the protein part of life functions. It studies the components and it is independent from the organism. Genetics focuses on the gene part. Usually mutants are used. It is organism without the component. Molecular Biology integrates those two, as can be quite well ascertained from the “central dogma”.",
    "Developed of biochemistry influenced molecular biology and genetics.Because if one of the people or scientists discover new knowledge in biochemistry,other people use that in  molecular biology.Every new content allowed to discovered new characteristics,matters,pyhsical features or allowed to made invention.For example,find to cells influenced to Mendel.After that,Mendel studied 29.000 peas.",
    "They  usually work with about  similar things.",
    "Biochemistry : Collect data on subject(animal plant or other living things) Genetics :  Look for DNA and genetic information. Compare with others. Molecular Biology : Combination of genetics and biochemistry. Biochemistry    +     Genetics   =    Molecular Biology",
    "Biochemistry is really important for understand relationship between biology and chemistry. Moleculer biology is starting from chemistry that's why biochemical studies really important. For understand genetics we have to measure and observ some thinks. We use biochemistry and moleculer biology for that.",
    "biochemistry interested about protein. Molecular biology and genetics study on gens.Sometimes they work both of other.But biochemistry not work on gens and Molecular biology and genetics not work on proteins generally. But this different type of science about living things and human so sometimes they have same things and work place. Molecular biology study  genetics and biochemistry and biology",
    "They are all sub-studies of biology. All of them can help us understand more about amino acids.",
    "Biochemistry:Biology +Chemistery. Molecular Biology: Minimal biology .They are using extends and ganateic rules. Genetic:Extends",
    "Moleculay biology and genetics both studied about gens. And biochemistry and molecular biology search about bilogy. Their searching areas is connected. They are different things, under the same subject : Modern biology.",
    "Molecular biology compose biochemistry and genetics.",
    "Molecular biology and genetics focus on plant , animal or any cells. Biochemistry also focus on components of samples.(What it occurs). We can reach information about parents of any animal or plant with genetics. Furthermore we can reach that similarity between the cells thanks to molecular biology. molecular biology > genetics.",
    "biochemistry is about chemistry, molecular biology is about biology. there is a  flow chart between from biochemistry to genetics, from genetics to molecular biology , from biology to biochemistry. There is a loop between this three things. They influence each other.  generally biochemistry works be in lab.",
    "Biochemistry interested in chemistry using biology. Genetics and molecular biology interested in DNA combination. All of them use cells. Molecular biology explains DNA and genetics. ",
    "Genetic searchs how to inherit traits from parents to offsprings,molecular biology make researches about living creatures' cells in molecular level and biochemistry works for both chemistry and biology and uses both. We can not think this 3 area of science seperately.They influence and effect each other.",
    "biochemistry interst chemical things inside organism .they isterest how to work electricity compounds in organism.Some organic compound are able to find chemists to thankful them. organism occurs lot of compounds and how to they work and chemistry find answes of this questions.molecular biology is interest total cell . for example, how work protein in cell, what happen in cell. genetics just interst part of the chromosomes that specipic genes local.genes transfered to new organism, genetics works , for example, GDO product , recombination ...",
    "The biochemistry is searching for the relationship between biology and the chemistry.Molecular biology is searching for the smallest size of the living thinks and the genetics also looking for similar thinks(like DNA).At the end of the all topic we need them all for the defined the fundamentals of the biologic ecosystem .",
    "It is a circle actually. All them in a relatioship, its called central dogma.  They are in a connection and complete each other. It's about cells, organisma, DNA, gene, proteins and things like that.",
    "genetics studies in gens and into gens and more small protein adenin siztozine. molecular biologis studies with cells and more small things in animal and plants cells. biochemistry studies about chemical event about biolochical thing. They all studies based on gens. and events in gens and chemical events between molecules in gens.",
    "They study on biology, structure of atoms ,elements. They study on Central dogma. They explains similar areas.",
    "Scientists work together on  biochemistry ,molecular biology and genetics.This subjectS of science are connected each other.For example if  one of the scientist doesn't good at molecular biology,he can't good at genetics.",
    "biochemistry just look protein structures. without looking organism just structure, genetics just looking gens with looking organism  molecular biology look both of them ",
    "All of them deal with cells. These are the branches of science that affect the life cycle. Mutation experiments have been carried out.",
    "Biochemistry  focus to protein part of living cell genetic   focus  gen part molecular biology focus  both  how convered gen to protein",
    "Biochemistry, molecular biology and genetics have common a lot, but also they have different points. Biochemistry is a science which studies the chemical substances in the structure of all living things, microorganisms and animal the chemical processes. Molecular Biology is a science which studies biological topics in the molecular area. Genetics is a science which studies all living things genetic materials and inheritance. The relationship between the biochemistry, molecular biology and genetics is that they have common work area. They are connected each other. ",
    "A biochemist inspect the given problem by obtaining chemical and biological materials. A molecular biologist inspect it by the molecular level and a genetic engineer inspect it by obtaining different examples and comparing them to each other.",
    "Biochemistry focused on the protein part of life functions. It studies the components independent of the organism.Genetics focuses on the gene part. Usually mutants are used. So, it is organism without the component. DNA/DNR Molecular Biology integrates those two, as can be quite well ascertained from the “central dogma”.",
    "Biocehemistry let us connect it with molecular biology and them let us research about genetics and we can light the way of genetics. Scientists could understand DNA because of them.",
    "Biochemistry study the living creatures in case of chemistry point of view. It contains biology and chemistry at the same time. It is also sub-subject of chemistry. Molecular biology study features of biological matters in the molecular and microscopic level. So it interested in how thinks work for the smallest and most tiny particular. Genetics study the how living creatures cells act when they want to transfer the their being data from parent to next generation. If we want to know more about genetics, do experiments and discover treatment for different diseases we need to be advance in genetics more. In order to do that we have to depend on biochemistry and molecular biology at the same time. Because they all affect each other.",
    "Each branch research relationship  between the smallest parts of their branch.",
    "you can think about the expirince of fly fruit Biochmestry islote fly fruit and study  it by searching about the  thıng whıch made the diffrenet betyween eye colors biology will compare between the two once of fly fruit and try to fınd what the different betyween them genetecs wıll focus on gene and try to reach to the protın",
    "accoding to the central dogma. there is transaction for from gen to protein  when occuring the protein with ribozoms. three of them can look into dropholia cells for the different part. genetics compare the traits with crossing the each other. molecular biology look into gens and its protein as isolating gen. biochemisty make a PURE for drocholia thus three of them look into different way. Biochemisry interest the function and protein . molecular biology interest gens and protein and finally genetics interest the gen and function.",
    "They all tend to study even smaller than cells. They explain the bonds between atoms.",
    "Biochemistry researchs organic matter aside from the organism. Genetics inspect on the genes of organisms. Molecular biology is the cross-product of both. The researchers of molecular biology works on both organic matter and genetics. In case of a fruit fly, an biochemist squeezes the fly and inspects on the puree, a genetist inspects on the varition occurs on different generations of a fruit fly and, and a molecular biologist isoletes the genes of the organism and work on it.",
    "scientist use these together when they work on living oraganizms.",
    "All of them research atoms and molecules. they are as organic chemistry. Biochemistry is chemistry of biology, molecular biology is elements in bodies ;  genetics is located elements parents by children. they are relevant organic chemistry.",
    "Biochemistry work on components (proteins) but doesn't work on the life form (its genes) which contains the components. Genetics work on life forms, but does not work on components of the life form. Molecular Biology works on the relationship between life form (its genes) and its components (proteins). For example Molecular Biology works on central dogma.",
    "Biochemistry is using chemistry to explain biollogical reactions and trying to make that reactions in labratory envoriment. Moleculer biology is where we use biochemistry to make profit and improve human life . For example scientist realised that all  proteins we produce in our body is come from the information in our DNA so they cut the part of our DNA that produced inusilin and gave it to bacterias and now we can mass produce inusilin in labs. One of the biggest factor of how living organism work is genetics so understanding genetic helps biochemistry an molecular biology but in other way we can solve genetic illnesses with using biochemistry and molucar biology.",
    "Biochemistry researches proteins genetics goes with the interaction of  genes become by that proteins.Molecular biology researches evolution and interaction with all genes and living things.",
    "All of them somehow interacts with living organizms. They all work on some organic compounds. What biochemistry works on could be found in living organisms and how those compounds were made in these organisms could be found in their genetic codes.",
    "its like you study physics but you need to know maths to do some of the physics quesiton . they all about the cells but they investigate them in other ways. İf you want to be better from others at any of these you need to study others to support your ideas and improve your discustion ability about that ",
    "Biochemistry focuses on the protein part of life functions. It studies the components independent of the organism. Genetics focuses on the gene part. Usually mutants are used. So, it is organism without the component. Molecular Biology integrates those two, as can be quite well ascertained from the central dogma",
    "they are connected because they all have relationship with the DNA and the atoms .all of them is a science and they are helping for get easier our lives.",
    "Biochemistry focused on the protein part of life functions. It studies the components independent of the organism. Genetics focuses on the gene part. Usually mutants are used. So, it is organism without the component. Molecular Biology integrates those two, as can be quite well ascertained from the “central dogma”.",
    "Biochemistry focuses on compounds like proteins other than genes. Molecular Biology focuses on the gene part, it ignored compounds. Genetics are the connection between Biochemistry and Molecular biology. Just like creb cycle.",
    "biochemistry is about actions into body, molecular biology is about more detailed property of body and genetics about  increase the quantities",
    "They have common point of studies: organic. They both study organic compounds and creatures. ",
    "Biochemistry's purpose is understand the human nature in case of hormons. Molecular biology can be helpful for it in case of understand the hormons nature within molecules. Genetics interested in what caused hormons to human nature.",
    "Biochemistry is related about chemical component of living creatures. Molecular biology is related about cells of living creatures and components of this cell. Genetics is related about gens order of living creatures. Molecular biology supported by biochemistry, and genetics supported by molecular biology.",
    "All of them work on  interacts with living organizms . They work on organizms.",
    "We can understand genetics and molecular biology with biochemistery. Biochemistery is needed to investigate about them. Also, molecular biology is needed to investigate genetics.",
    "All of three subjects are based on chemistry and biology. Genetics focus on gens by using molecular biology. Biochemistry is the sum of biology and chemistry. Studies of molecular biology is about cells.",
    "Genetics are the key of DNA. And DNA became from proteins. So proteins and genetics are related. This relation's name is molecular biology. Protein and function are related. If you want to understand functions you should understand the proteins. So you should look proteins in atomic. It means biochemistry. Functions and genes are related. If you want to understand them you should look at the genetics.",
    "all these thing is connected and they are connected with biology and they makes research about living thins. genetic connected with moleculer biology because base of genetic transfer is molecular biology. and moleculer biology is connected with biochemistry because when some part of gen that in transfer section biochemistry is on work.  "
    
]




model = SentenceTransformer('bert-base-nli-mean-tokens')
model2 = SentenceTransformer('all-mpnet-base-v2')
model3 = SentenceTransformer('distiluse-base-multilingual-cased-v1')
sentence_embeddings = model.encode(sentences)
sentence2_embeddings = model2.encode(sentences)
sentence3_embeddings = model3.encode(sentences)
#print(sentence_embeddings.shape)
#print(sentence_embeddings)

print(sentences[2])

print("bert-base")
print(cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
)
)

#print(sentence2_embeddings.shape)
#print(sentence2_embeddings)
'''
print("mpnet-base")
print(cosine_similarity(
    [sentence2_embeddings[0]],
    sentence2_embeddings[1:]
)
)
print("distilbert")
print(cosine_similarity(
    [sentence3_embeddings[0]],
    sentence3_embeddings[1:]
)
)
'''