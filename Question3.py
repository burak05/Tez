from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



sentences = [
    "Morgan was trying to understand if Mendel's Genetics Hypothesis is right with Animals. He chose Fruit Flies as model animal. Because they are cheap, easy to grow, multiply fast, has short life-cycle and logistically easy to handle. Fruit Flies share 70% of the genome with Humans.",
    "because can division easly that atoms is control other atoms",
    "Because of Fruit Fly has nearly same genetics like humans. Because of similarity of genetics, he studied on Fruit Fly.",
    "Morgan choose fruit flies because they have a short life cycle . They grow fast and also dies fast.",
    "morgan choose fruit fly because of their lifetime is just 30 days. And their make too much eggs so population rapidly increasing. and it cost's is acceptable.",
    "he want to cahnge something and because fruit fly dont take polen out of the flower and you can see the changing of genetics. you can calculate genetics code",
    "awards always motivatite people .because when you do stuff and someone appreaciate you .that feels really good .and  the thing that youdone  ,your spending time worth it with awards. in this case Morgans  motivations is NOBEL. ",
    "He choosed fruit flies because they can easily groves up and when they are affordable to make an experiment. And not hard to grow up. There can be many fruit flies multiples. That was Morgans motivation ",
    "Morgan choose Fruit Flies because of a couple factor: *They are cheap. Reproduction speed of fruit flies very fast. They are very small so a simple jar could contain a lot of them. Finally studying genetics of fruit fly could give insights about human genetics because their % 70   of genes responsible for diseases.",
    "Because it is easy to see the different kind of fenotipe of fruit fly. He can change the qualification of it and see the raito between them. So he chose the fruit fly.",
    "Morgan's motivation was Nobel. He choosed fruit flies because fruit flies increase so fast and they have many obvious phenotips.",
    "Morgan used fruit flies to study how physical traits  were transmitted from parents to childrens, and he was able to show that genes are stored in chromosomes and form the basis of heredity so that's why he choosed work with Fruit Flies",
    "Fruit flies have genes which occurs diesase about %70 on humans so he could understand the human pyhsology. And as fruit flies have short life cycle, he could observe them easily.",
    "He was able to obtain a experimental large dataset because of the fruit flies laying many eggs. So he was able to quickly verify his data on very large samples. He was extremely repetitive with these experiment datasets. So he got quick results from his experiments. ",
    "this is very hard work but morgan want the begin this work . Morgan studies for future.morgan choose the Fruit Flies because Fruit Flies's %70 genetics same the humans. ",
    " ",
    "Because fruit flies have a little life of time. They live short so we can observe their life cycle easily. It can easily find so we don't use a lot of energy to find them. They produce a lot of child so we can observe their geneticts easily and find more accurate genetic statistic.",
    " ",
    " as to morgan studies  and his motivation  he think actually  it depends how the element of chemical component of ype of element or composition of things ",
    "Morgan Studies Common Fruit Fly to study genetics.His motivation could be lead to about this subject.",
    "He choosed fruit flies, because it was easy to raise and see their babies. He tried to see which specific character were more dominant to other. ",
    "Mendeleev discovered the inheritance of genes. This was the reason and motivation for Morgan. Fruit flieses is very similar to human organism and they reproduct faster than the other animals or insects. ",
    "His motivation was that he wanted to learn structure of DNA. He chose the Fruit Flies because their DNA is similar to human DNA(%70). Because of that he chose the Fruit Flies. ",
    "Because they can be a big population faster than the other variants. Also, they have specific qualities. For example eyes color. They can live average temperatures.",
    " ",
    "He choose fruit flies because they were easy to produce, they have short life cycles, and they are easy to store at huge amounts. His motivation was Mendel's work. He tought that we can understand the nature with making experiments like Mendel but on the animals. ",
    "Fruit flies's characteristic property almost the same with Human's gen or gens. Basically the range of similarity is %70. Fruit flies shares own gens with humans. Morgan believed that phrase. If reasearch gens of fruit flies, he can learn Humans Genetics.",
    "His motivation was Mendel. He choose Fruit Flies because he can watch in short time his work, for 30 days.",
    "He wanted to explain why some flies has normal eye color(red) and some flies has mutant eye color(white). he choose Fruit Flies because they small and they have short life circle (multiply fastly). Working fruit flies has more avantages as logistic.",
    "The fruit flies genetics is the closest genetic to humans genetics, that's why he choose the Fruit Flies and also Morgan decided to use fruit flies to study how physical traits (for example, eye color) were transmitted from parents to offspring, and he was able to elegantly show that genes are stored in chromosomes and form the basis of heredity.",
    "He chose it because it was useful. It was easy to store . He said that we could understand nature if we make experiments. Also Mendel thought like that.",
    "Because Morgan can grow Fruit Fly easily and Fruit Flies was so cheap.Also he can do something on Fruit Fly easily.",
    "There are a lot of varius of Fruit fly. Making experiment and seems results better than the other. Also Fruit fly growing up very fastly. So Morgan seems result easily and understood genetics better. ",
    "The motivation for his work was Mendel's work with pea plants. It encourged him to do more and learn more genetics and how sex choromosoms involved. He chose fruit flies because they had short-term life. They were able to propagate easily and fast. They were cheap. They have 70% genes that cause disease in human body. So he thought that he could learn more about human genetics by searching fruit flies.",
    "Fruit flies DNA's are very similar to human DNA. The 80 percent of human diseases can be seen on fruit flies.",
    "He choose fruit flies because they have a short life cycle. Easy to get obtain and you can see easily the difference of color.His motivation was he wanted to be a scientist and he wanted to figure this out.",
    "Because fruit fly life cycle is short.",
    " ",
    "Because the fruit fly becomes pregnant easily. ",
    "He choose fruit flies because they are growing fast and numbers of flies increasing fast. They have diseas that  %70 similar genetics of human disaeas.",
    "He's motivation is believe how genetic is important for human life and believe cominucation believe fruit-fly. He choose fruit flies because making observation is easy on fruits and taking result is fast.",
    "Morgan choosed Fruit Flies because Fruit Flies like human and they very similar to human organism and cell.So he choosed Fruit flies and his motivation for his work is this similarity.",
    "Morgan choose fruit flies because they were easy to find and produce.",
    "Because fruit flies  has many genetic extends.  ",
    "Before Morgan , Mendel did some experiments about plants. And some of people said his theories was true but not for humans / animals. And with Morgan's experiments people believe Mendel's work. They are independent persons but tehir works are connected. Morgan choose fruit flys becasuse it's genetics was similar to human genetics. And he can control and store  them easily.",
    "Because fruits multiply so many.",
    " ",
    "because they can easily replicate. they can live easy life standart. they are small. thats why we can control of fruit flies. they dont give harm",
    " ",
    "Morgan studied Common Fruit Fly to study genetics because working with fruit fly was less expensive and working with them gives him quick results.(working with have longer life cycle animal or plants more expensive it requires more time.)So he choose common fruit fly because it has shorter life cycle,cheap and reviewing them was more easier.",
    "because fruit flies can easy find, easy produced, you have lots samples for your experiment. cheap, lot of flies can put in a jam. they are crossed easily. he saw some flies has red eyes but some flies white eyes.he crossed them and results were surprising so he decided to crossed them lots of times.",
    "He studied with fruit flies because he can reach the fruit flies more easily and if he want to raise them more for testing about the genetic it won't be a problem.",
    "Morgan wanted to learn about human's DNA. He chose fruit flys because their DNA similar to human's DNA 70%. Also fruit flies live and increase fast, and it's easy to study.",
    " ",
    "He chose fruit fly because fruit fly is easily growed. Fruit fly give a lot of fruit a few times.He want to learn human' s DNA . Fruit fly 's DNA is similar to human 's DNA.  And also fruit fly' s DNA is very good for experiments.",
    "Because fruit flies life is very short so it's easy.He want to improve genetic laws so he start to work with fruit flies.Fruit flies are beautiful choice on this work.Seeing their life habits and style very easy.Because waiting time is shorter than other insects,or animals.They just lewn a few days.",
    "Because he saw that with some patient and  he think  fly and people have  some common genetics he saw that with some patient ı remembered %70 as same as people.",
    "He is motivation a beans. Because when other scineist worked a beans, he saw %75 yellow bean and %25 green bean.",
    "Morgan studies commeon fruits Flies's eyes .they research speice, colors, weights breeds    Morgan studies and  research different of   fruit flies's eyes and    He looked   red fruit  flies's eyes",
    "Morgan studies common fruit fly to study genetics. His motivation was explore the genetic secrets. He choose fruit flies because fruit flies could be found easily and they were appropriate for genetic studies.",
    "Because the fruit flies had the same (like %80) genetic diease structure as humans. So fruit flies were suitable for studying genetics.",
    "Fruit flies have a short life cycle that lasts an average of just 30 days. Fruit flies produce huge numbers of offspring. Fruit flies shares own gens with humans. Morgan believe that idea.",
    "Because it was easy to see their cells under microscope and their DNA was similar to human's DNA.",
    "Morgan Studies Common Fruit Fly to study genetics and His motivation for his work was to need of know the how genetic data transfers from parent to next generations. He choose Frit Flies because they were easy, cheap and fast to produce. There were a lot of different probability to happen and he can easiliy see that. Also when he study with Fruit Fly it was quite clear to see what happens in next generation.",
    "Morgan knows that fruit flys genetic is very similar to his tests then he used fruit fly.",
    "the motivation because the fy fruit has %70 of similars of human and he choose fly fruit because it born quickly (ıncreases easily)and die in 30 days ",
    "beacuse dropholia contain the %70 of gens that providing disease for the human. and breeding(üreme) was very fast and easy. he found the cromozom and he said that traits tranfers on X cromozoms",
    "They have simple genetical situation. And they are fast to produce.",
    "He tried to apply Mendel's genetic research to animals and see if it works on them too. He chose Fruit Flies because their life cycle is short, they reproduce much, they are genetically alike humans by 70%, and they are also cheap.",
    " ",
    "Because he thought Fruit Flies are important for genetics and biology hence they are relevant dna .  This is the most important thing of genetics of herbals.  The different various of fruits grow up with Fruit Flies.",
    "Because fruit flies have a short circle of life which makes them jump in between generations more oftenly. His motivation was to prove and enhance the Mendel Genetics.",
    " ",
    "Understand the genes and their working.He choose the fruit flies because it is similar to humans.And their genes are easier to reach.",
    "He choose fruit flys because their life cycle was short and they were easy to breed. And they had more characteristic genes. For example their eye clolor, how their wings looked. Motivation for his work was to find if some genes were affecting how living organisms looked. He wanted to find if some genes were more oppressive to others or more receceif than others. And by doing his experiments he concluded that some genes were oppresive and some were rececif.",
    "his motivation was the beens because  that the other scientist used  in their experiments they get  positive results from them like they  saw that  %75 of the beens is yellow and   %25 is green and he goes that way ",
    "Because they have short life cycle.  Morgan decided to use fruit flies to study how physical traits (for example, eye color) were transmitted from parents to offspring, and he was able to elegantly show that genes are stored in chromosomes and form the basis of heredity",
    "because there is so many fruit flys in the world and easy to found.and they have so many diffirent genotips and its easy to get so many of them.",
    "Because fruits can give fast results and they can grow faster than any animal. I didn't see the genetic pdf sir so i couldn't study about it sorry.",
    "I think his motivation was, his friends and school members in Cambridge.He chosed common fruit flies because, common fruit fly genetics were %70 identical to human disease genetics. Also common fruit flies had a short life cycle and were produced at large amount of numbers. ",
    "because they can go to other more easily and it is more helpful to his work",
    "He choose fruit flies because it's easy to product new flies for experiments and studies. Also, when you product new flies, you can product lots of at once time. Lastly, it's easy the study on flies genetics than other creatures.",
    " ",
    "Morgan choose fruit flies because they rapidly newborn so they demonstrate results of the genetic studies.",
    "Morgan studied with fruit flies because fruit flies was very cheap and also , they can find easily. ",
    "He choose fruit flies because their lifespan is short and they produced fastly. Also they have same genetics.",
    "He choose Fruit Flies because of the difference in their apperance. Some of these flies has curly wings. By crossing them he studied on genetics like mendels studies about peas. ",
    "%70 percent of trait enherited to the new generations. So Morgan studied common fruit fly because of this gen transfermations. Fruit Fly can enhireted traits to new born fruit fly, it means if you change something in parent fruit fly, it will be enhireted to the new generation. And also Morgan can find in everywhere this fly. They can quickly have a lot of new born fruit fly.",
    "because fruit flys' genetic type is so basic and easy to work. "
    
    
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