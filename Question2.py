from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd




sentences = [
    "Sensitive and Accurate measurements can be made with Gases in Reaction. Gases are also easy to capture and observe. You could get numbers using the PV=nRT Equation.",
    "Because ,volume,weight, size calculasion in the air more easy",
    "It allowed that measurement of gas with Analytical Balance. Weight, volume, pressure and templature was quantifiable now.",
    "Observing gases are more easy than the others . Also with the barometer and such tools we can measure its volume , pressure and the output of some reactions volume . Basically gases are more effective than the others.",
    "because of the easly working on gases. they can expand or change the volume. they doing a vacuum tube in the liquid when some reactions occured they can observe with liqud volume in tube.",
    "because its easy to calculate the everything and this time gas is important",
    "P.V=n.R.T   with this formula we can figure out gases exacly .gases are pressable.and heat effects it. and gases have P.  which means scientist choose to study gases. scientist wanted to know gases how to react in different conditions .",
    "Because gases are base thing in chemistry. It is huge area. Studying gases achieved many thing in chemistry. Maybe choosed for many gases are invisible and wanted to know whats behind the scene.",
    "Because there are too many unknown about gases at that time.Scientist didn't know  what air is composed of and how gases behave at certain situations. So scientist study gases to understand and predict...",
    "Because it is easy to see the relationship between inner and outer matters. They can calculate the masses easily and find the ratio.",
    "Because gases are pressable and they can fit everything. ",
    "Because most important invents found through the gas for instance O2, Hydrojen, Carbon and CO2 and these are sceleton of the Chemistry",
    "They believed that everything is composed of earth, air,fire and water which was Aristotle's thought. With refinement of the instruments, scientist began to discover gases. Then they realized air is not an element. It is a matter which is composition of elements. After that they began to discover elements.",
    "because of theirs attributes such as weight, pressure, volume and temprature. These attributes can be observed under different circumtances.",
    "Gas's relationships very low and easily study .give a shape for a gas.always same give a reaction.Gas very low volume.gas easily protect.gas easily transfer.gas easily pick.",
    "Cavendish,Robert Boyle,Antony Lavoiser, ( He found that human burn O2 and produce CO2 for create energy - and this was the first explanation for the human metabolism )",
    "Because with help of gases, we can modify Temprature and Pressure.Scientist find relation with Pressure , Volume , Heat and amount of gas(PV=n.R.T) . So we can change things with help of gases. So in our experimets we can change things easily. It is inital area for  Chemistry. If they didn't work at gases maybe we can't talk about Chemistry.",
    "Because gases are more easy to execute , save, find,easy to take out",
    "it is like you need to others to support you it is to understand micro peoples biology and  says sometimes you need to mathematical description for explain nature and this thing most unitidy from object ",
    "Because gases are compressible, flowable, their atomic structure is vacant and therefore easier to study than other states of matter.That's why chemists choose to study with gases.",
    "Because we have air all around to world. And it was unknown what air included.",
    "they wondered the gases. Chemistry is not thought without gases. Air consists of gases(O2,N2). They wondered air. They were interested in air pressure, for example Torricelli and Pascal. And also almost half of the elements is gases at 25 Celcius. Because of these sirtuations almost all of the scientist of chemistry -like Boyle, Torricelli, Pascal- interested in gases.  ",
    "Scientists choose to study gases because they wanted to understand structure of earth. Earth made up from some gases. Also they noticed that humans need oxien to live. Because of that they study gases.",
    "First of all, they thought about air because they wondered about how to breath. They believed that air is an element. So, they studied on gases. ",
    "Because gases are easier to put reactions and get results.",
    "Gases allow measurement. You can easily measure temperature, pressure, volume of a gas. With measurement you obtain numbers, with that numbers you can write new theorems.  ",
    "Because, to study with gases not expensive than other methods. Ideal gas  more suitable.",
    "Because gases are everywhere. Gases are part of our life. ",
    "in Greek, they believe people has illness because of pollution. bad air. Then they understood people ha ilness due ",
    "The most important thing about the study of gases for Chemistry, is that, it allowed measurement, you could get numbers. Weight, pressure, volume, and temperature were now quantifiable. So it is much more useful than the other forms.",
    "We can think gases like measurements . For an example, you can easily measure temperature. With this measurements scientist get some numbers which they use them at finding out theorems.",
    "Scientist can observe gases easily,and gases have weak bonds.Scientist can do something on gases easily.",
    "Because Joseph Black developed the analityic balance and thanks of analtyic balance , working gases became easy. Working gases are better than working liquid or the others due to analitic balance. 18th century lead to can measurement. Volume, pressure , weight and tempetutare became qualified. Experiment were more effective. ",
    "Because the air had always been a discussion topic. They were curious about what was in the air? Is it element or compound? They wanted to learn more about the air so they study gases. Also at first Joseph Black found something which he called fixed air and the other scientist wanted know more about that fixed air.",
    "Because it gives us numbers. We can measure volume, pressure and temperature with numbers. And the results of measurements can be seen easily.",
    "They choose gases because its easy to work with them. Atmosphere is made by gases.Hydrogen and oxygen atoms are easy to obtain. And these gases ratio is the same all around the world.Also they know more properties than solid and liquid.They can even work with inert(nobel) gases. And there were a lot of gases law like pv=nrt.",
    "Scientist choose to study gases because they are weaker and useful on the experiments then other materials.",
    "Lavoisier studided of carbon and other gases.He found that carbon is a combustion element. Also each oxygen atoms compound 2 hidroygen atoms.He discovered organik compounds.Carbons hold carbons or other atoms.",
    "Because they wondered chemical reaction  and they saw result chemical reaction with gases.",
    "It was new are for scientist. There were a lot of unknown subject in gases. There was no theory and systematic works.",
    "Because gases are main thema of chemistry. And study with gases  easier than other thinks.",
    "Scientist choose to study gasses for Chemistry.Chemistry didn't study if they didn't resarch gasses.Chemistry resarch gasses characteristic skills.Robert boyle find P.V=C its mean is if gas pressure up ,gas volume down.John dalton find law of particilar pressure.",
    "Because chemical reactions occur faster with gases. They are easier to contain. And most of the elements & compounds were found in gas forms in room temperatures.",
    "Chemistry is more complicated than other fields. Chemistry using extends.Kökeni simyaya dayanıyor bu yüzden çok köklü bir bilim dalı( ingilizcesini yazamadım kusura bakmayın).",
    "They realesed air isn't an element. It have different elements in it. And gases have different types. ",
    "Because gasses take shape of things.This is way of reach more true results.",
    "Because save gases easier than save liquids.Volume of gases are smaller than liquids.Gases are more sensitive than liquids for the experiment so scientist can reach easily to the result.",
    "because gases properties can be measured, volume, pressure.",
    "Becasuse gases are basic and simple state. Earth has lots of gases. Gases seperate each other easily. ",
    "Scientist choose to work about study of gases because pnemautic chemistry allowed to measurement in chemistry so you could getnumbers with this and you can quantify properties of  gases.With study of gases foundations of modern chemistry were laid.Also for unsderstanding how elements behave you should undestand how gases behave and what is inside of them.",
    "because , in Italy , workers try to up water to outside from inside. but water was not carry enough height. and they were curious about it .they asked about this to galileo . and galileo and later scientist tried to solved it and they were solving they discovered and invented new things about air.",
    "Because the form of gas which the particules of the elements is the most unreleated form .So forcing them to make a new bound or force them to destroy the previous bounds is much easier at this time.",
    "Gases are unknown things for them. They wanted to learn about gases more. They thougth what's the air and they wanted to learn about air. ",
    "beacuse it allowed measurement, you could get numbers studying with gases.",
    "Because Chemists need to share and compare results. And Aristotle's matter theory (Fire ,Water ,Air, ) isn't  enough. They need to define new breakthough. And they need to mathematical science.",
    "Because gases are most interesting subject in the history.We can't see and touch.So scientist's want to study gases.Chemistry work on gases because P.V=n.R.T if we work on gases we have to use temperature,mol numbers...",
    "beacuse scientists can easily have gases and some gases can give reaction easily also studiying with gases cheap. lavoiser believe that air need to life and burn . they could make some reaction with adding two gases with right volume they easily could have some mate like water and it wasw easiy and funny.",
    "Combustible gases have been generated after some experiments. Then the research started.",
    "Scientist of chemisty  choose studing with gases . Gases are  chemistry's subject and  Gases are part of living ,  all  people  need  energy for life for easier life , it can product greater machines to use gases ( example kompazasyon) all life and all world  use gases . so this is very  large subjects and  very important ",
    "Scientist choose to study gases for Chemistry because gases are complex topic and they explains Chemistry fundamentals. Also one reason is that if scientist works on gases they can understand Physic, Biology and Chemistry together and the relationship with each other.",
    "Because they discovered that many metals when dropped down into some acid or a certain liquid releases some kind of a gas. They wanted to inspect these gases and chemistry was born.",
    "The study of gases allows us to understand the behavior of matter at its simplest. And gases important for chemistry. The most important thing about pneumatic chemistry, or the study of gases, is that, it allowed measurement, you could get numbers. Weight, pressure, volume, and temperature were now quantifiable.",
    "Because they can measure them and say punctual numbers about their volume, weight,temperature and so on.",
    "Scientist choose to study gases for Chemistry in order to convert Chemistry in to a real, mathematical science. Study of gases is important because it is quantative. We can determine the quantity of gases thanks to several inventions. Scientists want to calculate and find objective results. So they want to get matematich and chemistry together in order to produce scientific equations with mathematics.",
    "Gases are more usable to change conditions and they can change the volume and pressure easily.",
    "to understand the life around us and gases are found into many ımportant thıngs so they had to study it and if you think about air it is around us like we are breathing air which is very important for living ",
    "gases is fundemantal key for chemical reaction. a lot of reaction contains the gases. even organic matters occur thanks to the atmosferic gases. as discovering the gases, a lot of mysterious can be revealed. and with discovering gases, we can explain combostion reaction and flojiston teorem. ",
    "Because gases are most stable elements. Therefore they can be found in nature without making compounds. And that makes our job easier to detect property of elements more accurately. ",
    "Because gases can be mesaured and give you numbers. Studying the gases made temperature, pressure and weight quantifiable. And if it is quantifiable, it is good subject to work on for a chemist.",
    " ",
    "Because gases are useful and they give result immediately in experiment. We can understand easily what happened in experiment with gases. They can located easily .",
    "Because gas state is so much unfamiliar than the solid state and the liquid state. This situation urged an emerge for a research on gasses. And the search for the concept of air naturally led scientist to a research on gasses.",
    "Because for a long time the only gas known was air and when they discoverd carbon dixiode they reliased there were diffirent types of air. This discovery led to curisity of air. Then they discovered element and air was not an element anymore but it was a mixture of element. It was a mile stone in science.",
    "Because gases are more light and have more specific rules. PV=nRT",
    "It allowed for measurments and it gave us numbers. Volume, temputure, weight, pressure were now quaintifiable. Study of gases made it easier to progress in chemistry because its products allowed us to make more proggress in chemistry. ",
    "the form of the gas is the most untidy form and that will make easier the things to make experiments on the choosen thing and have correct results ",
    "We can heat them , gases are flexible, they can move everywhere freely. With pressure, volume we can change them. Easy to experiment. ",
    "because gases are everywhere and gases even water is coming from two diffirent gases.Its precious for understand the nature.",
    "Gases have temperature, heat and pressure. This helps chemistry making some computes. Gases make materials for chemistry . Because of gases we can measure somethings in chemistry.",
    "Because study of gases or pneumatic chemistry allowed measures, you could get numbers. Height, volume, air pressure and temperature were now quantifible.",
    "because gases have no shape and more useful for experiments",
    "Because gases are more complicate than shapes of matter. It doesn't behave usual like others.  So, scientist choose the study on gases because of these.",
    "Cholera, Black Death etc. diseases, which caused by miasma (noxious form of bad air), are forced scientists to study gases. Gases can exist particules in their own easily, and they want to understand that. For instance, Louis Pasteur's Germ Theory; He showed that harmful gases are not reason to disease such as bad air , it caused by the particules in air. And also; weight, volume, pressure, temperature etc. can be measured by studying gases.",
    "because before the explore the gases, people thinks atmosphere is empty. However scientist think that maybe there is something in the atmosphere like water but we can not see. After that , orderly they research components of the atmosphere, flammable gases, gases pressure etc.",
    "Study of gases made it easier to progress in Chemistry because its products made more practical in Chemistry.",
    "Because, gas can measured by weight, volume, temprature and pressure. P/V=c Robert Boyle",
    "Gases is very important subject for chemistry. Scientist choose it because gases have different and usable behaviours . For example Toricelli invented barometer by using gas's behaviours and the atm was measured. Moreover they observed that some gasses occurs during some chemical operations. As an example for this Cavendish studied for inflammable gases.When some metals and acid come together, a gas (H2) occurs. ",
    "Because if they understand gases they can understand world. They believed world became from gases. If they can understand gases they can understand experiments and other natural things. Also in the air there are a lot of elements.",
    "because gases are basic type of madde and easy to work on it. and they are found free in nature. "
    
    
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
'''

