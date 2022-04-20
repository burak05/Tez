from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



sentences = [
    "Adam Smith believe that wealth was enhanced by division of labor.",
    "wealth increases by the division of labory , work ,job State just use economy for the bridge , justice",
    "Adam Smith believe that wealth was enhanced by division of labor. ",
    "According to Smith division of labors is the key of increasing the wealth.",
    "according to smith wealth icreases because of effecting of temprature pressure volume and liquid type.",
    "Its for posion gas",
    " ",
    "Wealth increases exponantial. The more you have money more you can earn. Because if you have enough money, you dont work for money, money works for you.",
    "According to Smith wealth increases by division of labor.",
    "Wealth increases if the people get the pay of their works.",
    "with studying",
    "Wealth of Nations When the nations get reach it means the people be more wealth because of expensive things more healthier than the other so when peoples get rich they can buy healthier things so they live in wealth life",
    "Adam Smith proposed Division of Labor. It provides increasing the production. Then people could produce more products in a short time. In addition, he proposed the Invisible Hand thought. It says government should not impact the economy directly. People led the way economy. ",
    "division of labor, workers should focus each spesific tasks at production stages.",
    "division of labor invisible hand (economic laws)",
    "Division of labor which increases the overall wealth of nations, and thereby raises the standart of living of capitalists and workers alike. The Invincible hand are the best regulators of economic activity. Smith promotes that state should not touch economy except justice and some limited public works (roads, bridges they has similar influence)",
    "if we specify the labor ,wealth of nation  will increase. For example when you make a car ,every worker have a different job and they have a master about their work area. Smith's ideas apply by industrial era. So in modern world we should be a speacialist.",
    " ",
    "according to smith  wealth increases when we understanding the earth and decribe the  nature  in terms of quality of life ",
    "According to Smith ; To increase it's wealth Smith argued a nation needed to expand its economic production.",
    "It increases if the goverment doesn't touch the free economy. If goverment gets into the economy more than it needed, it would lower the wealth. Wealth will increase with individual factories and other things.",
    "Adam Smith was interested in economy. They thought, between state and economy, there must not be a bond except some specific situations. This situations were justice , order and some works like make of roads and bridges.",
    "According to Smith, wealth increases.",
    "According to Smith wealth was enhanced by the division of labor.",
    " ",
    "Smith has a theory named invisible hand. If the government doesn't involve in echonomics except making roads, highways etc.  the society will produce the most efficiently according to Smith. ",
    "According to smith, division of labors which increase the overall wealth of nations.",
    "Adam Smith believe that wealth was enhanced by the division of labor.",
    "According to Smith, wealty increase with divison of labor. If one work which has more than one part, if one worker do all parts , they can do less works than one part was done by a worker and the other parts were done by the other worker.",
    "Smith believed that wealth was enhanced by the division of labor. Manufacturing can be made more profitable if the laborers are assigned specific and limited tasks. This is adopted widely during the industrial revolution. According to Smith, it is the division of labor which increases the overall wealth of nations, and thereby raises the standard of living of capitalists and workers alike. ",
    "Smith has a theory about invisible hand. If the government doesn't involve in economical things except making roads , highways etc. the society will produce the most efficiently according to the Smith.",
    " ",
    " ",
    "Welath was enhanced by the division of labor.Manufactoring can be made more profitable if the laborers are assigned to specific and limited tasks. It is the division of labor increases wealth overall. ",
    "With division of labor. The government do not make desicions on free economy except some contributes like bridges, ways and roads. The invisible hand. With division of labor. The government do not make desicions on free economy except some contributes like bridges, ways and roads. The invisible hand. ",
    "Richs get rich and poors gets poor.Smith thought like this.",
    " ",
    "Adam Smith studied history of economy,including ancient Rome,to discover patterns of economic activity. He believed that wealth enchaned division of labor.",
    " ",
    "Division of Labor : It increases the healt quality of humans. The Invisible Hands : State should not effect( interrupt  ) the economy. It makes things worst.",
    "Smith is really important for economic history. He study on roman economy and beliece division of labor. He makes really important ecenomical studies for USA.",
    "Adam Smith said wealth increases from divison of labor. He said state don't touch economy.State just look justice,bridge,road and other same things. Adam smith study history of economy.",
    "Wealth increases with division of labor. According to Smith to get the maximum efficiency we need to give right amount of work to individuals. Hence, wealth of nations will increase.",
    "Invisabel hand and labor.",
    "Adam Smith believes Invisible hand.  And it says if there are helps between humans (işbirliği ve yardımlaşma) , wealth increases.",
    " ",
    " ",
    "division of labor ",
    "According to Smith if you go high level, air pressure is increasing. ",
    "Acoording to the Smith ”wealth enchanced by the division of labor”,manufacturing can be made more profitable if the laborers are assigned to specific and limited tasks.it increases the overall wealth of nations. ",
    " ",
    " ",
    "According to Smith the state should be avoid the economy unless they made bridge and something like that. He belived invisible hand the best way. He thought that divide on work increases wealth. If they give people some divide jobs , efficient increases. This was a two ways winning according to him.",
    "with division of work weight he think if they work same. the price of goods and labor and also divison in work wealth, and theories of rent and interest. Smith believed that wealth was enhanced and increases by the division of labor. ",
    "Economy and dividing of folder is very important for Adam Smith. Developing is a possible with dividing a folder in civiliziations.",
    "Smith believed that  wealth was enhanced by the division of labor , manufacturing can be made more profitable if laborers are assigned specific and limited tasks",
    "With division tasks and let people to freely make shopping but  sometime goverment can help people  to life peacily  actually its about divison works and tasks because if a person interested in a job it will be better than a person interested in more jobs than one.",
    "According to Smith's belief, there will be a division of labor. If the division of labor is done, less effort is spent and more productivity is achieved.",
    "wealty increases is  divison of labor ,   when it willl  be  division of labor  it can be  quality of  life  for workers  for everythings",
    "According to Adam Smith, wealth increases with farming, investments etc. To increase wealth, Smith argue that a nation need to expand its economic production. Also manufacture was so important to Smith.",
    "Adam Smith has the theory named Wealth of Nations. According to this theory a person works for improving his own wealth. Every person in a society does that. So wealth increases by the prosperity of nations.",
    "According to Smith, it is the division of labor which increases the overall wealth of nations. Wealth increases is divides of work.",
    "illegally ",
    "Wealth of Nations reveals the basis of how Adam Smith thinks. Divison of labor increases the wealth of nation. He thinks that individuals should maximize profit and minimize loss , and when we do that capitalists and also workers will be rich.",
    "Wealth increases with acknowledge and science according to Smith",
    "he belived that if the work has been shared and if we give more tasks the wealth will  increases (we should share work and give the work to who can do it perfectly )",
    "according to smith, invisible of hand and divison of labour is very imortant. thanks to the these concept, wealth of country will be better.  apart from the construction road and the other building, govenment should not enter the economy. individual sucess and loses is more important. disivion of labor is very important, each worker should works its own work area and thus success will come in easy way.",
    "To increase wealth, the economy should be improved. And that depends on farm output and created manufactures through the effort.",
    "Wealth increases by division of labor and specificating the jobs of workers. He states that the state should not interfere with economy aside from providing justice and doing public services such as constructing bridges etc. ",
    " ",
    "If we put a lot of things or add things , wealth increases according to Smith.  If gravity increase , wealth increses.",
    "With the division of the labor, with a government which does not interfere economics besides to maintain the justice and with the sum of all economic gains and loses to be positive.",
    "It ıncreases if the economics controled by The Invisible Hand method and with focusing on indivisual activity . If  indivisuals focuses on certians skill rather than learning all the things and useses their capacity to fulllest overall wealth of the nation will improve. His way of looking to economics was popularized in industiral age.",
    "People which lives at same country does job without the pressure of government or  other authority.And with more free economy.",
    "Wealth increases as labor is divided across humans. By Checks and balances. And making workers work on spesific works to improve productivity. ",
    "doing a one work with one people is hard to do but with many peoples and like many brains that will be easier and  you get  more correct results . he thinks like that",
    "According to Smith, it is the division of labor which increases the overall wealth of nations, and thereby raises the standard of living of capitalists and workers alike.",
    "share the work help the increases the wealth",
    "According to Smith, it is the division of labor which increases the overall wealth of nations, and thereby raises the standard of living of capitalists and workers alike.",
    "Wealth increases according to the nation person lives and the people. I shoot my shot sir.",
    "wealth incraeses with air",
    "According the Smith; state should not rule economy. Economy should be personal. So, people can run their companies and employees on their own. They can make profit better and give best conditionals for employees. As a conclude, state, company and employees can increase their wealth. So, that causes to increase total wealth.",
    "According to Smith, wealthnes most enhanced by dividing labor. Economic health of population can increase with assigning sepsific and limited works to labors. And also, state should avoid to interference to the economic activities except basic population works such as bridges, according to Smith.",
    "Smith prefers posteriori to increasing wealth. he does not prefer intreori. he thinks goverment interests with water cannel etc. ",
    "Wealth icreases is divided across humans. Checks and balances. ",
    "With division labor and team work.",
    " ",
    "According to Smith individual activity increases economy. Because he believed if person in public works for himself, person has more benefit to the economy. And state should interferences economy instead of roads, basic economical things. If everyone in public, shares working it will benefits to capitalism and economy. Economy can grow with that. If one person try to finish all work it will decrease economy. But if people share works, economy will grow. ",
    " "
    
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