from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance



sentences = [
    "Due to internal crystal structure of the molecules.",
    "nuclears theory  3D structure cube added or removed atoms",
    "Because of count of elements. Mostly about carbon dioxide.",
    "Allotrops such as diamond and graphit they have the same chemical composition but they have a different type of crystal structures.",
    "this name is izonom or isonome this caused from difference connection type of atoms and if is it dipol.",
    "because of elements energy and pyhsical features has different reaction for every composition.",
    "for example water  and ice. both materials are same chemical composition =H2O but physical features are compelety different.and there is bağ .and  this determine  ice or water but i couldnt remember.the diffrence is heat.",
    "Weather and temparature. Because fruit flies' genetics are easily effected affected by temparature and weather. ",
    "Chemical Structure.",
    "We call that allotropism. We may say diamond and grafit as an example.",
    "They are different physical features because same chemical's connection build can be different each other.",
    "We call it isomerism. In isomerism molecules that have the same numbers of the same kinds of atoms but differ in chemical and physical properties.",
    "Same chemical composition has different physical features because they have different crystal structure.",
    "isotopes, becuase they have same atomic number but has different electron numbers.",
    "Change of genetics. Function. dominant genetics and recessive genetics. blend of genetics.modificasion",
    "They called Allotropes. Allotropes are different crystal structures of the same element. Lavoiser showed that, he showed using diamonds and graphite. Burning diamonds and graphite produced the sam gas (CO2) (This was the first example of allotropism)",
    "For example graphit and diomond have pure Carbon(6) but they have different physical compound becasue in this case Carbon(6) can make 4 bond with other Carbon(6) atoms and it changes the physical features . So we can say that in compound , shape of bounding is matters.",
    "differences in their struct differences in their chemical connection",
    "because when atoms combined with each other their  composition is different for example if ""h=h and h-h"" their physical features will be different ",
    "Because the atomic weights and molecular weights of the compounds may be different. The bond types and bond numbers between atoms or molecules may be different.",
    "It might be from temparature. Or it might change by how much material we have. And the ratio between them.",
    "allotrophism. they have different structures. But they include same elements.",
    "Allotropes. They have same chemical composition but different physical features.",
    "It depends on bounds. Bounds depend on locations of elements and we call it allotropism.",
    "Because they have different allotrops which is changes the physical features. For example coil and diamond.",
    "Difference in activation energy in a chemical composition can change the compound in the result. For example: we can burn carbon to gain CO (Carbon monoxide) or to gain CO2 (Carbon dioxide) if we give to the reaction different amount of activation energy we can gain one of them.",
    "When the atom joined, then it lose own properties. So  its characteristic property change. Actually it looks same physcial property but chemical property is different.",
    "Because atoms are in different places and they have different bonds.",
    "Allotrops.",
    "Atoms of the same element are alike( I mean they are also different but has something in common ), but atoms of each element different one from the other in size and weight.",
    " ",
    "When the elements joined,the atoms lose their charachteristic proporties.",
    "Molecular bonds are covalent and ionic lead to different pyhsical properties. Different weight , structure and can oxcid.(I mean oksitlenebilme in Turkish). ",
    "The way that atoms are bonded are different. For example diamond and graphite has the same chemical composition but they have different properities. They are different structures of the same element and it is called allotropy.",
    "Allotropism. Difference in diziliş. Difference in molecular shape of the matter. Like coal and diamond. Those two are only carbons. But their molecular shapes are different.",
    "Allatropism.Its a different crystal structure of an element.An example coal(grafit) and diamond.They are both made by carbon atoms but their crystal structure is different.So their physical features is different. The heat and pressure on coal turning this coal into diamond, and it takes so much time.",
    " ",
    "Because elements should behave of differences.And their weight are should be different.The radioactive is change their characteristic.Every elements feature are difference from others.",
    "Shape, molecular built, freezing point",
    "Like grafit and diamond. They made from carbon but have different physical apperences and when you burn them same gases out come. The reason is chemical bond differences . ",
    "Because they are in different area.(high,P,d)",
    "allotrophism (allotropism). Anthony lavoisier found and explain it. He burned diamound and carbon and he see same element carbon dioxid.Carbon and diamound released same gas.",
    "They are called allotropes. Their features change due to differences in their chemical structures.",
    "Chemical slecture and extends.",
    "They called allotrops. Their physical features was different because they can occure in different orans ( farklı oranlarda birleşebiliyorlar.) . (kimyasal bağları ve çekimleri yüzünden farklılar.)",
    "Distance between cells are different.",
    "If same chemical composition been in different place , these are going to behavior different. It is due to temperature , high from sea distance . I",
    "because of allotrope of elements",
    "Because elements can be make ionik or covalent band. If there are metal and nonmetal, they can be ionik band. If there are two nonmetal, they can be covalent  band. They could be izotop each other. Izotop is, in the same compositions proton counts are same but mass numbers are different.",
    "While working with diamond and graphite Lavoiser noticed that when burned them they released same type gases.(allotropism)Then lieb and Wöhler were about Two different compounds with same percentages of elements.this implies compounds have different arrangements for have different structurals. Because of the stereochemistry.İn a nutshell it is like this they have diiferent physical features because they different arrangement of atoms their 3D structure is different.  ",
    "Allotrop. they have the same atoms but  structure of atoms, their bonds is different each other.",
    "The element particuleses got bounds with different ways.",
    "This differences in the strecture make this. If the same chemical composition but different strecture its become different physical features. ",
    "diiference in chemical connecting type. it names is allotrop if there is chemicial composition but physical features of compounds are due to different it means they are allotrops.",
    "Difference is structure of atoms and arrengement of atoms. And also smell,taste can be different.",
    "Chemical composition and physical features are not connected each other.They just represent  each other.",
    "I think it says allotrop beacuse there arre some carbon but became different mate  it's about their connecting differences for example carbon has 4 connector sometimes it has 4 connects sometimes 2 times double connects so it makes mate different from each other. if it's not about allotrop. Maybe it is about during connecting time's air pressure or temparture differences beacuse they are effecting.",
    "Cells of plants and animals are almost the same, but physically very different from each other.",
    " alllogritms  is  same  chemical composittion but  different compounds because Carbon element is different order so    example diamong and grafiti",
    "Same chemical composition but different physical features of compounds are due to differences in isomerism. Its called Isomerism. Isomerism, the molecules that have same numbers of the same kinds of atoms but difference in chemical and physical properties. Energy are also is a factor in isomerism.",
    "It is called something like allotropes. It is due to differences in molecular structures.",
    "Some of chemical composition are similar but there are difference among their. When the elements are joined, the atoms lose their individual properties and have different properties from the elements they are composed of. A chemical formula is used a quick way to show the composition of compounds. Letters, numbers, and symbols are used to represent elements and the number of elements in each compound.",
    "Because of allotropism . We see that their chemical formulas are the same but their featuers are completly different.",
    "Same chemical composition but different physical features of compounds are due tou differences in period number. Elements which in the same group acts in a similar way as chemical composition. Because the number of atoms they have in the last orbital is same they have same chemical composition. But the total quantity of atoms they have is different so their atomic number is different and they have different physical features of compounds.",
    "Same chemical composition with different physical features gives different results",
    "covid (virus)",
    "allographics composition has different features of compounds for such as elmas and grafit. they have different arragement of atoms and they have different crystial structures. elmas and grafit is same chemical comporisiton but they have different arragment of atoms. so elmas is sert but grafit is yumuşak",
    "Due to bonds in-between. And bonds depends on temperature and shape.",
    "It is due to allotropy. Differences in either bound lines and thus the differences in 3D models of the elements changes the physical features of compounds.",
    "their atoms are attached eachother in different ways.",
    "This is allotropes , ı could not remember correct name of this.",
    "Chemical Structure. For example when we heat some certain compounds, we get a different compound. But they will share the same formula. What differs between those two compounds is their Chemical Structure.",
    "Allatrop . For example diamond and coal. Even though  they are both made of carbon they are totaly different.",
    "They are same in chemical but have differences in physical.They might have different bonds with same atoms which called polarity and dipolarity.",
    "Due to diffrences in how their crystal structures were bonded-made.(Allotropizm)",
    "İf it is the same chemical composition the type of the elements and the numbers of the elements on this composition is same so it depends only how they connected to themself",
    "Compounds by elemental analysis, color, crystal form, melting point, boiling point, taste, smell.",
    "this is called allotropism for example diamond and  k      .  its because of atoms order and any other species",
    "When the elements are joined, the atoms lose their individual properties and have different properties from the elements they are composed of. A chemical formula is used a quick way to show the composition of compounds. Letters, numbers, and symbols are used to represent elements and the number of elements in each compound.",
    "Because of the polarity and dipolarity. When we look at the geometry of compounds, we can see bonds and angles that makes the compound itself. Also different structures as well.",
    "they are due to differences in allotrops' s properties",
    "Chemical bonds cause these different physical features. Compositions may have same kind and many of elements but they might have different physical features. Different kinds of bonds and different bonding styles on same elements causes this differs.",
    "Allotropism. Allotrops have the same proposition in chemical reactions, but different physical features. For instance, graphit and diamond produced same gas (CO2). Lavoisier were explained allotrops first, as well.",
    "because this compounds have different location in space. like mirror or left hand right hand. Sometimes this compounds have different chemical bonds.",
    "due to differences are. ",
    "Due to stereo (3D) structure. It has same atoms but structure is different.",
    "Because their structures are different. The connection type between elements are different. So in spite of same chemical composition and same close formulation they are different.",
    "Allotropes are same chemical composition's different structed styles. For example carbon has two different physical shape because of the heat and force on carbon. If you put carbon in bottom of the down and wait too long it will be diamond. So time, force on it and heat are the most important things in this differences.",
    "isomerism, the existence of molecules that have the same numbers of the same kinds of atoms  hence the same formula but differences  in chemical and physical properties."
    
    
    
    
    
    
    
    
    
    
    
    
]




model = SentenceTransformer('bert-base-nli-mean-tokens')
model2 = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
model3 = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
sentence_embeddings = model.encode(sentences)
sentence2_embeddings = model2.encode(sentences)
sentence3_embeddings = model3.encode(sentences)
#print(sentence_embeddings.shape)
#print(sentence_embeddings)

print(sentences[1])

print("bert-base")

print(cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
)
)
"""
print("multi-qa")
print(cosine_similarity(
    [sentence2_embeddings[0]],
    sentence2_embeddings[1:]
)
)

#print(sentence2_embeddings.shape)
#print(sentence2_embeddings)
"""
"""
print("paraphrase")
print(cosine_similarity(
    [sentence3_embeddings[0]],
    sentence3_embeddings[1:]
)
)
"""
print("jaccard")
print(textdistance.jaccard(sentences[0], sentences[2]))