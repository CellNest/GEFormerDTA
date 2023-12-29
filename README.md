# GEFormerDTA

Detailed introduction to the paper "GEFormerDTA: drug target affinity prediction based on Transformer graph for early fusion"


=========

Datasets:

=========

**Introduction**

These files were used to re-produce the results of two other methods [(Pahikkala et al., 2017)](https://academic.oup.com/bib/article/16/2/325/246479) and [(He et al., 2017)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z).

- The original Davis data and more explanation can be found [here](http://staff.cs.utu.fi/~aatapa/data/DrugTarget/).
- The original KIBA data and more explanation can be found [here](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0209-z).

**Usage**

1.&emsp;if `data/` not exist, create it by

```mkdir -p data/```

2.&emsp;download the `Davis` and `KIBA` dataset from the following link:

[https://github.com/CellNest/GEFormerDTA/tree/main/data](https://github.com/CellNest/GEFormerDTA/tree/main/data)

**Description**

**`1. profile folders`**

These profile files can be generated using [it](https://github.com/realbigws/RaptorX_Property_Fast).

(1)&emsp;overall detailed results:
		

​	SeqID.all

e.g., AAK1.all. This file contains all the detail prediction results for Secondary Structure Element (SS3 and SS8), Solvent Accessibility Surface (SAS), and Order/Disorder prediction (DISO) of AAK1.

--------------------

(2)&emsp;detailed informations in separate files:

​	SeqID.ss3<br>
​	SeqID.ss8<br>
​	SeqID.acc<br>
​	SeqID.diso

These files contain more detail proteins structure informations in the form of probability. We only use two of these files information, SeqID.ss8 and SeqID.acc.

---

(3)&emsp;simple informations in separate files:

​	SeqID.ss3_simp<br>
​	SeqID.ss8_simp<br>
​	SeqID.acc_simp<br>
​	SeqID.diso_simp

The information contained in these files is not currently used by us.

**`2. SDF folders`**

SDF (Structure Data File) is a common chemical file format used to store structural information of molecules, typically encompassing chemical structures, atomic arrangements, bond details, and more. This file format is commonly utilized for storing compound information in chemical databases. SDF files can contain various types of information, primarily including **molecular structural details**, **physical and chemical properties**, **biological activity and pharmaceutical information**, as well as **identifiers and attributes**. These SDF files can be generated using [it](https://github.com/kaiwang0112006/smilesTosdf).

**`3. map folders`**

These profile files can be generated using the [TAPE](https://github.com/songlab-cal/tape) tool. It typically contain distance information between residues within a protein structure. They provide spatial location details between residues, describing the distances among them within the protein structure, which is crucial for analyzing the structure and properties of proteins. These files can be used for simulating, analyzing structural dynamics of proteins, as well as studying protein folding and functionality.

ex: smileSeq.sdf

**`4. emb folders`**

The protein sequence matrix is mainly generated based on the one-dimensional protein sequence information in a certain encoding (for example, one-hot encoding, BPE encoding, etc.).

**`5. split folders`**

The `split/` folder mainly contains three subsets of the data set, namely the training set, verification set and test set. The format of the data set is shown in Table 1 below.

| compound_iso_smiles                      | target_name | target_sequence                                              | affinity |
| ---------------------------------------- | ----------- | ------------------------------------------------------------ | -------- |
| Nc1ncnc2c1c(-c1cnc3[nH]ccc3c1)nn2C1CCCC1 | O75116      | MSRPPPTGKMPGAPETAPGDGAGASRQRKLEALIRDPRSPINVESLLDGLNSLVLDLDFPALRKNKNIDNFLNRYEKIVKKIRGLQMKAEDYDVVKVIGRGAFGEVQLVRHKASQKVYAMKLLSKFEMIKRSDSAFFWEERDIMAFANSPWVVQLFYAFQDDRYLYMVMEYMPGGDLVNLMSNYDVPEKWAKFYTAEVVLALDAIHSMGLIHRDVKPDNMLLDKHGHLKLADFGTCMKMDETGMVHCDTAVGTPDYISPEVLKSQGGDGFYGRECDWWSVGVFLYEMLVGDTPFYADSLVGTYSKIMDHKNSLCFPEDAEISKHAKNLICAFLTDREVRLGRNGVEEIRQHPFFKNDQWHWDNIRETAAPVVPELSSDIDSSNFDDIEDDKGDVETFPIPKAFVGNQLPFIGFTYYRENLLLSDSPSCRETDSIQSRKNEESQEIQKKLYTLEEHLSNEMQAKEELEQKCKSVNTRLEKTAKELEEEITLRKSVESALRQLEREKALLQHKNAEYQRKADHEADKKRNLENDVNSLKDQLEDLKKRNQNSQISTEKVNQLQRQLDETNALLRTESDTAARLRKTQAESSKQIQQLESNNRDLQDKNCLLETAKLKLEKEFINLQSALESERRDRTHGSEIINDLQGRICGLEEDLKNGKILLAKVELEKRQLQERFTDLEKEKSNMEIDMTYQLKVIQQSLEQEEAEHKATKARLADKNKIYESIEEAKSEAMKEMEKKLLEERTLKQKVENLLLEAEKRCSLLDCDLKQSQQKINELLKQKDVLNEDVRNLTLKIEQETQKRCLTQNDLKMQTQQVNTLKMSEKQLKQENNHLMEMKMNLEKQNAELRKERQDADGQMKELQDQLEAEQYFSTLYKTQVRELKEECEEKTKLGKELQQKKQELQDERDSLAAQLEITLTKADSEQLARSIAEEQYSDLEKEKIMKELEIKEMMARHKQELTEKDATIASLEETNRTLTSDVANLANEKEELNNKLKDVQEQLSRLKDEEISAAAIKAQFEKQLLTERTLKTQAVNKLAEIMNRKEPVKRGNDTDVRRKEKENRKLHMELKSEREKLTQQMIKYQKELNEMQAQIAEESQIRIELQMTLDSKDSDIEQLRSQLQALHIGLDSSSIGSGPGDAEADDGFPESRLEGWLSLPVRNNTKKFGWVKKYVIVSSKKILFYDSEQDKEQSNPYMVLDIDKLFHVRPVTQTDVYRADAKEIPRIFQILYANEGESKKEQEFPVEPVGEKSNYICHKGHEFIPTLYHFPTNCEACMKPLWHMFKPPPALECRRCHIKCHKDHMDKKEEIIAPCKVYYDISTAKNLLLLANSTEEQQKWVSRLVKKIPKKPPAPDPFARSSPRTSMKIQQNQSIRRPSRQLAPNKPS | 13.50001 |
| ...                                      | ...         | ...                                                          | ...      |

