# GEFormerDTA

Detailed introduction to the paper "GEFormerDTA: drug target affinity prediction based on Transformer graph for early fusion"

<br><br>

=========

Datasets:

=========

1.&emsp;if `data/` not exist, create it by

```mkdir -p data/```

2.&emsp;download the `Davis` and `KIBA` dataset from the following link:

[https://github.com/CellNest/GEFormerDTA/tree/main/data](https://github.com/CellNest/GEFormerDTA/tree/main/data)

<br><br>

==================================================

Description of `profile/` , `sdf`/, `split`/, `map/` and `emb`/ folders:

==================================================

**`1. profile/`**

(1)&emsp;overall detailed results:
		

​	SeqID.all

e.g., AAK1.all. This file contains all the detail prediction results for Secondary Structure Element (SS3 and SS8), Solvent Accessibility Surface (SAS), and Order/Disorder prediction (DISO).

--------------------

(2)&emsp;detailed informations in separate files:

​	SeqID.ss3
​	SeqID.ss8
​	SeqID.acc
​	SeqID.diso

These files contain more detail proteins structure informations in the form of probability. We only use two of these files information, SeqID.ss8 and SeqID.acc.

---

(3)&emsp;simple informations in separate files:

​	SeqID.ss3_simp
​	SeqID.ss8_simp
​	SeqID.acc_simp
​	SeqID.diso_simp

The information contained in these files is not currently used by us.

**`2. sdf/`**

SDF (Structure Data File) is a common chemical file format used to store structural information of molecules, typically encompassing chemical structures, atomic arrangements, bond details, and more. This file format is commonly utilized for storing compound information in chemical databases. SDF files can contain various types of information, primarily including **molecular structural details**, **physical and chemical properties**, **biological activity and pharmaceutical information**, as well as **identifiers and attributes**.

**`3. map/`**

Map files typically contain distance information between residues within a protein structure. They provide spatial location details between residues, describing the distances among them within the protein structure, which is crucial for analyzing the structure and properties of proteins. These files can be used for simulating, analyzing structural dynamics of proteins, as well as studying protein folding and functionality.

ex: smileSeq.sdf

**`4. emb/`**

The protein sequence matrix is mainly generated based on the one-dimensional protein sequence information in a certain encoding (for example, one-hot encoding, BPE encoding, etc.).

**`5. split/`**

The `split/` folder mainly contains three subsets of the data set, namely the training set, verification set and test set. The format of the data set is shown in Table 1 below.

| compound_iso_smiles                      | target_name | target_sequence                                              | affinity |
| ---------------------------------------- | ----------- | ------------------------------------------------------------ | -------- |
| Nc1ncnc2c1c(-c1cnc3[nH]ccc3c1)nn2C1CCCC1 | O75116      | MSRPPPTGKMPGAPETAPGDGAGASRQRKLEALIRDPRSPINVESLLDGLNSLVLDLDFPALRKNKNIDNFLNRYEKIVKKIRGLQMKAEDYDVVKVIGRGAFGEVQLVRHKASQKVYAMKLLSKFEMIKRSDSAFFWEERDIMAFANSPWVVQLFYAFQDDRYLYMVMEYMPGGDLVNLMSNYDVPEKWAKFYTAEVVLALDAIHSMGLIHRDVKPDNMLLDKHGHLKLADFGTCMKMDETGMVHCDTAVGTPDYISPEVLKSQGGDGFYGRECDWWSVGVFLYEMLVGDTPFYADSLVGTYSKIMDHKNSLCFPEDAEISKHAKNLICAFLTDREVRLGRNGVEEIRQHPFFKNDQWHWDNIRETAAPVVPELSSDIDSSNFDDIEDDKGDVETFPIPKAFVGNQLPFIGFTYYRENLLLSDSPSCRETDSIQSRKNEESQEIQKKLYTLEEHLSNEMQAKEELEQKCKSVNTRLEKTAKELEEEITLRKSVESALRQLEREKALLQHKNAEYQRKADHEADKKRNLENDVNSLKDQLEDLKKRNQNSQISTEKVNQLQRQLDETNALLRTESDTAARLRKTQAESSKQIQQLESNNRDLQDKNCLLETAKLKLEKEFINLQSALESERRDRTHGSEIINDLQGRICGLEEDLKNGKILLAKVELEKRQLQERFTDLEKEKSNMEIDMTYQLKVIQQSLEQEEAEHKATKARLADKNKIYESIEEAKSEAMKEMEKKLLEERTLKQKVENLLLEAEKRCSLLDCDLKQSQQKINELLKQKDVLNEDVRNLTLKIEQETQKRCLTQNDLKMQTQQVNTLKMSEKQLKQENNHLMEMKMNLEKQNAELRKERQDADGQMKELQDQLEAEQYFSTLYKTQVRELKEECEEKTKLGKELQQKKQELQDERDSLAAQLEITLTKADSEQLARSIAEEQYSDLEKEKIMKELEIKEMMARHKQELTEKDATIASLEETNRTLTSDVANLANEKEELNNKLKDVQEQLSRLKDEEISAAAIKAQFEKQLLTERTLKTQAVNKLAEIMNRKEPVKRGNDTDVRRKEKENRKLHMELKSEREKLTQQMIKYQKELNEMQAQIAEESQIRIELQMTLDSKDSDIEQLRSQLQALHIGLDSSSIGSGPGDAEADDGFPESRLEGWLSLPVRNNTKKFGWVKKYVIVSSKKILFYDSEQDKEQSNPYMVLDIDKLFHVRPVTQTDVYRADAKEIPRIFQILYANEGESKKEQEFPVEPVGEKSNYICHKGHEFIPTLYHFPTNCEACMKPLWHMFKPPPALECRRCHIKCHKDHMDKKEEIIAPCKVYYDISTAKNLLLLANSTEEQQKWVSRLVKKIPKKPPAPDPFARSSPRTSMKIQQNQSIRRPSRQLAPNKPS | 13.50001 |
| ...                                      | ...         | ...                                                          | ...      |

