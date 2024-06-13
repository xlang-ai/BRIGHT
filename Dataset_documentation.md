# BRIGHT
BRIGHT is the first text retrieval benchmark that requires intensive reasoning to retrieve relevant documents. 
The queries are collected from diverse domains (StackExchange, LeetCode, and math competitions), all sourced from realistic human data.
Experiments show that existing retrieval models perform poorly on BRIGHT, where the highest score is only 21 measured by nDCG@10.
BRIGHT provides a good testbed for future retrieval research in more realistic and challenging settings.

#### [Dataset Link](https://huggingface.co/datasets/xlangai/BRIGHT)

#### Data Card Author(s)
<!-- info: Select **one role per** Data Card Author:

(Usage Note: Select the most appropriate choice to describe the author's role
in creating the Data Card.) -->
<!-- width: half -->
- **Hongjin Su, HKU:** Owner
- **Howard Yen, Princeton:** Owner
- **Mengzhou Xia, Princeton:** Owner
- **Weijia Shi, UW:** Contributor
- **Niklas Muennighoff:** Contributor
- **Han-yu Wang, HKU:** Contributor
- **Haisu Liu, HKU:** Contributor
- **Quan Shi, Princeton:** Contributor
- **Zachary S. Siegel, Princeton:** Contributor
- **Michael Tang, Princeton:** Contributor
- **Ruoxi Sun, Google:** Contributor
- **Jinsung Yoon, Google:** Contributor
- **Sercan Ö. Arik, Google:** Contributor
- **Danqi Chen, Princeton:** Contributor
- **Tao Yu, HKU:** Contributor

## Authorship
### Publishers
#### Publishing Organization(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the institution or organization responsible
for publishing the dataset: -->
The University of Hong Kong

#### Industry Type(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** industry types to which the publishing
organizations belong: -->
- Academic - Tech

#### Contact Detail(s)
<!-- scope: microscope -->
<!-- info: Provide publisher contact details: -->
- **Publishing POC:** N/A
- **Affiliation:** N/A
- **Contact:** N/A
- **Mailing List:** N/A
- **Website:** N/A

### Dataset Owners
#### Team(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the groups or team(s) that own the dataset: -->
Hongjin Su, Howard Yen and Mengzhou Xia

#### Contact Detail(s)
<!-- scope: periscope -->
<!-- info: Provide pathways to contact dataset owners: -->
- **Dataset Owner(s):** Hongjin Su, Howard Yen and Mengzhou Xia
- **Affiliation:** The University of Hong Kong and Princeton University
- **Contact:** hjsu@cs.hku.hk, {hyen,mengzhou}@cs.princeton.edu
- **Group Email:** N/A
- **Website:** N/A

#### Author(s)
<!-- scope: microscope -->
<!-- info: Provide the details of all authors associated with the dataset:

(Usage Note: Provide the affiliation and year if different from publishing
institutions or multiple affiliations.) -->
- Hongjin Su, PhD student, The University of Hong Kong
- Howard Yen, Masters student, Princeton University
- Mengzhou Xia, PhD student, Princeton University

### Funding Sources
#### Institution(s)
<!-- scope: telescope -->
<!-- info: Provide the names of the funding institution(s): -->
- Princeton University
- Google cloud AI research

#### Funding or Grant Summary(ies)
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide a short summary of programs or projects that may have funded
We thank the Princeton University and Google cloud AI research for supporting Nvidia GPUs to benchmark retrieval models on BRIGHT.

## Dataset Overview
#### Data Subject(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable**** subjects contained the dataset: -->
- Non-Sensitive Data about people
- Public data accessible to everyone

#### Dataset Snapshot
<!-- scope: periscope -->
<!-- info: Provide a snapshot of the dataset:<br><br>(Use the additional notes
to include relevant information, considerations, and links to table(s) with
more detailed breakdowns.) -->
Category | Data
--- | ---
Size of Dataset | 607 MB
Number of Instances | 1322
Number of Fields | 6
Domains | 11

**Above:** We collect 1322 diverse queries from realistics human data. Each example is annotated with the gold documents and the reasning traces to fine them.

#### Content Description
<!-- scope: microscope -->
<!-- info: Provide a short description of the content in a data point: -->
The datasets are collected from [StackExchange](https://stackexchange.com/), [TheoremQA](https://arxiv.org/abs/2305.12524), [LeetCode](https://leetcode.com/) and [Math competition](https://artofproblemsolving.com/).

#### Descriptive Statistics
<!-- width: full -->
<!-- info: Provide basic descriptive statistics for each field.

Use additional notes to capture any other relevant information or
considerations.

Usage Note: Some statistics will be relevant for numeric data, for not for
strings. -->

Dataset | # Q | # D | # D+ | Q.L. | D.L. | 
--- |-----|-----|------|------|------| 
Biology | 103 | 57,364 | 3.6 | 83.6 | 115.2 |
Earth Science | 118 | 122,388 | 7.7 | 132.4 | 113.3 |
Economics | 103 | 50,221 | 8.0 | 120.2 | 181.5 |
Psychology | 101 | 52,841 | 7.3 | 118.2 | 149.6 | 
Robotics | 101 | 62,198 | 5.5 | 120.6 | 818.9 |
Stack Overflow | 117 | 101,100 | 7.0 | 704.5 | 478.3 | 
Sustainable Living | 108 | 60,732 | 5.6 | 108.0 | 148.5 |
LeetCode | 142 | 413,932 | 1.8 | 483.1 | 497.5 |
Pony | 112 | 7,894 | 22.5 | 98.3 | 102.6 |
AoPS | 111 | 188,177 | 4.7 | 89.0 | 250.5 |
TheoremQA | 206 | 188,177 | 3.2 | 117.1 | 250.5

Data statistics of BRIGHT
For each dataset, we show the number of queries (# Q) and documents (# D), 
the average number of positive documents (# D+) per example, 
the average length of queries (Q.L.) and documents (D.L., measured by the GPT-2 tokenizer)


### Sensitivity of Data
#### Sensitivity Type(s)
<!-- scope: telescope -->
<!-- info: Select ***all applicable*** data types present in the dataset: -->
- None

#### Field(s) with Sensitive Data
<!-- scope: periscope -->
<!-- info: List fields in the dataset that contain S/PII, and specify if their
collection was intentional or unintentional.

Use additional notes to capture any other relevant information or
considerations. -->
- None

**Intentional Collected Sensitive Data**

- None

**Unintentionally Collected Sensitive Data**

- None


#### Security and Privacy Handling
<!-- scope: microscope -->
<!-- info: Summarize the measures or steps to handle sensitive data in this
dataset.

Use additional notes to capture any other relevant information or
considerations. -->

We select academia-oriented domains and remove all user information in StackExchange data.

#### Risk Type(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** risk types presenting from the
dataset: -->
- No Known Risks

#### Supplemental Link(s)
<!-- scope: periscope -->
<!-- info: Provide link(s) for documentation pertaining to sensitive data in
the dataset: -->
- None

#### Risk(s) and Mitigation(s)
<!-- scope: microscope -->
<!-- info: Summarize the steps taken to identify and mitigate risks from PII
or sensitive information.

Use additional notes to capture any other relevant information or
considerations. -->
- N/A

### Dataset Version and Maintenance
#### Maintenance Status
<!-- scope: telescope -->
<!-- info: Select **one:** -->

**Actively Maintained** - No new versions will be made
available, but this dataset will
be actively maintained,
including but not limited to
updates to the data.

#### Version Details
<!-- scope: periscope -->
<!-- info: Provide details about **this** version of the dataset: -->
**Current Version:** 1.0

**Last Updated:** 06/2024

**Release Date:** 06/2024

#### Maintenance Plan
<!-- scope: microscope -->
<!-- info: Summarize the maintenance plan for the dataset:

Use additional notes to capture any other relevant information or
considerations. -->
We will mainly use Github issues and huggingface communities to address any issue the users encounter in using the BRIGHT data.

**Versioning:** If new versions are released, it will become 1.1 or 2.0 depending on the update.

**Updates:** There may be updates in the future.

**Errors:** We will address the error users encounter

**Feedback:** Either by email, Github issue, huggingface community, we welcome all fedback to make this benchmark better

#### Next Planned Update(s)
<!-- scope: periscope -->
<!-- info: Provide details about the next planned update: -->
**Version affected:** N/A

**Next data update:** N/A

**Next version:** N/A

**Next version update:** N/A

#### Expected Change(s)
<!-- scope: microscope -->
<!-- info: Summarize the updates to the dataset and/or data that are expected
on the next update.

Use additional notes to capture any other relevant information or
considerations. -->
**Updates to Data:**  N/A

**Updates to Dataset:** N/A

**Additional Notes:** Add here

## Example of Data Points
#### Primary Data Modality
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Text Data

#### Sampling of Data Points
<!-- scope: periscope -->
<!-- info: Provide link(s) to data points or exploratory demos: -->
- [Typical Data Points Link](https://huggingface.co/datasets/xlangai/BRIGHT/blob/main/data_examples.pdf)


#### Typical Data Point
<!-- width: half -->
<!-- info: Provide an example of a typical data point and describe what makes
it typical.

**Use additional notes to capture any other relevant information or
considerations.** -->
Summarize here. Include any criteria for typicality of data point.

```
{
  "query": "Claim in article about why insects are attracted to light\nIn this article they are addressing the reason insects are attracted to light when they say\nHeat radiation as an attractive component is refuted by the effect of LED lighting, which supplies negligible infrared radiation yet still entraps vast numbers of insects.\nI don't see why attraction to LEDs shows they're not seeking heat. Could they for example be evolutionarily programmed to associate light with heat? So that even though they don't encounter heat near/on the LEDs they still \"expect\" to?",
  "reasoning": "The question probes why insects are drawn to low-heat LED lights, challenging the idea that their attraction to light is heat-based. The document helps distinguish between heat attraction and evolved behaviors, shedding light on why insects might be attracted to LEDs despite their minimal heat.",
  "id": "0",
  "excluded_ids": [
    "N/A"
  ],
  "gold_ids_long": [
    "insects_attracted_to_light/Proximate_and_ultimate_causation.txt",
    "insects_attracted_to_light/Phototaxis.txt"
  ],
  "gold_ids": [
    "insects_attracted_to_light/Phototaxis_3.txt",
    "insects_attracted_to_light/Proximate_and_ultimate_causation_0.txt",
    "insects_attracted_to_light/Phototaxis_4.txt",
    "insects_attracted_to_light/Proximate_and_ultimate_causation_1.txt",
    "insects_attracted_to_light/Phototaxis_0.txt"
  ]
}
```

## Motivations & Intentions
### Motivations
#### Purpose(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Research

#### Domain(s) of Application
<!-- scope: periscope -->
<!-- info: Provide a list of key domains of application that the dataset has
been designed for:<br><br>(Usage Note: Use comma-separated keywords.) -->

`retrieval`

#### Motivating Factor(s)
<!-- scope: microscope -->
<!-- info: List the primary motivations for creating or curating this dataset:

(Usage Note: use this to describe the problem space and corresponding
motivations for the dataset.) -->
* Existing retrieval benchmarks can be solved by lexical or semantic match
* Many realistic scenarios cannot be solved by such simple match
* To bridge this gap, we introduce BRIGHT to evaluate retrieval models in realistic settings where intensive reasoning is required

### Intended Use
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Evaluate retrieval systems in realistic scenarios

#### Suitable Use Case(s)
<!-- scope: periscope -->
<!-- info: Summarize known suitable and intended use cases of this dataset.

Use additional notes to capture any specific patterns that readers should
look out for, or other relevant information or considerations. -->
**Suitable Use Case:** Evaluate retrieval models

#### Unsuitable Use Case(s)
<!-- scope: microscope -->
<!-- info: Summarize known unsuitable and unintended use cases of this dataset.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
**Unsuitable Use Case:** Train retrieval models

#### Research and Problem Space(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the specific problem space that this
dataset intends to address. -->
We investigate new directions of retrieval, where the relevance between queries and documents go beyond lexical and semantic similarities.

#### Citation Guidelines
<!-- scope: microscope -->
<!-- info: Provide guidelines and steps for citing this dataset in research
and/or production.

Use additional notes to capture any specific patterns that readers should look
out for, or other relevant information or considerations. -->
**Guidelines & Steps:** Include citation when using BRIGHT

**BiBTeX:**
```
@inproceedings{BRIGHT,
  title={BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval},
  author={Su, Hongjin and Yen, Howard and Xia, Mengzhou and Shi, Weijia and Muennighoff, Niklas and Wang, Han-yu and Liu, Haisu and Shi, Quan and Siegel, Zachary S and Tang, Michael and Sun, Ruoxi and Yoon, Jinsung and Arik, Sercan O and Chen, Danqi and Yu, Tao},
  year={2024},
}
```

## Access, Rentention, & Wipeout
### Access
#### Access Type
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- External - Open Access

#### Documentation Link(s)
<!-- scope: periscope -->
<!-- info: Provide links that describe documentation to access this
dataset: -->
- Dataset Website URL: https://huggingface.co/datasets/xlangai/BRIGHT
- GitHub URL: https://github.com/xlang-ai/BRIGHT

#### Prerequisite(s)
<!-- scope: microscope -->
<!-- info: Please describe any required training or prerequisites to access
this dataset. -->

N/A

#### Policy Link(s)
<!-- scope: periscope -->
<!-- info: Provide a link to the access policy: -->
- Direct download URL: https://huggingface.co/datasets/xlangai/BRIGHT

Code to download data:
```
from datasets import load_dataset
data = load_dataset('xlangai/BRIGHT', 'examples')['biology']
```

#### Access Control List(s)
<!-- scope: microscope -->
<!-- info: List and summarize any access control lists associated with this
dataset. Include links where necessary.

Use additional notes to capture any other information relevant to accessing
the dataset. -->
N/A

### Retention
Free retention

### Wipeout and Deletion
We are not currently considering wiping out or deleting the data

## Provenance
### Collection
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used to collect data: -->
- Data are collected by authors

#### Methodology Detail(s)
<!-- scope: periscope -->
<!-- info: Provide a description of each collection method used.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for collection method
type.) -->
**Collection Type**

**Source:** [StackExchange](https://stackexchange.com/), [TheoremQA](https://arxiv.org/abs/2305.12524), [LeetCode](https://leetcode.com/) and [Math competition](https://artofproblemsolving.com/).

**Platform:** N/A

**Is this source considered sensitive or high-risk?** No

**Dates of Collection:** 2024.03~2024.05

**Primary modality of collection data:**

*Usage Note: Select one for this collection type.*

- Text Data

**Update Frequency for collected data:**

*Usage Note: Select one for this collection type.*

- Static

**Additional Links for this collection:**

N/A

#### Source Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of each upstream source of data.

Use additional notes to capture any other relevant information or
considerations. -->
- **Source:** StackExchange is a popular question-answering platform where users ask questions and receive
answers from the community. [One example](https://sustainability.stackexchange.com/questions/4691/how-good-is-it-to-reuse-water-from-plant-pots) is
```
How good is it to reuse water from plant pots?

I'm living in an apartment, and after I water my plants the water goes to plates below the pots. The pots are in a metallic structure above the plates, so I can take the plates to reuse the water (throwing it at the plants again).

This reuse seems beneficial, because I think I can get rid of mosquitoes that would reproduce in the stagnated water. And also some nutrients of the soil (as well as earthworms) can return to the vase.

Is there some negative points in doing that?

EDIT: I think I must add that I'm at 3 degrees of latitude, in a hot and humid tropical rainforest, where the precipitation used to be around 1700 mm. So I use lots of water everyday, more than once a day sometimes, so the reused water is a small fraction of the water used.

waterreuseplants
Share
Improve this question
Follow
edited Mar 17, 2016 at 15:27
asked Sep 3, 2015 at 18:39
Rodrigo's user avatar
Rodrigo
16311 silver badge66 bronze badges
i think you mean "pots" if they have dirt in them. "vases" hold water and cur flowers. – 
Kate Gregory
 Mar 17, 2016 at 14:53
Yes, @KateGregory, you're absolutely right. That's because in Portuguese we call them "vasos" :) – 
Rodrigo
 Mar 17, 2016 at 15:25
Add a comment
2 Answers
Sorted by:

Highest score (default)
7

In my experience plants suffer in the long term from accumulation of salts in the soil, so fresh water would be better than reusing the water. Even better would be to get hold of fresh rain water (tricky in an apartment though, unless perhaps you have a balcony that gets rained on) for watering them, as that won't contain the salts that tap water does.

More detail here.

Share
Improve this answer
Follow
```
- **Source:** LeetCode is a popular coding platform for programmers to practice. One example is:
```
5. Longest Palindromic Substring
Medium
Topics
Companies
Hint
Given a string s, return the longest 
palindromic
 
substring
 in s.

 

Example 1:

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
Example 2:

Input: s = "cbbd"
Output: "bb"
 

Constraints:

1 <= s.length <= 1000
s consist of only digits and English letters.
```
- **Source:** AoPS contains math competition questions. [One example]() is:
```
Problem 1
What is the ones digit of 222,22 -222,222, -2,222,-222222 \
A. 0
B. 2
C. 4
D. 6
E. 8      
        

Solution 1
We can rewrite the expression as\[222,222-(22,222+2,222+222+22+2).\]
We note that the units digit of the addition is $0$ because all the units digits of the five numbers are $2$ and $5*2=10$, which has a units digit of $0$.
Now, we have something with a units digit of $0$ subtracted from $222,222$. The units digit of this expression is obviously $2$, and we get $\boxed{B}$ as our answer.
```

#### Collection Cadence
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
**Static:** Data was collected once from single or multiple sources.

#### Data Integration
<!-- scope: periscope -->
<!-- info: List all fields collected from different sources, and specify if
they were included or excluded from the dataset.

Use additional notes to
capture any other relevant information or considerations.

(Usage Note: Duplicate and complete the following for each upstream
source.) -->
**Source**

**Included Fields**

Data fields that were collected and are included in the dataset.

Field Name | Description
--- | ---
Post | The content of post where users ask questions


**Additional Notes:** Add here

**Excluded Fields**

Data fields that were collected but are excluded from the dataset.

Field Name | Description
--- | ---
Answer | Community answers
Votes | The votes for the post of answers

#### Data Processing
<!-- scope: microscope -->
<!-- info: Summarize how data from different sources or methods aggregated,
processed, or connected.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the following for each source OR
collection method.) -->

All the data collection and processing are done manually or with the help of python scripts.

### Collection Criteria
#### Data Selection
<!-- scope: telescope -->
<!-- info: Summarize the data selection criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- **StackExchange:** We select posts that have links in answers receiving user accept or more than 5 votes
- **Math and Code:** We select questions that require a theorems of syntax documentation.

#### Data Inclusion
<!-- scope: periscope -->
<!-- info: Summarize the data inclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- We include data from diverse domains including psychology, robotics, etc.

#### Data Exclusion
<!-- scope: microscope -->
<!-- info: Summarize the data exclusion criteria.

Use additional notes to capture any other relevant information or
considerations. -->
- We exclude examples that do not require reasoning in retrieval or do not use theorems.

### Relationship to Source
#### Use & Utility(ies)
<!-- scope: telescope -->
<!-- info: Describe how the resulting dataset is aligned with the purposes,
motivations, or intended use of the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- **StackExchange:** We use the post and linked web pages in answers
- **Math & Code:** We use the questions and tags in websites.

#### Benefit and Value(s)
<!-- scope: periscope -->
<!-- info: Summarize the benefits of the resulting dataset to its consumers,
compared to the upstream source(s).

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
- Using this method, we collect retrieval instances that require intensive reasoning to retrieve documents

#### Limitation(s) and Trade-Off(s)
<!-- scope: microscope -->
<!-- info: What are the limitations of the resulting dataset to its consumers,
compared to the upstream source(s)?

Break down by source type.<br><br>(Usage Note: Duplicate and complete the
following for each source type.) -->
- The judgement of relevance can be subjective, leading to non-perfect human performance.

### Version and Maintenance
<!-- info: Fill this next row if this is not the first version of the dataset,
and there is no data card available for the first version -->
#### First Version
<!-- scope: periscope -->
<!-- info: Provide a **basic description of the first version** of this
dataset. -->
- **Release date:** 06/2024
- **Link to dataset:** BRIGHT 1.0: https://huggingface.co/datasets/xlangai/BRIGHT
- **Status:** Actively Maintained
- **Size of Dataset:** 607 MB
- **Number of Instances:** 1322

#### Note(s) and Caveat(s)
<!-- scope: microscope -->
<!-- info: Summarize the caveats or nuances of the first version of this
dataset that may affect the use of the current version.

Use additional notes to capture any other relevant information or
considerations. -->
None

#### Cadence
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Daily

#### Last and Next Update(s)
<!-- scope: periscope -->
<!-- info: Please describe the update schedule: -->
- We have not updated the datasets since release.

#### Changes on Update(s)
<!-- scope: microscope -->
<!-- info: Summarize the changes that occur when the dataset is refreshed.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each source type.) -->
N/A

## Human and Other Sensitive Attributes
#### Sensitive Human Attribute(s)
<!-- scope: telescope -->
<!-- info: Select **all attributes** that are represented (directly or
indirectly) in the dataset. -->
None

#### Intentionality
<!-- scope: periscope -->
<!-- info: List fields in the dataset that contain human attributes, and
specify if their collection was intentional or unintentional.

Use additional notes to capture any other relevant information or
considerations. -->
**Intentionally Collected Attributes**

We only use human-labeled links or tags to find examples or documents, but not directly include human labels.

**Unintentionally Collected Attributes**

None

#### Rationale
<!-- scope: microscope -->
<!-- info: Describe the motivation, rationale, considerations or approaches
that caused this dataset to include the indicated human attributes.

Summarize why or how this might affect the use of the dataset. -->
We follow links or tags to find relevant documents or examples

#### Source(s)
<!-- scope: periscope -->
<!-- info: List the sources of the human attributes.

Use additional notes to capture any other relevant information or
considerations. -->
None

#### Methodology Detail(s)
<!-- scope: microscope -->
<!-- info: Describe the methods used to collect human attributes in the
dataset.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each human
attribute.) -->

We follow links or tags to find relevant documents or examples

#### Distribution(s)
<!-- width: full -->
<!-- info: Provide basic descriptive statistics for each human attribute,
noting key takeaways in the caption.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each human
attribute.) -->
N/A

#### Known Correlations
<!-- scope: periscope -->
<!-- info: Describe any known correlations with the indicated sensitive
attributes in this dataset.

Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate for each known correlation.) -->
[`query`, `gold_ids`, `gold_ids_long`]

**Description:** The documents corresponding to `gold_ids` or `gold_ids_long` are relevant to queries.

**Impact on dataset use:** It helps evalute retrieval models in realistic setting.

#### Risk(s) and Mitigation(s)
<!-- scope: microscope -->
<!-- info: Summarize systemic or residual risks, performance expectations,
trade-offs and caveats because of human attributes in this dataset.

Use additional notes to capture any other relevant information or
considerations.

Usage Note: Duplicate and complete the following for each human attribute. -->
**Human Attribute**

None

## Extended Use
### Use with Other Data
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to use with other data

#### Known Safe Dataset(s) or Data Type(s)
<!-- scope: periscope -->
<!-- info: List the known datasets or data types and corresponding
transformations that **are safe to join or aggregate** this dataset with. -->
The data in BRIGHT benchmark focus on academia-oriented domains, and they should be safe.

#### Best Practices
<!-- scope: microscope -->
<!-- info: Summarize best practices for using this dataset with other datasets
or data types.

Use additional notes to capture any other relevant information or
considerations. -->
Evaluate retrieval systems on BRIGHT.

#### Known Unsafe Dataset(s) or Data Type(s)
<!-- scope: periscope -->
<!-- info: Fill this out if you selected "Conditionally safe to use with other
datasets" or "Should not be used with other datasets":

List the known datasets or data types and corresponding transformations that
are **unsafe to join or aggregate** with this dataset. -->
None

#### Limitation(s) and Recommendation(s)
<!-- scope: microscope -->
<!-- info: Fill this out if you selected "Conditionally safe to use with
other datasets" or "Should not be used with
other datasets":

Summarize limitations of the dataset that introduce foreseeable risks when the
dataset is conjoined with other datasets.

Use additional notes to capture any other relevant information or
considerations. -->
The judgement of relevance between queries and documents can be subjective, so marginal difference between model evaluation could be ignored, while significant difference gives good signals of model capabilities.

### Forking & Sampling
#### Safety Level
<!-- scope: telescope -->
<!-- info: Select **one**: -->
- Safe to form and/or sample

#### Acceptable Sampling Method(s)
<!-- scope: periscope -->
<!-- info: Select **all applicable** acceptable methods to sample this
dataset: -->
- Cluster Sampling
- Haphazard Sampling
- Multi-stage sampling
- Random Sampling
- Retrospective Sampling
- Systematic Sampling
- Weighted Sampling
- Unsampled

#### Best Practice(s)
<!-- scope: microscope -->
<!-- info: Summarize the best practices for forking or sampling this dataset.

Use additional notes to capture any other relevant information or
considerations. -->
Although sampling is possible, we recommend not to do it because the size of BRIGHT is not very large.

#### Risk(s) and Mitigation(s)
<!-- scope: periscope -->
<!-- info: Fill this out if you selected "Conditionally safe to fork and/or
sample" or "Should not be forked and/or sampled":

Summarize known or residual risks associated with forking and sampling methods
when applied to the dataset.

Use additional notes to capture any other
relevant information or considerations. -->
N/A

#### Limitation(s) and Recommendation(s)
<!-- scope: microscope -->
<!-- info: Fill this out if you selected "Conditionally safe to fork and/or
sample" or "Should not be forked and/or sample":

Summarize the limitations that the dataset introduces when forking
or sampling the dataset and corresponding recommendations.

Use additional notes to capture any other relevant information or
considerations. -->
N/A

### Use in ML or AI Systems
#### Dataset Use(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable** -->
- Evaluation

#### Notable Feature(s)
<!-- scope: periscope -->
<!-- info: Describe any notable feature distributions or relationships between
individual instances made explicit.

Include links to servers where readers can explore the data on their own. -->

The intensive reasoning required to retrieve documents.

#### Usage Guideline(s)
<!-- scope: microscope -->
<!-- info: Summarize usage guidelines or policies that consumers should be
aware of.

Use additional notes to capture any other relevant information or
considerations. -->
**Usage Guidelines:** Follow [the tutorial](https://github.com/xlang-ai/BRIGHT) to evaluate retrieval systems.

**Approval Steps:** Steps are [here](https://github.com/xlang-ai/BRIGHT).

**Reviewer:** We authors review the dataset for publication.

#### Distribution(s)
<!-- scope: periscope -->
<!-- info: Describe the recommended splits and corresponding criteria.

Use additional notes to capture any other
relevant information or considerations. -->

The BRIGHT benchmark is for the purpose of evaluation, i.e., all data are in test set.

#### Known Correlation(s)
<!-- scope: microscope -->
<!-- info: Summarize any known correlations with
the indicated features in this dataset.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate for each known
correlation.) -->
`query`, `gold_ids`, `gold_ids_long`

**Description:** The documents corresponding to gold_ids or gold_ids_long are relevant to queries.

**Impact on dataset use:** It can help evaluate retrieval systems in more realistic scenarios.

**Risks from correlation:** The judgement of correlation is by real users, and can be subjective.

#### Split Statistics
<!-- scope: periscope -->
<!-- width: full -->
<!-- info: Provide the sizes of each split. As appropriate, provide any
descriptive statistics for features. -->

Dataset | # Q | # D | # D+ | Q.L. | D.L. | 
--- |-----|-----|------|------|------| 
Biology | 103 | 57,364 | 3.6 | 83.6 | 115.2 |
Earth Science | 118 | 122,388 | 7.7 | 132.4 | 113.3 |
Economics | 103 | 50,221 | 8.0 | 120.2 | 181.5 |
Psychology | 101 | 52,841 | 7.3 | 118.2 | 149.6 | 
Robotics | 101 | 62,198 | 5.5 | 120.6 | 818.9 |
Stack Overflow | 117 | 101,100 | 7.0 | 704.5 | 478.3 | 
Sustainable Living | 108 | 60,732 | 5.6 | 108.0 | 148.5 |
LeetCode | 142 | 413,932 | 1.8 | 483.1 | 497.5 |
Pony | 112 | 7,894 | 22.5 | 98.3 | 102.6 |
AoPS | 111 | 188,177 | 4.7 | 89.0 | 250.5 |
TheoremQA | 206 | 188,177 | 3.2 | 117.1 | 250.5

Data statistics of BRIGHT
For each dataset, we show the number of queries (# Q) and documents (# D), 
the average number of positive documents (# D+) per example, 
the average length of queries (Q.L.) and documents (D.L., measured by the GPT-2 tokenizer)

## Transformations
<!-- info: Fill this section if any transformations were applied in the
creation of your dataset. -->
### Synopsis
#### Transformation(s) Applied
<!-- scope: telescope -->
<!-- info: Select **all applicable** transformations
that were applied to the dataset. -->
- Data Aggregation

#### Field(s) Transformed
<!-- scope: periscope -->
<!-- info: Provide the fields in the dataset that
were transformed.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied. Include the data types to
which fields were transformed.) -->
**Transformation Type**

Field Name | Source & Target
--- | ---
gold_ids | links: gold_ids
gold_ids_long | links: gold_ids_long

#### Library(ies) and Method(s) Used
<!-- scope: microscope -->
<!-- info: Provide a description of the methods
used to transform or process the
dataset.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each transformation
type applied.) -->
**Transformation Type**

**Method:** We follow the links or tags to find relevant documents.

**Platforms, tools, or libraries:**
We do not leverage other platforms or tools in transformation

**Transformation Results:** We collect 1322 examples that can be used for evaluating retrievers.

### Breakdown of Transformations
<!-- info: Fill out relevant rows. -->
We find documents for all instances following the procedure above

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
The risk is that the relevance judgement is subjective.

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
We require human annotators to write down the judgement for relevance and reasoning steps.

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
None

#### Cleaning Mismatched Value(s)
<!-- scope: telescope -->
<!-- info: Which fields in the data were corrected
for mismatched values? -->
We select high-quality data instance from websites, so there is no further cleaning.

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: How were incorrect or mismatched
values cleaned? What other choices
were considered? -->
We follow links or tags in the websites.

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were incorrect or mismatched
values cleaned using this method (over
others)? Provide a comparative
analysis demonstrating before and
after values were cleaned. -->
We do not use incorrect or mismatched values.

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
M/A

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
The data and notes written down by annotators are reviewed

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
None

#### Anomalies
<!-- scope: telescope -->
<!-- info: How many anomalies or outliers were
detected?
If at all, how were detected anomalies
or outliers handled?
Why or why not? -->
We select data from websites, so no anomaly or outlier is excluded.

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What methods were used to detect
anomalies or outliers? -->
N/A

**Platforms, tools, or libraries**
N/A

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Provide a comparative analysis
demonstrating before and after
anomaly handling measures. -->
N/A

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
N/A

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
The data and notes written by annotators are reviewed.

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
N/A

#### Dimensionality Reduction
<!-- scope: telescope -->
<!-- info: How many original features were
collected and how many dimensions
were reduced? -->
N/A

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What methods were used to reduce the
dimensionality of the data? What other
choices were considered? -->
N/A

**Platforms, tools, or libraries**
N/A

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were features reduced using this
method (over others)? Provide
comparative charts showing before
and after dimensionality reduction
processes. -->
N/A

#### Residual & Other Risks
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
N/A

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
N/A

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
N/A

#### Joining Input Sources
<!-- scope: telescope -->
<!-- info: What were the distinct input sources that were joined? -->
We use StackExchange, LeetCode, TheoremQA and math competitions

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What are the shared columns of fields used to join these
sources? -->
They are independent splits, so no join is performed

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why were features joined using this
method over others?

Provide comparative charts showing
before and after dimensionality
reduction processes. -->
N/A

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
N/A

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
N/A

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
N/A

#### Redaction or Anonymization
<!-- scope: telescope -->
<!-- info: Which features were redacted or
anonymized? -->
N/A

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What methods were used to redact or
anonymize data? -->
N/A

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why was data redacted or anonymized
using this method over others? Provide
comparative charts showing before
and after redaction or anonymization
process. -->
N/A

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
N/A

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
N/A

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were
made? -->
N/A

#### Others (Please Specify)
<!-- scope: telescope -->
<!-- info: What was done? Which features or
fields were affected? -->
N/A

#### Method(s) Used
<!-- scope: periscope -->
<!-- info: What method were used? -->
N/A

#### Comparative Summary
<!-- scope: microscope -->
<!-- info: Why was this method used over
others? Provide comparative charts
showing before and after this
transformation. -->
N/A

#### Residual & Other Risk(s)
<!-- scope: telescope -->
<!-- info: What risks were introduced because of
this transformation? Which risks were
mitigated? -->
N/A

#### Human Oversight Measure(s)
<!-- scope: periscope -->
<!-- info: What human oversight measures,
including additional testing,
investigations and approvals were
taken due to this transformation? -->
N/A

#### Additional Considerations
<!-- scope: microscope -->
<!-- info: What additional considerations were made? -->
N/A

## Annotations & Labeling
<!-- info: Fill this section if any human or algorithmic annotation tasks were
performed in the creation of your dataset. -->
#### Annotation Workforce Type
<!-- scope: telescope -->
<!-- info: Select **all applicable** annotation
workforce types or methods used
to annotate the dataset: -->
- Human Annotations (Expert)
- Human Annotations (Non-Expert)

#### Annotation Characteristic(s)
<!-- scope: periscope -->
<!-- info: Describe relevant characteristics of annotations
as indicated. For quality metrics, consider
including accuracy, consensus accuracy, IRR,
XRR at the appropriate granularity (e.g. across
dataset, by annotator, by annotation, etc.).

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**Annotation Type** | **Number**
--- | ---
Total number of annotations | 1322

#### Annotation Description(s)
<!-- scope: microscope -->
<!-- info: Provide descriptions of the annotations
applied to the dataset. Include links
and indicate platforms, tools or libraries
used wherever possible.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->

**Description:** Description of annotations (labels, ratings) produced.
Include how this was created or authored.

We follow links/tags to find relevant documents

**Link:** N/A

**Platforms, tools, or libraries:**
N/A

#### Annotation Distribution(s)
<!-- scope: periscope -->
<!-- info: Provide a distribution of annotations for each
annotation or class of annotations using the
format below.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
Dataset | number | 
--- |--------|
Biology | 103    | 
Earth Science | 118    | 
Economics | 103    | 
Psychology | 101    | 
Robotics | 101    |
Stack Overflow | 117    | 101,100 | 7.0 | 704.5 | 478.3 | 
Sustainable Living | 108    | 60,732 | 5.6 | 108.0 | 148.5 |
LeetCode | 142    | 413,932 | 1.8 | 483.1 | 497.5 |
Pony | 112    | 7,894 | 22.5 | 98.3 | 102.6 |
AoPS | 111    | 188,177 | 4.7 | 89.0 | 250.5 |
TheoremQA | 206    | 188,177 | 3.2 | 117.1 | 250.5

Distribution of data splits in each domain

#### Annotation Task(s)
<!-- scope: microscope -->
<!-- info: Summarize each task type associated
with annotations in the dataset.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each task type.) -->
**(Task Type)**

**Task description & instructions:** 
In this section, we describe the instructions for annotators to collect data in BRIGHT.

StackExchange
1. Browse posts from the newest to the oldest.
2. Discard posts without an answer accepted by the user or obtains more than 5 votes

3. Discard answers of posts without URL links.
4. For each link in the answer, write down the answers to: (1). why are the document and the
post relevant; (2). what is the reasoning required to understand the relevance between the
post and the document. If there answers are not possible, discard the link.
5. Use LLMs (e.g., ChatGPT, Claude, etc.) to generate post key words, or use the post title to
search for web pages with large keyword or semantic overlap in Google. Search for at most
5 negative web pages per query.
6. Split every web page into small passages either by two newline symbols, "#" in markdonw
files or fixed-length tokens

TheoremQA

In TheoremQA, the main task for the annotator is to check if the GPT-4 rewritten questions are valid.
The specific instructions are as follows:
1. Read the rewritten question and determine if it is solvable.
2. If it is solvable, read the original question and solution, and determine if the rewritten
question is consistent with the original question. That is, the same reasoning steps and the
final answer should hold.
3. If it is also consistent, mark the question as valid, and make any minor edits to the problem
statement (e.g., to improve grammar or fluency) as you see fit.
4. If it is not solvable or not consistent, read the original question and solution, and correct the
rewritten question if possible. If not, then discard the problem.

AoPS
In AoPS, annotators are tasked to find questions from the AoPS Wiki and record the problems:
1. Browse through the AoPS Wiki and find topic/category pages (example 1, example 2).
2. Look through each page and find pages specific theorems or techniques that can be used
to solve problems. The page should link to at least two competition problems (example 1,
example 2).
3. Record the links of both the theorem/technique as well as the problem pages.
The annotators are assigned a category to look for theorems in to avoid overlaps, and the categories
are {algebra, geometry, calculus, probability, number theory, other}. After all links are
collected, we use a web scraper to collect the problem statement and solutions, and we manually
check the quality of the scraped data.

LeetCode
In LeetCode, annotators determine whether a question is grounded in real-world concepts. We give a
similar instruction to the annotator as to GPT-4:
1. Read the problem statement carefully.
2. Categorize the question into one of three categories:
• 0: The question is not grounded in any real-world concepts. The description only uses
coding-specific terms, such as "linked list", "binary search", "palindrome", "sorting",
etc..
• 1: The question is not grounded in any real-world concepts or real-world concepts that
are commonly used in the context of coding, such as needle in a haystack, strings/words,
or a spiral matrix.

• 2: The question is grounded in real-world concepts that are not commonly used in the
context of coding, such as building height, planting trees, or games. It may still uses
some code-specific terms to specify the data structure involved.

**Methods used:** Basically we follow links/tags to find documents

**Inter-rater adjudication policy:** Reviewers annotate where the pairing of queries and documents are not convincing.

**Golden questions:** N/A

### Human Annotators
<!-- info: Fill this section if human annotators were used. -->
#### Annotator Description(s)
<!-- scope: periscope -->
<!-- info: Provide a brief description for each annotator
pool performing the human annotation task.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**(Annotation Type)**

**Task type:** Annotate StackExchange data

**Number of unique annotators:** 3

**Expertise of annotators:** Both experts and non-experts

**Description of annotators:** PhD students in computer science, biology, environment, etc.

**Language distribution of annotators:** They all speak fluent English

**Geographic distribution of annotators:** They come from Asia

**Summary of annotation instructions:** Follow links to find documents with filtering

**Summary of gold questions:** N/A

**Annotation platforms:** Google sheets

**Additional Notes:** N/A

#### Annotator Task(s)
<!-- scope: microscope -->
<!-- info: Provide a brief description for each
annotator pool performing the human
annotation task.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**(Task Type)**

**Task description:** Annotate math and code data

**Task instructions:** Follow tags to find similar problems/questions

**Methods used:** Follow tags annotated by websites

**Inter-rater adjudication policy:** The data is reviewed

**Golden questions:** N/A

**Additional notes:** N/A

#### Language(s)
<!-- scope: telescope -->
<!-- info: Provide annotator distributions for
each annotation type.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and
complete the following for each
annotation type.) -->
**(Annotation Type)**

- 100% English

#### Location(s)
<!-- scope: periscope -->
<!-- info: Provide annotator distributions for each
annotation type.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each annotation type.) -->
**(Annotation Type)**

- Asia [50 %]
- US [50 %]

#### Gender(s)
<!-- scope: microscope -->
<!-- info: Provide annotator distributions for
each annotation type.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete
the following for each annotation
type.) -->
**(Annotation Type)**

- Male [80 %]
- Female [20 %]

## Validation Types
<!-- info: Fill this section if the data in the dataset was validated during
or after the creation of your dataset. -->
#### Method(s)
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
- Code/cross-reference Validation

#### Breakdown(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the fields and data
points that were validated.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
**(Validation Type)**

**Number of Data Points Validated:** 1322

**Fields Validated**

All fields in data are validated

#### Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of the methods used to
validate the dataset.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
**(Validation Type)**

**Method:** Describe the validation method here. Include links where
necessary.

We require annotators to write the logic to determine the relevance between queries and documents. The reviewers not only check the data, but also annotators' notes.

**Validation Results:** 

Over 90% of annotation passes peer review, and we discard the the rest part.

### Description of Human Validators
<!-- info: Fill this section if the dataset was validated using human
validators -->
#### Characteristic(s)
<!-- scope: periscope -->
<!-- info: Provide characteristics of the validator
pool(s). Use additional notes to capture any
other relevant information or considerations. -->
**(Validation Type)**
- Unique validators: 8
- Number of examples per validator: 300
- Average cost/task/validator: N/A
- Training provided: N
- Expertise required: N

#### Description(s)
<!-- scope: microscope -->
<!-- info: Provide a brief description of the validator
pool(s). Use additional notes to capture any
other relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each validator type.) -->
**(Validation Type)**

**Validator description:** Validators are domain experts, e.g., PhD students from the corresponding domains.

**Training provided:** We do not provide training, but verify that the annotators, reviewers are qualified

**Validator selection criteria:** We have a test containing verified examples. An annotator is qualified if they can work out these examples.

**Training provided:** N/A

#### Language(s)
<!-- scope: telescope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
**(Validation Type)**

- English [100 %]

#### Location(s)
<!-- scope: periscope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
**(Validation Type)**

- Asia [60 %]
- US [40 %]

#### Gender(s)
<!-- scope: microscope -->
<!-- info: Provide validator distributions.
Use additional notes to capture any other relevant information or
considerations.

(Usage Note: Duplicate and complete the following for each annotation type.)-->
**(Validation Type)**

- Male [70 %]
- Female [30 %]

## Sampling Methods
<!-- info: Fill out the following block if your dataset employs any sampling
methods. -->
#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used in the creation of this
dataset: -->
- Unsampled

#### Characteristic(s)
<!-- scope: periscope -->
<!-- info: Provide characteristics of each sampling
method used.

Use additional notes to capture any other
relevant information or considerations.

(Usage Note: Duplicate and complete the
following for each sampling method
used.) -->
N/A

#### Sampling Criteria
<!-- scope: microscope -->
<!-- info: Describe the criteria used to sample data from
upstream sources.

Use additional notes to capture any other
relevant information or considerations. -->
N/A

## Known Applications & Benchmarks
<!-- info: Fill out the following section if your dataset was primarily
created for use in AI or ML system(s) -->
#### ML Application(s)
<!-- scope: telescope -->
<!-- info: Provide a list of key ML tasks
that the dataset has been
used for.

Usage Note: Use comma-separated keywords. -->
Retrieval evaluation

#### Evaluation Result(s)
<!-- scope: periscope -->
<!-- info: Provide the evaluation results from
models that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
SFR-Embedding-Mistral 17.8

**Model Card:** https://huggingface.co/Salesforce/SFR-Embedding-Mistral/tree/main

Evaluation Results

- nDCG@10: 17.8

#### Evaluation Process(es)
<!-- scope: microscope -->
<!-- info: Provide a description of the evaluation process for
the model's overall performance or the
determination of how the dataset contributes to
the model's performance.

Use additional notes to capture any other relevant
information or considerations.

(Usage Note: Duplicate and complete the following
for each model and method used.) -->
We write python scripts to run retrieval models on BRIGHT.

#### Description(s) and Statistic(s)
<!-- scope: periscope -->
<!-- info: Provide a description of the model(s) and
task(s) that this dataset has been used
in.

Use additional notes to capture any
other relevant information or
considerations.

(Usage Note: Duplicate and complete the
following for each model.) -->
SFR-Embedding-Mistral

**Model Card:** https://huggingface.co/Salesforce/SFR-Embedding-Mistral/tree/main

**Model Description:** The best-class retrieval model trained from mistral-7b

- Model Size: 7.11B
- Model Weights: 7.11B
- Model Layers 32
- Latency: 2s

#### Expected Performance and Known Caveats
<!-- scope: microscope -->
<!-- info: Provide a description of the expected performance
and known caveats of the models for this dataset.

Use additional notes to capture any other relevant
information or considerations.

(Usage Note: Duplicate and complete the following
for each model.) -->
Claude-3 + BM25

**Expected Performance:** surpasses results obtained without using LLMs

**Known Caveats:** The inference of LLMs can be expensive

## Terms of Art
### Concepts and Definitions referenced in this Data Card
<!-- info: Use this space to include the expansions and definitions of any
acronyms, concepts, or terms of art used across the Data Card.
Use standard definitions where possible. Include the source of the definition
where indicated. If you are using an interpretation,
adaptation, or modification of the standard definition for the purposes of your
Data Card or dataset, include your interpretation as well. -->
#### BRIGHT
Definition: The name of this benchmark

Source: https://huggingface.co/datasets/xlangai/BRIGHT

Interpretation: N/A

## Reflections on Data
<!-- info: Use this space to include any additional information about the
dataset that has not been captured by the Data Card. For example,
does the dataset contain data that might be offensive, insulting, threatening,
or might otherwise cause anxiety? If so, please contact the appropriate parties
to mitigate any risks. -->

We believe that BRIGHT paves the way for future research on retrieval20
systems in more realistic and challenging settings.
