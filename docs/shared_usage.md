# Usage Guidelines

Regardless of the versions, both builds share the same core architecture and functionality. Use this chapter as the operating manual.

## 4.1 Prompting the requirements

Before letting the agent autonomously handle your request, there are several steps that
decides the quality of downstream execution:

- **Crafting the Initial Prompt**  
  Imagine you are a manager delegating tasks to a team member. Your initial prompt should
  clearly mention the specific goals you want the agent to achieve. Include all relevant
  details such as disease names, protein targets, mechanisms of action, drug candidates, or
  any other information necessary for the task, and align with your preferences. 

- **Planning Phase**  
  After you submit your prompt, the planning agent will take some time to process it using supporting information from SOPs and episodic memory. During this phase, the agent generates an initial plan
  outlining a potential approach to fulfill your request.

- **Review and Refinement**  
  Carefully review the proposed plan. This step is crucial: use your domain expertise, continue refining
  the plan through additional prompts until you feel satisfied with the
  approach. The success of the entire workflow largely depends on the quality of this
  initial planning stage.

- **Approval and Execution**  
  Once you are satisfied with the plan, type **"Approved"** (or a similar confirmation). At
  At this point, the supervisor agent will take over to execute the full
  plan autonomously by delegating sub-tasks to specialized agents. The processing time can vary depending on task complexity. For
  For example, the complete pipeline demonstrated in the demo may take approximately 25–35
  minutes.

- **Performance Considerations**  
  Execution will be significantly faster if you already have knowledge graph files
  available. If the system needs to generate the knowledge graphs on the fly, the process
  will naturally take longer.


## 4.2 Working With Files

1. **Uploading**

- Drag-and-drop files into the upload widget or select them via the file picker.
- For the knowledge graph file, we are using the [Knowledge Graph Generator](https://github.com/Fraunhofer-ITMP/kgg) package from Fraunhofer-ITMP. The acceptable KG file is the pickle file of the knowledge graph object.
- You can create such a pickle file by using the template below:

``` python
from kgg.src.kgg_api import createKG
import pickle

kg = createKG()

kg_path = "data/kg_file.pkl"
with open(kg_path, 'wb') as f:
   pickle.dump(kg, f)
```

2. **Listing & content**

- The sidebar shows every conversation you own (plus demo conversation).
- Click on one conversation, and you will see all the files associated with it.
- Clicking a filename issues a download file to your device.
- Deleting a conversation removes *all* uploads and outputs for that conversation. Back up anything important first—the action cannot be undone.

## 4.3 Episodic Learning

- **What it is**  
  A Chroma store captures successful planning decompositions so the next run can reuse proven strategies.
- **How it works**  
  Every conversation can be extracted into an “episode” with *Task* and *Decomposition* patterns. When enabled, the planning agent automatically pulls the two most relevant examples into its system prompt.
- **Extraction trigger**  
  The extraction process happens only for the current conversation you are in, and only when you click the `Extract Learning` button in the UI.