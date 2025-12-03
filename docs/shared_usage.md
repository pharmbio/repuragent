# Usage Guidelines

Regardless of versions, both builds present the similar Gradio UI and LangGraph architecrture. Use this chapter as the operating manual once your deployment is running.

## 4.1 Prompting the requirements
Before letting the agent autonomously handle your request, there are several steps that decides the quality of downstream exection: 

- **Crafting the Initial Prompt:** Imagine you are a manager delegating tasks to a team member. Your initial prompt should clearly mention the specific goals you want the agent to achieve. Include all relevant details such as disease names, protein targets, mechanisms of action, drug candidates, or any other information necessary for the task.

- **Planning Phase:** After submitting your prompt, the Planning Agent will take some time to process it using SOPs and episodic memory. During this phase, the agent generates an initial plan outlining a potential approach to fulfill your request.

- **Review and Refinement:** Carefully review the proposed plan. This step is crucial: continue iterating and refining the plan through additional prompts until you feel confident and satisfied with the approach. The success of the entire workflow largely depends on the quality of this initial planning stage.

- **Approval and Execution:** Once you are satisfied with the plan, type **"Approved"** (or a similar confirmation). At this point, the Supervisor Agent and its sub-agents will take over to execute the full plan autonomously. The processing time can vary depending on task complexity. For example, the complete pipeline demonstrated in the demo may take approximately 25–35 minutes.

- **Performance Considerations:** Execution will be significantly faster if you already have knowledge graph files available. If the system needs to generate the knowledge graphs on-the-fly, the process will naturally take longer.


## 4.2 Working With Files

1. **Uploading**
   - Drag-and-drop files into the upload widget or select them via the file picker.
   - For knowledge graph file, we are using [knowledge graph generator](https://github.com/Fraunhofer-ITMP/kgg) package from Fraunhofer-ITMP. The acceptable KG file is the pickle file of the knowledge graph oject in the package.
   - You can create such pickle file by using the below pattern:
    ```python
   from kgg.src.kgg_api import createKG
   import pickle

   kg = createKG()
   
   kg_path = "data/kg_file.pkl"
   with open(kg_path, 'wb') as f:
      pickle.dump(kg, f)
   ```

2. **Listing & content**
   - The sidebar shows every conversation you own (plus demo threads on the web edition).
   - Click on one conversation, you will see all the files associated with it.
   - Clicking a filename issues a download file to your device.
   - Deleting a conversation removes *all* uploads and outputs for that conversation. Back up anything important first—the action cannot be undone.

## 4.3 Episodic Learning UI

- **What it is** – LangMem + Chroma capture successful planning decompositions so the next run can reuse proven strategies.
- **How it works** – every conversation can be extracted into an “episode” with task and decomposition patterns. When enabled, the planning agent automatically pulls the two most relevant examples into its system prompt.
- The extraction process only happens with current conversation that you are in, only when you click `Extract Learning` button on the UI. 
