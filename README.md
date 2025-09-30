# Finatars

**A Knowledge Graph + Multi-LLM Q/A System for complex and inference-based financial questions**

## Running Instructions:

**Note:** do not install autogen with pip. We are running customized llamaindex and autogen, some of the source code is modified. 

#### Instructions to run Updated System:
1. Create a new environment, and install *requirements.txt*
        
        pip install -r requirements.txt
2. Add a *.env* file with your API_KEY within *finatars* main directory.

    **.env** file containg two keys: 
    - OPENAI_API_KEY = _your-key_
    - AUTOGEN_USE_DOCKER = 0

3. If you want to run the system directly without UI, follow the below steps. Else skip this section and directly check Streamlit UI section
   - Run *finatars_pipeine/run.py* from its directory after modifying the necessary parameters within the file (towards the end). This part is currently commented, please uncomment section **For running directly** (line 66)

4. Explore *finatars_pipeine/experimentation.ipynb* to test capabilities and check the format of "answer" from Multi-LLM System.

#### Streamlit UI:
1. **For running UI**, update the following paths inside *path_config.json* file 
- triplet_data_path - absolute path of *finatars/data_processing/triplet_extraction/FinalData*
- llm_config_file - absolute path of *finatars/finatars_pipeline/llm_config.json*

2. **Run command:** 
        
        streamlit run finatars_pipeine/streamlit_run.py