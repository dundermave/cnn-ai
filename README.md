# Project Setup

## Setting Up the Environment

1. **Clone the Repository**  
   Start by cloning the project repository to your local machine:  
   ```bash
   git clone https://github.com/dundermave/cnn-ai.git

2. **Download Anaconda**  
   Download and install Anaconda from the official website: [https://www.anaconda.com/](https://www.anaconda.com/).

3. **Set up the environment**  
   Navigate to the project directory using Anaconda Prompt and run the following command:
   ```bash
   conda env create -f environment.yml
   conda activate cnn-env
---

# Dataset Setup

## Steps to Download the Dataset

1. **Run the Data Loader**  
   Open the file `data_loader.ipynb` in Jupyter Notebook or any compatible IDE.

2. **Execute All Cells**  
   Run all cells in the notebook sequentially to download the **ChestMNIST** dataset. The notebook is configured to fetch the **ChestMNIST** data from the **MedMNIST database** automatically.

3. **Dataset Splitting**  
   - The dataset will be split into three parts by default:
     - **Training** data
     - **Validation** data
     - **Testing** data
   - These splits are automatically created during the execution of the notebook.

4. **File Storage**  
   The downloaded and processed dataset files will be stored in the `/data` directory within the project folder.


  Note: You only need to run this process once unless the dataset needs to be refreshed or updated.
  
---


