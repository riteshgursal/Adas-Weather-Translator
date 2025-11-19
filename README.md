<h1>🌧️ ADAS Adverse Weather Domain Translation Pipeline</h1>

This project establishes a complete, modular pipeline using Generative AI concepts to synthesize adverse weather data (e.g., fog, rain) onto clear road scenes. This technique is critical for augmenting training data and improving the robustness of Autonomous Driving (ADAS) systems.

The focus of this repository is on demonstrating a clean, runnable workflow and modularity, with placeholder code where the complex PyTorch model would be implemented.

**🌟 Key Features**

**Modular Design:** Code is cleanly separated into main.py (CLI), model_cyclegan.py (data loading/model logic), and requirements.txt.

**Data Isolation:** Uses separate, flat directories (data_clear/, data_fog/) for unpaired image loading.

**Functional Pipeline:** CLI commands (train, translate) execute successfully, confirming all file paths and environment setups are correct.

**Informative Demo:** The translate command generates an image clearly indicating that the pipeline is ready for the deep learning component.

---

**📁 Project Structure**

adas_weather_translator/
The **adas_weather_translator project** directory is organized for modularity, containing two main scripts: **main.py**, which serves as the command-line interface and controls the execution flow, and **model_cyclegan.py**, which handles the data loading via the UnpairedDataset and holds the model logic placeholders. The project relies on packages listed in **requirements.txt** and uses **.gitignore** to exclude environment files (venv/, saved_weights/, etc.) from version control. Input data is separated into two flat folders: **data_clear/** for clear road images (Domain A) and **data_fog/** for adverse weather images (Domain B), with the final translated demo output stored in the **saved_weights/** directory.

---

**🛠️ How to Execute**
**1. Setup Environment**
Clone the Repository and navigate into the project directory:

```
git clone https://github.com/riteshgursal/Adas-Weather-Translator`
```
cd adas_weather_translator
```
Create and Activate Environment:
```
```
python -m venv venv
venv\Scripts\activate`  # Windows
# source venv/bin/activate # macOS/Linux
```

**Install Dependencies:**

```
pip install -r requirements.txt`
```
---

**2. Prepare Data**

The project requires unpaired images in two domains.

Place clear road images in the data_clear/ folder.

Place fog/adverse weather images in the data_fog/ folder. (The script will automatically generate a placeholder input image if needed.)

---

**3. Run Pipeline Verification**

Run the two core commands to verify the setup.

**A. Training Simulation**
This confirms the data loader and training loop are ready for the PyTorch model.

```
python main.py train --epochs 10`
```

**B. Translation Demo**
This runs the full inference pipeline and saves the informative placeholder image.

```
python main.py translate`
```
The result, confirming pipeline functionality, will be saved to saved_weights/translated_sample.jpg.

---


**💡 Future Work**

This project is structured for immediate scalability. The natural next steps include:

Model Implementation: Define and integrate the full PyTorch ResNet Generator and PatchGAN Discriminator models into model_cyclegan.py.

Training Loop: Implement the Adversarial, Cycle Consistency, and Identity loss functions required for the CycleGAN optimization.

Real-World Data: Train the model using public datasets like Cityscapes or KITTI to achieve genuine domain translation.

---
