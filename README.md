# Wildfire Risk Detection Project 

This repository contains the complete codebase, reports, models, and documentation for our wildfire risk detection research project. The work was supported under the Mitacs program.  

## 📂 Repository Structure  

- **`/code/`**  
  - Source code for training and evaluation  
  - Downloaded Colab notebooks (offline versions)  
  - Utility scripts  

- **`/models/`**  
  - Trained model checkpoints  
  - Final weights used for experiments  

- **`/video_processing/`**  
  - The final script for video processing and result visualization 

- **`/reports/`**  
  - Weekly meeting reports (PDF format)  
  - Final project documentation
    
- **`/spexi_codebase/`**
  - provides the source code for the Spexi wildfire prediction and geolocation mapping pipeline

- **`/paper/`**  
  - Drafts and Final version of our paper  

- **`/dataset_samples/`**  
  - A few representative dataset images and annotations (not the full dataset)  
  - The dataset can be found in Roboflow universe, id: spexi-forest-fire-oblique-images
    https://universe.roboflow.com/search?q=spexi-forest-fire-oblique-images

- **`/log_rgb/`**  
  - Source code for LogRGB project [[Paper]](https://www.mdpi.com/2072-4292/17/9/1503)

## 👥 Contributors  

- [Supervisor] 
- [Victor Wu]  
- [Tracy Huang]  
- [Lesley Chen]  
- [Sujit Nashik]  
 
 
 ## 🙌 Reflection  

##### 👩‍💻Tracy
Working on this project has been both challenging and rewarding.  

- On the technical side, we explored advanced models like **Mask2Former**, and also used classic ones like **U-Net** for comparison.  This helped me better understand **image segmentation** and how it applies in real-world situations.
- We faced real challenges— **dataset imbalance**, **time-consuming annotation**, and **the complexity of high-resolution UAV imagery**. Dealing with **oblique images** also brought unexpected issues like distortion, occlusion, and lighting changes. These problems weren’t easy, but solving them taught me a lot and made our models stronger.
- Weekly discussions and collaboration helped shape the direction of the work and kept the focus on practical applications. 
- What made this project stand out for me was **the sense of purpose**. Forest fires are real, urgent problems. Knowing that this project might actually help with early detection made it more than just an academic exercise—it felt relevant. 
- This project has been a meaningful learning experience. I’ve grown not only in my AI and computer vision skills, but also in how I approach teamwork, communication, and persistence—especially when things got tough. There’s still plenty of room to improve, but I’m proud of what we’ve built so far.

##### David

Working on this project has been a valuable and rewarding research experience. The collaboration within our team played a key role in accelerating the research process and enabling us to develop better solutions. Each member’s expertise contributed to refining the methodology and strengthening the overall study. In addition, collaborating with external partners to obtain the drone data was essential for achieving the project’s objectives, as access to high-quality data directly influenced the accuracy and reliability of our findings.  

Through this project, I dedicated my efforts to ensure the best possible outcomes, not only by contributing to technical aspects but also by supporting the integration of ideas and resources. I have learned the importance of both internal teamwork and external collaboration in driving research success.  

##### Lesley
Through this valuable project, I learned both technical skills and research practices.

I worked on detecting water areas and identifying living trees. U-Net was my starting point, and I compared backbones such as ResNet50 and DeepLab. This showed me how architecture choice can affect performance. I also experimented with optimizers and learning rates to fine-tune results. For alive tree, I tested traditional methods like RGB-based indices and selected the one that fit our dataset best.

I also explored how to convert 2D image predictions into geometric maps. This step helped me connect computer vision outputs with spatial analysis and see their value in forest monitoring.

Collaboration played a big role. Weekly discussions with teammates and feedback from our professor guided my work and kept me focused on practical applications.

What motivated me most was the project’s purpose. Forest health and wildfire prevention are urgent issues. Knowing that our work could make a difference made the effort feel meaningful.

This experience strengthened my technical skills in computer vision and geospatial analysis. It also taught me persistence, adaptability, and teamwork. There is still room to improve, but I’m proud of what we achieved together.

##### Victor

This project gave me the chance to apply my skills in machine learning and software engineering to a real-world challenge.
	•	On the technical side, I focused on building and refining the geolocation pipeline, which mapped pixel detections in UAV imagery to GPS coordinates. This required working with geometry-based calculations, integrating metadata such as pitch, heading, and altitude, and improving results with offset correction models.
	•	I also contributed to the evaluation framework, developing metrics like Prediction Cluster Compactness and integrating tools for analyzing geolocation errors. This helped us measure not just accuracy, but also consistency across multiple UAV views.
	•	Working with both nadir and oblique imagery gave me a deeper understanding of the trade-offs between perspectives, particularly the advantages of oblique images in identifying fuels and debris that nadir views often miss.
	•	Collaborating with the team on weekly discussions and experiments was a key part of keeping the project moving forward. The iterative process of testing, debugging, and refining our methods taught me how to balance research creativity with engineering discipline.

What stood out most was the practical significance of the work. Wildfire detection and risk assessment are urgent challenges, and it felt meaningful to see how UAVs and computer vision could contribute to solutions. Through this project, I strengthened not only my technical foundation in computer vision, machine learning, and UAV systems, but also my ability to collaborate in a research setting and communicate complex results clearly.

##### Sujit

Throughout this project, I had the opportunity to engage with a wide range of new tools and technologies that expanded my skill set. I began by exploring computer vision frameworks, experimenting with different models and training parameters to analyze how various configurations affected performance. I also worked with 3D visualization libraries, which, although new and at times challenging, were highly rewarding. These tools opened new perspectives on data representation and deepened my understanding of how results can be communicated effectively. The geospatial mapping pixel project work was entirely new to me, teaching me to quickly adapt to unfamiliar technologies and find solutions to challenging problems.

Overall, I dedicated myself fully to this project, from learning new technologies to innovating and implementing solutions. I'm happy to have contributed meaningfully and proud of the progress achieved.

