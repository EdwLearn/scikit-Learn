# Exploration and Production Deployment with Scikit-Learn

This project has been implemented using the powerful machine learning library, scikit-learn, to explore different machine learning models, automated to find the most effective one across our various datasets.

To ensure a clean and reproducible development environment, we have "partitioned" this project into 3 main sections:

1) Virtual Environment and Project Structure
2) Machine Learning Techniques with Scikit-Learn
3) Production Deployment with Flask

## Roadmap Project Organization

------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Data for this project
    │
    ├── files               <- An Explained Data for this project
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── scripts            <- Python Scripts, same as notebooks but with another format
    │
    │
    ├── tools              <- Script tool for this project, read csv, create the analysis, for automatization
    │
    │
    ├── environment.yml    <- If you want to install the same Environment 
    
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── main.py            <- Run main.py in console to run the analysis
    │
    └── server.py          <- Run server.py in console to run the local server

## Installation and Deployment with Flask

To clone this repository to your local machine, follow these steps:

1. Open your terminal.

2. Use the `git clone` command followed by the repository URL to clone it to your local machine. Run the following command:

    ```bash
    git clone https://github.com/EdwLearn/scikit-Learn
    ```

3. Open a terminal and navigate to the directory of the cloned repository.

4. Create a virtual environment using Conda or Python's default virtual environment:

    - **Conda:**
    
        ```bash
        conda create --name scikit python=3.9
        ```
    
    - **Python's Virtual Environment:**
    
        ```bash
        python3 -m venv scikit
        ```

5. Activate the newly created virtual environment:

    - **Conda:**
    
        ```bash
        conda activate scikit
        ```

    - **Python's Virtual Environment (on Linux/Mac):**
    
        ```bash
        source scikit/bin/activate
        ```

        **Python's Virtual Environment (on Windows):**
    
        ```bash
        .\scikit\Scripts\activate
        ```

6. Install the necessary dependencies using `pip` and the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

    This will ensure that all required libraries are properly installed in your virtual environment.

## Running the Project

Once you have set up the virtual environment and installed the dependencies, you can run the project by executing the following command:

```bash
python server.py
```

This command will start the project execution, and you can view the results in the console.

Additionally, if you wish to view the result on the local server, visit http://localhost:8080/predict in your web browser once the project is running.

