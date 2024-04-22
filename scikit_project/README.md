
# Exploración y Despliegue a producción con Scikit-Learn

Este proyecto se ha implementado utilizando la potente biblioteca de machine learning, scikit-learn para hacer una exploración de diferentes modelos de Machine Learning, autamatizado para buscar el más eficaz en nuestros diferentes conjuntos de datos. 

Para garantizar un entorno de desarrollo limpio y reproducible, 
hemos "partido" este proyecto en 3 secciones principales:

1) Entorno Virtual y Esquema de el proyecto
2) Técnicas de Machine Learning Con Scikit-Learning
3) Despliegue en producción con Flask



## Instalación

Claro, aquí está la sección actualizada del `readme.md` con las instrucciones para crear un entorno virtual con Conda o con el entorno virtual por defecto de Python, y también menciona que las bibliotecas necesarias están listadas en `requirements.txt`:

---

## Instalación

Para clonar este repositorio en tu máquina local, sigue estos pasos:

1. Abre tu terminal.

2. Utiliza el comando `git clone` seguido de la URL del repositorio para clonarlo en tu máquina local. Ejecuta el siguiente comando:

    ```bash
    git clone https://github.com/EdwLearn/scikit-Learn
    ```

3. Abre una terminal y navega hasta el directorio del repositorio clonado.

4. Crea un entorno virtual usando Conda o el entorno virtual por defecto de Python:

    - **Conda:**
    
        ```bash
        conda create --name scikit python=3.9
        ```
    
    - **Entorno virtual de Python:**
    
        ```bash
        python3 -m venv scikit
        ```

5. Activa el entorno virtual recién creado:

    - **Conda:**
    
        ```bash
        conda activate scikit
        ```

    - **Entorno virtual de Python (en Linux/Mac):**
    
        ```bash
        source scikit/bin/activate
        ```

        **Entorno virtual de Python (en Windows):**
    
        ```bash
        .\scikit\Scripts\activate
        ```

6. Instala las dependencias necesarias utilizando `pip` y el archivo `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

    Esto asegurará que todas las bibliotecas necesarias estén instaladas correctamente en tu entorno virtual.

## Ejecutar el Proyecto

Una vez que hayas configurado el entorno virtual y hayas instalado las dependencias, puedes ejecutar el proyecto ejecutando el siguiente comando:

```bash
python main.py
```

Este comando iniciará la ejecución del proyecto y podrás ver los resultados en la consola.

Además, si deseas ver el resultado en el servidor local, visita http://localhost:8080/predict en tu navegador web una vez que el proyecto esté en ejecución.
