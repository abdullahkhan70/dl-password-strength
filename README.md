# **DL Strength Password**

## <u>**Project Scope**</u>

This project is based upon one of the [Kaggle's Dataset](https://www.kaggle.com/datasets/utkarshx27/passwords). In this project, we basically train our model on the basis of the given passwords, and then we are trying to find out the strength (whether your password is weak or strong) on the basis of the password as input. So, for this purpose we train our Neural Network based model using Pytorch.

## <u>**Requirements**</u>
In order to run this project on your system, you need to install the following things.

|   **Packages**  |    **Versions**    |
|   :---          |    ---:            |
|   [Python](https://www.python.org/downloads/)        |    3.11.5          |
|   [VSCode](https://code.visualstudio.com/download) (for IDE)       |    Download latest version |

You need to install all the required packages related to this project through the requirements.txt file. Here is the command for it.

```bash
pip install -r requirements.txt
```

After installed all the required packages, then you need to run the following commands.

```bash
python src/components/data_ingestion.py
```

The above command will do the Data Processing, Feature Engineering, and Model training stuff. It will create a folder called <b>Artifacts</b>. And inside this folder we will get three major files like:

-  **model.pt** 
-  **svd.pkl** 
-  **tfidf_vectorizer.pkl** 
-  **train_preprocessor.pkl**

### <u>**Note**</u>
If you are facing any issue during running this project, then you can ask any question related to it through our [issues](https://github.com/abdullahkhan70/dl-password-strength/issues) section.

## <u>**Web App**</u>

We have made our own custom web app for demonstrating our model for generating the results through it. So, for run the web app on your system you need to enter the following command:

```bash
python app.py
```

The above command will run your web app as localhost, and then you can even try the model through the web app.

## <u>**License**</u>

This project is under the [MIT](https://choosealicense.com/licenses/mit/) License. You can check out for further inquires.

## <u>**Other Solution**</u>

We have made another solution using Machine Learning algorithms only. Here is the link of our [Github Project](https://github.com/abdullahkhan70/ml-password-strength).

## <u>**Reference**</u>

**Special thanks to Efim Polianskii, and here are his social links**

[Github](https://github.com/efimpolianskii)